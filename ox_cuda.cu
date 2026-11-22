/*
 * OX-CUDA — GPU-стадия компрессора OX (формат v2).
 *
 * Параллелизация: каждый поток токенов режется на независимые блоки
 * (по умолчанию 32 КБ) с ОДНОЙ общей таблицей частот на поток:
 *   1. k_hist  — гистограмма: нить на блок, локальный cnt[256] + atomicAdd;
 *   2. k_norm  — нормализация частот к сумме T (одна нить);
 *   3. k_enc   — rANS-кодирование: нить на блок (state BE + reversed emission);
 *   4. k_compact — уплотнение слотов в компактный bodies (блок CUDA на блок OX).
 * Декодирование: k_dec — нить на блок, таблица/LUT читаются из global.
 *
 * Раскладка body блока совпадает с CPU-версией (ox.c), поэтому выход GPU
 * декодируется CPU и наоборот.
 */
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define SCALE   12
#define T       (1u << SCALE)
#define RANS_L  (1u << 23)

extern "C" {

int ox_cuda_available(void) {
    int c = 0;
    return (cudaGetDeviceCount(&c) == cudaSuccess && c > 0) ? 1 : 0;
}

const char *ox_cuda_device_name(void) {
    static char name[256] = "";
    cudaDeviceProp p;
    if (cudaGetDeviceProperties(&p, 0) == cudaSuccess) {
        snprintf(name, sizeof(name) - 1, "%s", p.name);
        return name;
    }
    return "";
}

} /* extern "C" */

/* -------------------------------------------------------------- ядра --- */
__global__ void k_hist(const uint8_t *d, uint32_t n, uint32_t chunk,
                       uint32_t *gcnt) {
    uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t nch = (n + chunk - 1) / chunk;
    if (c >= nch) return;
    uint32_t lo = c * chunk, hi = min(n, lo + chunk);
    uint32_t cnt[256];
    for (int i = 0; i < 256; i++) cnt[i] = 0;
    for (uint32_t i = lo; i < hi; i++) cnt[d[i]]++;
    for (int i = 0; i < 256; i++)
        if (cnt[i]) atomicAdd(&gcnt[i], cnt[i]);
}

/* Одна нить: нормализация к сумме T, freq >= 1 у встреченных символов. */
__global__ void k_norm(const uint32_t *cnt, uint32_t n,
                       uint16_t *freq, uint32_t *cum) {
    uint32_t sum = 0;
    for (int i = 0; i < 256; i++) {
        uint32_t f = 0;
        if (cnt[i]) {
            f = (uint32_t)(((uint64_t)cnt[i] * T) / n);
            if (!f) f = 1;
        }
        freq[i] = (uint16_t)f;
        sum += f;
    }
    while (sum != T) {
        int bi = 0;
        if (sum > T) {
            uint32_t bf = 1;
            for (int i = 0; i < 256; i++)
                if (freq[i] > bf) { bf = freq[i]; bi = i; }
            freq[bi]--; sum--;
        } else {
            uint32_t bf = 0;
            for (int i = 0; i < 256; i++)
                if (freq[i] > bf) { bf = freq[i]; bi = i; }
            freq[bi]++; sum++;
        }
    }
    cum[0] = 0;
    for (int i = 0; i < 256; i++) cum[i + 1] = cum[i] + freq[i];
}

/* Нить на блок: rANS назад, emission в слот, reverse на месте.
 * Слот: [u32 state BE][emission], slotcap = 4 + 2*chunk. */
__global__ void k_enc(const uint8_t *d, uint32_t n, uint32_t chunk,
                      const uint16_t *freq, const uint32_t *cum,
                      uint8_t *slots, uint32_t slotcap, uint32_t *blens) {
    uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t nch = (n + chunk - 1) / chunk;
    if (c >= nch) return;
    uint32_t lo = c * chunk, hi = min(n, lo + chunk);

    uint16_t lf[256];
    uint32_t lc[257];
    for (int i = 0; i < 256; i++) lf[i] = freq[i];
    for (int i = 0; i < 257; i++) lc[i] = cum[i];

    uint8_t *sl = slots + (size_t)c * slotcap;
    uint8_t *tmp = sl + 4;
    uint32_t x = RANS_L, tp = 0;
    for (uint32_t i = hi; i-- > lo; ) {
        uint8_t s = d[i];
        uint32_t f = lf[s];
        uint32_t x_max = ((RANS_L >> SCALE) << 8) * f;
        while (x >= x_max) { tmp[tp++] = (uint8_t)(x & 0xFF); x >>= 8; }
        x = ((x / f) << SCALE) + (x % f) + lc[s];
    }
    sl[0] = (uint8_t)(x >> 24); sl[1] = (uint8_t)(x >> 16);
    sl[2] = (uint8_t)(x >> 8);  sl[3] = (uint8_t)x;
    for (uint32_t i = 0; i < tp / 2; i++) {
        uint8_t t = tmp[i]; tmp[i] = tmp[tp - 1 - i]; tmp[tp - 1 - i] = t;
    }
    blens[c] = 4 + tp;
}

/* Блок CUDA нитей на блок OX: coalesced-копия слота в компактный bodies. */
__global__ void k_compact(const uint8_t *slots, uint32_t slotcap,
                          const uint32_t *blens, const uint32_t *offs,
                          uint8_t *out) {
    uint32_t c = blockIdx.x;
    const uint8_t *src = slots + (size_t)c * slotcap;
    uint8_t *dst = out + offs[c];
    uint32_t len = blens[c];
    for (uint32_t i = threadIdx.x; i < len; i += blockDim.x)
        dst[i] = src[i];
}

/* Нить на блок: rANS-декодирование по общей таблице. */
__global__ void k_dec(const uint8_t *bodies, const uint32_t *offs,
                      const uint32_t *blens,
                      uint32_t n, uint32_t chunk,
                      const uint16_t *freq, const uint32_t *cum,
                      const uint8_t *lut, uint8_t *out) {
    uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t nch = (n + chunk - 1) / chunk;
    if (c >= nch) return;
    uint32_t lo = c * chunk, hi = min(n, lo + chunk);

    uint16_t lf[256];
    uint32_t lc[257];
    for (int i = 0; i < 256; i++) lf[i] = freq[i];
    for (int i = 0; i < 257; i++) lc[i] = cum[i];

    const uint8_t *body = bodies + offs[c];
    uint32_t blen = blens[c];
    uint32_t x = ((uint32_t)body[0] << 24) | ((uint32_t)body[1] << 16) |
                 ((uint32_t)body[2] << 8) | body[3];
    uint32_t pos = 4;
    uint8_t *o = out + lo;
    for (uint32_t i = 0; i < hi - lo; i++) {
        uint32_t slot = x & (T - 1);
        uint8_t s = lut[slot];
        o[i] = s;
        x = lf[s] * (x >> SCALE) + slot - lc[s];
        while (x < RANS_L) {
            x = (x << 8) | (pos < blen ? body[pos] : 0);
            pos++;
        }
    }
}

/* ------------------------------------------------------ host-обёртки --- */
#define CK(x) do { \
        cudaError_t e_ = (x); \
        if (e_ != cudaSuccess) { \
            fprintf(stderr, "CUDA: %s (%s:%d)\n", \
                    cudaGetErrorString(e_), __FILE__, __LINE__); \
            goto fail; \
        } \
    } while (0)

extern "C" int ox_cuda_enc_stream(const uint8_t *d, uint32_t n, uint32_t chunk,
                                  uint16_t *hfreq, uint32_t *hblens,
                                  uint8_t *hbodies, uint32_t *bodies_n) {
    memset(hfreq, 0, 256 * sizeof(uint16_t));
    *bodies_n = 0;
    if (!n) return 0;
    uint32_t nch = (n + chunk - 1) / chunk;
    uint32_t slotcap = 4 + 2 * chunk;

    uint8_t *dd = 0, *slots = 0, *dbodies = 0;
    uint32_t *cnt = 0, *cum = 0, *blens = 0, *offs = 0;
    uint16_t *freq = 0;
    uint32_t *hoffs = 0;
    uint32_t off = 0;
    int rc = -1;

    CK(cudaMalloc(&dd, n));
    CK(cudaMalloc(&slots, (size_t)nch * slotcap));
    CK(cudaMalloc(&dbodies, (size_t)nch * slotcap));
    CK(cudaMalloc(&cnt, 256 * 4));
    CK(cudaMalloc(&cum, 257 * 4));
    CK(cudaMalloc(&freq, 256 * 2));
    CK(cudaMalloc(&blens, nch * 4));
    CK(cudaMalloc(&offs, nch * 4));
    CK(cudaMemset(cnt, 0, 256 * 4));
    CK(cudaMemcpy(dd, d, n, cudaMemcpyHostToDevice));

    k_hist<<<(nch + 127) / 128, 128>>>(dd, n, chunk, cnt);
    CK(cudaGetLastError());
    k_norm<<<1, 1>>>(cnt, n, freq, cum);
    CK(cudaGetLastError());
    k_enc<<<(nch + 127) / 128, 128>>>(dd, n, chunk, freq, cum,
                                      slots, slotcap, blens);
    CK(cudaGetLastError());
    CK(cudaMemcpy(hblens, blens, nch * 4, cudaMemcpyDeviceToHost));

    hoffs = (uint32_t *)malloc(nch * 4);
    if (!hoffs) goto fail;
    for (uint32_t c = 0; c < nch; c++) { hoffs[c] = off; off += hblens[c]; }
    CK(cudaMemcpy(offs, hoffs, nch * 4, cudaMemcpyHostToDevice));
    k_compact<<<nch, 256>>>(slots, slotcap, blens, offs, dbodies);
    CK(cudaGetLastError());
    CK(cudaMemcpy(hbodies, dbodies, off, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hfreq, freq, 256 * 2, cudaMemcpyDeviceToHost));
    *bodies_n = off;
    rc = 0;
fail:
    cudaFree(dd); cudaFree(slots); cudaFree(dbodies);
    cudaFree(cnt); cudaFree(cum); cudaFree(freq);
    cudaFree(blens); cudaFree(offs);
    free(hoffs);
    return rc;
}

extern "C" int ox_cuda_dec_stream(const uint8_t *bodies, const uint32_t *hblens,
                                  uint32_t n, uint32_t chunk,
                                  const uint16_t *hfreq, uint8_t *out) {
    if (!n) return 0;
    uint32_t nch = (n + chunk - 1) / chunk;

    uint32_t cum[257];
    cum[0] = 0;
    for (int i = 0; i < 256; i++) cum[i + 1] = cum[i] + hfreq[i];
    uint8_t *hlut = (uint8_t *)malloc(T);
    uint32_t *hoffs = (uint32_t *)malloc(nch * 4);
    uint32_t off = 0;
    if (!hlut || !hoffs) { free(hlut); free(hoffs); return -1; }
    for (int s = 0; s < 256; s++)
        for (uint32_t k = cum[s]; k < cum[s + 1]; k++) hlut[k] = (uint8_t)s;
    for (uint32_t c = 0; c < nch; c++) { hoffs[c] = off; off += hblens[c]; }

    uint8_t *dbodies = 0, *dout = 0, *lut = 0;
    uint32_t *dcum = 0, *doffs = 0, *dblens = 0;
    uint16_t *dfreq = 0;
    int rc = -1;

    CK(cudaMalloc(&dbodies, off));
    CK(cudaMalloc(&dout, n));
    CK(cudaMalloc(&lut, T));
    CK(cudaMalloc(&dcum, 257 * 4));
    CK(cudaMalloc(&doffs, nch * 4));
    CK(cudaMalloc(&dblens, nch * 4));
    CK(cudaMalloc(&dfreq, 256 * 2));
    CK(cudaMemcpy(dbodies, bodies, off, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(doffs, hoffs, nch * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dblens, hblens, nch * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(lut, hlut, T, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dcum, cum, 257 * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dfreq, hfreq, 256 * 2, cudaMemcpyHostToDevice));

    k_dec<<<(nch + 127) / 128, 128>>>(dbodies, doffs, dblens, n, chunk,
                                      dfreq, dcum, lut, dout);
    CK(cudaGetLastError());
    CK(cudaMemcpy(out, dout, n, cudaMemcpyDeviceToHost));
    rc = 0;
fail:
    cudaFree(dbodies); cudaFree(dout); cudaFree(lut);
    cudaFree(dcum); cudaFree(doffs); cudaFree(dblens); cudaFree(dfreq);
    free(hlut); free(hoffs);
    return rc;
}
