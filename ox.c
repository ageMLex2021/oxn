/*
 * OX (objectx) — промышленный компрессор: LZ77 + статический rANS.
 *
 * Конвейер: LZ77 (хэш-цепочки + ленивое сопоставление) -> 5 потоков токенов
 *           -> статический rANS (T = 4096) по каждому потоку.
 *
 * v2: контейнер с magic, CRC32, u64 и БЛОЧНЫМ rANS: каждый поток режется на
 *     независимые блоки по CHUNK байт с ОДНОЙ общей таблицей частот на поток.
 *     Блоки кодируются/декодируются параллельно: OpenMP на CPU, CUDA на GPU
 *     (см. ox_cuda.cu, сборка `make gpu`).
 *
 * Формат v2:
 *   "OX2\n" [u32 crc32 LE][u64 n LE]
 *   x5 потоков (flags, lits, lens, off_hi, off_lo):
 *     [u32 count][256 x u16 freq][u32 nchunks][u32 chunk]
 *     [u32 blen x nchunks][bodies]
 *   body блока = [u32 state BE][emission]   (rANS, состояние в начале, BE)
 *
 * Формат v1 (legacy, совместим с ox.py) — чтение всегда, запись с --v1:
 *   [u32 n] x5{ [u32 blob_len][blob] },  blob = [u32 count][256 x u16 freq][body]
 *
 * Сборка:  make            (авто: GPU если есть nvcc, иначе CPU)
 *          make cpu / make gpu
 * Запуск:  ./ox c in out.ox [--cpu|--gpu] [--v1]
 *          ./ox d in.ox out [--cpu|--gpu]
 *          ./ox bench file [--cpu|--gpu]
 *          ./ox info
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define SCALE   12
#define T       (1u << SCALE)
#define RANS_L  (1u << 23)
#define CHUNK   32768u              /* размер блока rANS в формате v2 */
#define OX2_MAGIC "OX2\n"

/* --------------------------------------------------------- CUDA-стадия --- */
#ifdef OX_CUDA
extern int ox_cuda_available(void);
extern const char *ox_cuda_device_name(void);
extern int ox_cuda_enc_stream(const uint8_t *d, uint32_t n, uint32_t chunk,
                              uint16_t *freq, uint32_t *blens,
                              uint8_t *bodies, uint32_t *bodies_n);
extern int ox_cuda_dec_stream(const uint8_t *bodies, const uint32_t *blens,
                              uint32_t n, uint32_t chunk,
                              const uint16_t *freq, uint8_t *out);
#else
static int ox_cuda_available(void) { return 0; }
static const char *ox_cuda_device_name(void) { return ""; }
static int ox_cuda_enc_stream(const uint8_t *d, uint32_t n, uint32_t chunk,
                              uint16_t *freq, uint32_t *blens,
                              uint8_t *bodies, uint32_t *bodies_n) {
    (void)d; (void)n; (void)chunk; (void)freq; (void)blens;
    (void)bodies; (void)bodies_n; return -1;
}
static int ox_cuda_dec_stream(const uint8_t *bodies, const uint32_t *blens,
                              uint32_t n, uint32_t chunk,
                              const uint16_t *freq, uint8_t *out) {
    (void)bodies; (void)blens; (void)n; (void)chunk; (void)freq; (void)out;
    return -1;
}
#endif

/* ---------------------------------------------------------------- CRC32 -- */
static uint32_t crc_tab[256];
static void crc_init(void) {
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t c = i;
        for (int k = 0; k < 8; k++) c = (c & 1) ? (c >> 1) ^ 0xEDB88320u : c >> 1;
        crc_tab[i] = c;
    }
}
static uint32_t crc32_ox(const uint8_t *d, size_t n) {
    uint32_t c = 0xFFFFFFFFu;
    for (size_t i = 0; i < n; i++) c = crc_tab[(c ^ d[i]) & 0xFF] ^ (c >> 8);
    return ~c;
}

/* ---------------------------------------------------------------- rANS -- */
/* Нормализация частот к сумме T; у всех встреченных символов freq >= 1. */
static void build_table(const uint8_t *d, size_t n, uint16_t freq[256]) {
    uint32_t cnt[256] = {0};
    memset(freq, 0, 256 * sizeof(uint16_t));
    if (!n) return;
    for (size_t i = 0; i < n; i++) cnt[d[i]]++;
    uint16_t order[256]; int nu = 0;
    uint32_t sum = 0;
    for (int i = 0; i < 256; i++) if (cnt[i]) {
        uint32_t f = (uint32_t)((uint64_t)cnt[i] * T / (uint32_t)n);
        freq[i] = (uint16_t)(f ? f : 1);
        sum += freq[i];
    }
    /* порядок коррекции: по убыванию cnt (простая сортировка вставками) */
    for (int i = 0; i < 256; i++) if (cnt[i]) {
        int j = nu++;
        while (j > 0 && cnt[order[j - 1]] < cnt[i]) { order[j] = order[j - 1]; j--; }
        order[j] = (uint16_t)i;
    }
    long diff = (long)T - (long)sum;
    size_t k = 0;
    while (diff) {
        int i = order[k % nu];
        if (diff > 0)      { freq[i]++; diff--; }
        else if (freq[i] > 1) { freq[i]--; diff++; }
        k++;
    }
}

/* Кодирует блок по готовой таблице. out: [u32 state BE][emission reversed].
 * Ёмкость out: 4 + 2*n байт (на символ выходит не более 2 байт ренормализации). */
static size_t rans_enc_chunk(const uint8_t *d, uint32_t n,
                             const uint16_t *freq, const uint32_t *cum,
                             uint8_t *out) {
    uint32_t x = RANS_L;
    uint8_t *tmp = out + 4;
    size_t tp = 0;
    for (uint32_t i = n; i-- > 0; ) {          /* rANS — стек: с конца */
        uint8_t s = d[i];
        uint32_t f = freq[s];
        uint32_t x_max = ((RANS_L >> SCALE) << 8) * f;
        while (x >= x_max) { tmp[tp++] = (uint8_t)(x & 0xFF); x >>= 8; }
        x = (uint32_t)(((uint64_t)(x / f) << SCALE) + (x % f) + cum[s]);
    }
    out[0] = (uint8_t)(x >> 24); out[1] = (uint8_t)(x >> 16);
    out[2] = (uint8_t)(x >> 8);  out[3] = (uint8_t)x;
    for (size_t i = 0; i < tp / 2; i++) {
        uint8_t t = tmp[i]; tmp[i] = tmp[tp - 1 - i]; tmp[tp - 1 - i] = t;
    }
    return 4 + tp;
}

static void rans_dec_chunk(const uint8_t *body, uint32_t body_n,
                           uint8_t *out, uint32_t n,
                           const uint16_t *freq, const uint32_t *cum,
                           const uint8_t *lut) {
    uint32_t x = ((uint32_t)body[0] << 24) | ((uint32_t)body[1] << 16) |
                 ((uint32_t)body[2] << 8) | body[3];
    uint32_t pos = 4;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t slot = x & (T - 1);
        uint8_t s = lut[slot];
        out[i] = s;
        x = freq[s] * (x >> SCALE) + slot - cum[s];
        while (x < RANS_L)
            x = (x << 8) | (pos < body_n ? body[pos] : 0), pos++;
    }
}

/* ------------------------------------------------- rANS поток целиком ---- */
/* Кодирует поток d[0..n) блоками по chunk байт с общей таблицей.
 * Заполняет freq, blens[] и компактные bodies. bodies cap: 2*n + 4*nchunks. */
static void enc_stream_cpu(const uint8_t *d, uint32_t n, uint32_t chunk,
                           uint16_t *freq, uint32_t *blens,
                           uint8_t *bodies, uint32_t *bodies_n) {
    memset(freq, 0, 256 * sizeof(uint16_t));
    *bodies_n = 0;
    if (!n) return;
    uint32_t nch = (n + chunk - 1) / chunk;
    uint32_t slotcap = 4 + 2 * chunk;
    uint8_t *slots = malloc((size_t)nch * slotcap);
    uint32_t cum[257] = {0};
    build_table(d, n, freq);
    for (int i = 0; i < 256; i++) cum[i + 1] = cum[i] + freq[i];
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (uint32_t c = 0; c < nch; c++) {
        uint32_t lo = c * chunk, hi = lo + chunk; if (hi > n) hi = n;
        blens[c] = (uint32_t)rans_enc_chunk(d + lo, hi - lo, freq, cum,
                                            slots + (size_t)c * slotcap);
    }
    uint32_t off = 0;
    for (uint32_t c = 0; c < nch; c++) {
        memcpy(bodies + off, slots + (size_t)c * slotcap, blens[c]);
        off += blens[c];
    }
    free(slots);
    *bodies_n = off;
}

static void dec_stream_cpu(const uint8_t *bodies, const uint32_t *blens,
                           uint32_t n, uint32_t chunk,
                           const uint16_t *freq, uint8_t *out) {
    if (!n) return;
    uint32_t nch = (n + chunk - 1) / chunk;
    uint32_t cum[257] = {0};
    for (int i = 0; i < 256; i++) cum[i + 1] = cum[i] + freq[i];
    uint8_t *lut = malloc(T);
    for (int s = 0; s < 256; s++)
        for (uint32_t k = cum[s]; k < cum[s + 1]; k++) lut[k] = (uint8_t)s;
    uint32_t *offs = malloc(nch * sizeof(uint32_t));
    uint32_t off = 0;
    for (uint32_t c = 0; c < nch; c++) { offs[c] = off; off += blens[c]; }
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (uint32_t c = 0; c < nch; c++) {
        uint32_t lo = c * chunk, hi = lo + chunk; if (hi > n) hi = n;
        rans_dec_chunk(bodies + offs[c], blens[c], out + lo, hi - lo,
                       freq, cum, lut);
    }
    free(offs); free(lut);
}

/* ----------------------------------------------- legacy rANS (v1, ox.py) - */
static void corrupt(void);

static uint8_t *rans_encode_v1(const uint8_t *d, size_t n, size_t *out_n) {
    uint16_t freq[256];
    build_table(d, n, freq);
    uint32_t cum[257] = {0};
    for (int i = 0; i < 256; i++) cum[i + 1] = cum[i] + freq[i];
    size_t bn;
    uint8_t *body = malloc(4 + 2 * n + 16);
    bn = rans_enc_chunk(d, (uint32_t)n, freq, cum, body);   /* state BE + body */
    uint8_t *out = malloc(4 + 512 + bn);
    memcpy(out, &(uint32_t){(uint32_t)n}, 4);
    for (int i = 0; i < 256; i++) {
        out[4 + 2 * i]     = (uint8_t)(freq[i] & 0xFF);
        out[4 + 2 * i + 1] = (uint8_t)(freq[i] >> 8);
    }
    memcpy(out + 4 + 512, body, bn);
    free(body);
    *out_n = 4 + 512 + bn;
    return out;
}

static uint8_t *rans_decode_v1(const uint8_t *blob, size_t blob_n, size_t *out_n) {
    uint32_t n; memcpy(&n, blob, 4);
    if ((uint64_t)n > 8 * (blob_n - 4 - 512) + 4096) corrupt();
    uint16_t freq[256];
    uint32_t cum[257] = {0};
    for (int i = 0; i < 256; i++) {
        freq[i] = (uint16_t)(blob[4 + 2 * i] | (blob[4 + 2 * i + 1] << 8));
        cum[i + 1] = cum[i] + freq[i];
    }
    if (n && cum[256] != T) corrupt();
    uint8_t lut[T];
    for (int s = 0; s < 256; s++)
        for (uint32_t k = cum[s]; k < cum[s + 1]; k++) lut[k] = (uint8_t)s;
    uint8_t *res = malloc(n ? n : 1);
    rans_dec_chunk(blob + 4 + 512, (uint32_t)(blob_n - 4 - 512), res, n,
                   freq, cum, lut);
    *out_n = n;
    return res;
}

/* ---------------------------------------------------------- LZ-парсер --- */
#define MIN_M  4
#define MAX_M  258
#define WINDOW 65535
#define HBITS  17
#define HSIZE  (1u << HBITS)

static inline uint32_t load32(const uint8_t *p) {
    uint32_t v; memcpy(&v, p, 4); return v;
}
static inline uint32_t hash4(const uint8_t *p) {
    return (load32(p) * 2654435761u) >> (32 - HBITS);
}

typedef struct {
    uint8_t *flags, *lits, *lens, *oh, *ol;
    size_t n_flags, n_lits, n_ops;
} Streams;

static void lz_parse(const uint8_t *d, size_t n, Streams *st) {
    int32_t *head = malloc(HSIZE * sizeof(int32_t));
    int32_t *prev = malloc((n ? n : 1) * sizeof(int32_t));
    for (uint32_t i = 0; i < HSIZE; i++) head[i] = -1;
    st->flags = malloc(n / 8 + 16);
    st->lits  = malloc(n ? n : 1);
    st->lens  = malloc(n ? n : 1);
    st->oh    = malloc(n ? n : 1);
    st->ol    = malloc(n ? n : 1);
    st->n_flags = st->n_lits = st->n_ops = 0;
    size_t n_m = 0;
    uint8_t fb = 0; int fc = 0;

#define EMIT_OP() do {                                   \
        if (++fc == 8) { st->flags[st->n_flags++] = fb; fb = 0; fc = 0; } \
        st->n_ops++;                                     \
    } while (0)

#define INSERT(p) do {                                   \
        size_t _p = (p);                                 \
        if (_p + MIN_M <= n) {                           \
            uint32_t h = hash4(d + _p);                  \
            prev[_p] = head[h]; head[h] = (int32_t)_p;   \
        }                                                \
    } while (0)

    size_t i = 0;
    while (i < n) {
        /* best_match(i) */
        size_t bl = 0, bo = 0;
        if (i + MIN_M <= n) {
            uint32_t h = hash4(d + i);
            int32_t cand = head[h];
            size_t limit = n - i; if (limit > MAX_M) limit = MAX_M;
            int chain = 16;
            while (cand >= 0 && i - (size_t)cand <= WINDOW && chain--) {
                if (bl >= limit) break;
                /* защита от коллизий хэша + быстрый отсев */
                if (load32(d + cand) == load32(d + i) && d[cand + bl] == d[i + bl]) {
                    size_t l = MIN_M;
                    while (l < limit && d[cand + l] == d[i + l]) l++;
                    if (l > bl) { bl = l; bo = i - (size_t)cand; }
                }
                cand = prev[cand];
            }
        }
        if (bl >= MIN_M) {
            /* ленивое сопоставление: заглянуть на 1 позицию вперёд */
            if (i + 1 < n) {
                INSERT(i);
                size_t bl2 = 0;
                if (i + 1 + MIN_M <= n) {
                    uint32_t h = hash4(d + i + 1);
                    int32_t cand = head[h];
                    size_t limit = n - i - 1; if (limit > MAX_M) limit = MAX_M;
                    int chain = 16;
                    while (cand >= 0 && i + 1 - (size_t)cand <= WINDOW && chain--) {
                        if (bl2 >= limit) break;
                        if (load32(d + cand) == load32(d + i + 1) && d[cand + bl2] == d[i + 1 + bl2]) {
                            size_t l = MIN_M;
                            while (l < limit && d[cand + l] == d[i + 1 + l]) l++;
                            if (l > bl2) bl2 = l;
                        }
                        cand = prev[cand];
                    }
                }
                if (bl2 > bl) {                       /* литерал выгоднее */
                    fb >>= 1;
                    st->lits[st->n_lits++] = d[i];
                    EMIT_OP();
                    i++;
                    continue;
                }
            }
            fb = (uint8_t)((fb >> 1) | 0x80);
            st->lens[n_m] = (uint8_t)(bl - MIN_M);
            st->oh[n_m]   = (uint8_t)(bo >> 8);
            st->ol[n_m]   = (uint8_t)(bo & 0xFF);
            n_m++;
            EMIT_OP();
            size_t end = i + bl; if (end > n) end = n;
            for (size_t p = i; p < end; p++) INSERT(p);
            i += bl;
        } else {
            fb >>= 1;
            st->lits[st->n_lits++] = d[i];
            EMIT_OP();
            INSERT(i);
            i++;
        }
    }
    if (fc) st->flags[st->n_flags++] = (uint8_t)(fb >> (8 - fc));
    free(head); free(prev);
}

/* Сборка исходных данных из 5 потоков токенов (с проверкой границ). */
static uint8_t *lz_rebuild(uint32_t n, const uint8_t *flags, size_t n_flags,
                           const uint8_t *lits, size_t n_lits,
                           const uint8_t *lens, const uint8_t *oh,
                           const uint8_t *ol, size_t n_m) {
    uint8_t *out = malloc(n ? n : 1);
    if (!out) corrupt();
    size_t on = 0, li = 0, mi = 0, fi = 0;
    uint8_t fb = 0; int fc = 0;
    while (on < n) {
        if (!fc) {
            if (fi >= n_flags) corrupt();
            fb = flags[fi++]; fc = 8;
        }
        if (fb & 1) {
            if (mi >= n_m) corrupt();
            size_t l = lens[mi] + MIN_M;
            size_t off = ((size_t)oh[mi] << 8) | ol[mi];
            mi++;
            if (!off || off > on || l > n - on) corrupt();
            size_t st = on - off;
            for (size_t k = 0; k < l; k++) out[on++] = out[st + k];
        } else {
            if (li >= n_lits) corrupt();
            out[on++] = lits[li++];
        }
        fb >>= 1; fc--;
    }
    return out;
}

/* ---------------------------------------------------------- контейнер --- */
static uint8_t *ox_compress_v2(const uint8_t *d, size_t n, size_t *out_n, int use_gpu) {
    Streams st;
    lz_parse(d, n, &st);
    size_t nm = st.n_ops - st.n_lits;
    const uint8_t *streams[5] = { st.flags, st.lits, st.lens, st.oh, st.ol };
    size_t slens[5] = { st.n_flags, st.n_lits, nm, nm, nm };

    size_t sum = 0, nch_total = 0;
    for (int s = 0; s < 5; s++) {
        sum += slens[s];
        nch_total += slens[s] ? (slens[s] + CHUNK - 1) / CHUNK : 0;
    }
    size_t cap = 4 + 4 + 8 + 5 * (4 + 512 + 4 + 4) + 8 * nch_total + 2 * sum + 4096;
    uint8_t *out = malloc(cap);
    uint8_t *bodies = malloc(2 * (sum ? sum : 1) + 4 * nch_total + 4096);
    uint32_t *blens = malloc((nch_total ? nch_total : 1) * sizeof(uint32_t));
    uint16_t freq[256];

    memcpy(out, OX2_MAGIC, 4);
    memcpy(out + 4, &(uint32_t){crc32_ox(d, n)}, 4);
    memcpy(out + 8, &(uint64_t){(uint64_t)n}, 8);
    size_t op = 16;
    int gpu_ok = use_gpu && ox_cuda_available();
    for (int s = 0; s < 5; s++) {
        uint32_t sn = (uint32_t)slens[s];
        uint32_t nch = sn ? (sn + CHUNK - 1) / CHUNK : 0;
        uint32_t bn = 0;
        if (gpu_ok && ox_cuda_enc_stream(streams[s], sn, CHUNK, freq, blens,
                                         bodies, &bn) != 0)
            gpu_ok = 0;                       /* сбой CUDA -> CPU fallback */
        if (!gpu_ok)
            enc_stream_cpu(streams[s], sn, CHUNK, freq, blens, bodies, &bn);
        memcpy(out + op, &(uint32_t){sn}, 4); op += 4;
        for (int i = 0; i < 256; i++) {
            out[op + 2 * i]     = (uint8_t)(freq[i] & 0xFF);
            out[op + 2 * i + 1] = (uint8_t)(freq[i] >> 8);
        }
        op += 512;
        memcpy(out + op, &(uint32_t){nch}, 4);    op += 4;
        memcpy(out + op, &(uint32_t){CHUNK}, 4);  op += 4;
        memcpy(out + op, blens, 4 * nch);         op += 4 * nch;
        memcpy(out + op, bodies, bn);             op += bn;
    }
    free(st.flags); free(st.lits); free(st.lens); free(st.oh); free(st.ol);
    free(bodies); free(blens);
    *out_n = op;
    return out;
}

static uint8_t *ox_compress_v1(const uint8_t *d, size_t n, size_t *out_n) {
    Streams st;
    lz_parse(d, n, &st);
    size_t nm = st.n_ops - st.n_lits;
    const uint8_t *streams[5] = { st.flags, st.lits, st.lens, st.oh, st.ol };
    size_t slens[5] = { st.n_flags, st.n_lits, nm, nm, nm };

    size_t cap = 4 * n + 64 * 1024;
    uint8_t *out = malloc(cap);
    memcpy(out, &(uint32_t){(uint32_t)n}, 4);
    size_t op = 4;
    for (int s = 0; s < 5; s++) {
        size_t bn;
        uint8_t *blob = rans_encode_v1(streams[s], slens[s], &bn);
        memcpy(out + op, &(uint32_t){(uint32_t)bn}, 4); op += 4;
        memcpy(out + op, blob, bn); op += bn;
        free(blob);
    }
    free(st.flags); free(st.lits); free(st.lens); free(st.oh); free(st.ol);
    *out_n = op;
    return out;
}

static void corrupt(void) {
    fprintf(stderr, "ox: файл повреждён или имеет неподдерживаемый формат\n");
    exit(1);
}

static uint8_t *ox_decompress_v2(const uint8_t *blob, size_t blob_n,
                                 size_t *out_n, int use_gpu) {
    uint32_t crc; memcpy(&crc, blob + 4, 4);
    uint64_t n64; memcpy(&n64, blob + 8, 8);
    if (n64 > 0xFFFFFFFFull) corrupt();
    uint32_t n = (uint32_t)n64;
    size_t pos = 16;
    uint8_t *s[5]; uint32_t sn[5];
    int gpu_ok = use_gpu && ox_cuda_available();
    for (int k = 0; k < 5; k++) {
        uint16_t freq[256];
        if (pos + 4 + 512 + 8 > blob_n) corrupt();
        memcpy(&sn[k], blob + pos, 4); pos += 4;
        for (int i = 0; i < 256; i++)
            freq[i] = (uint16_t)(blob[pos + 2 * i] | (blob[pos + 2 * i + 1] << 8));
        pos += 512;
        uint32_t nch, chunk;
        memcpy(&nch, blob + pos, 4);   pos += 4;
        memcpy(&chunk, blob + pos, 4); pos += 4;
        uint32_t fsum = 0;
        for (int i = 0; i < 256; i++) fsum += freq[i];
        if (nch && fsum != T) corrupt();
        if (!nch && fsum) corrupt();
        if (!chunk || chunk > (1u << 24)) corrupt();
        if (nch != (sn[k] ? (sn[k] + chunk - 1) / chunk : 0)) corrupt();
        if (pos + 4 * (size_t)nch > blob_n) corrupt();
        s[k] = malloc(sn[k] ? sn[k] : 1);
        if (!s[k]) corrupt();
        if (nch) {
            uint32_t *blens = malloc(nch * sizeof(uint32_t));
            memcpy(blens, blob + pos, 4 * nch); pos += 4 * nch;
            const uint8_t *bodies = blob + pos;
            uint32_t total = 0;
            for (uint32_t c = 0; c < nch; c++) {
                if (blens[c] < 4 || blens[c] > 4 + 2 * chunk) corrupt();
                total += blens[c];
            }
            if (pos + total > blob_n) corrupt();
            int done = 0;
            if (gpu_ok && ox_cuda_dec_stream(bodies, blens, sn[k], chunk,
                                             freq, s[k]) == 0)
                done = 1;
            else if (gpu_ok) gpu_ok = 0;      /* сбой CUDA -> CPU fallback */
            if (!done)
                dec_stream_cpu(bodies, blens, sn[k], chunk, freq, s[k]);
            pos += total;
            free(blens);
        }
    }
    uint8_t *out = lz_rebuild(n, s[0], sn[0], s[1], sn[1], s[2], s[3], s[4], sn[2]);
    for (int k = 0; k < 5; k++) free(s[k]);
    if (crc32_ox(out, n) != crc) {
        fprintf(stderr, "ox: ОШИБКА CRC32 — файл повреждён\n");
        free(out);
        exit(1);
    }
    *out_n = n;
    return out;
}

static uint8_t *ox_decompress_v1(const uint8_t *blob, size_t blob_n, size_t *out_n) {
    if (blob_n < 4) corrupt();
    uint32_t n; memcpy(&n, blob, 4);
    size_t pos = 4;
    uint8_t *s[5]; size_t sn[5];
    for (int k = 0; k < 5; k++) {
        if (pos + 4 > blob_n) corrupt();
        uint32_t bn; memcpy(&bn, blob + pos, 4); pos += 4;
        if (bn < 4 + 512 || pos + bn > blob_n) corrupt();
        s[k] = rans_decode_v1(blob + pos, bn, &sn[k]);
        pos += bn;
    }
    uint8_t *out = lz_rebuild(n, s[0], sn[0], s[1], sn[1], s[2], s[3], s[4], sn[2]);
    for (int k = 0; k < 5; k++) free(s[k]);
    *out_n = n;
    return out;
}

static uint8_t *ox_decompress(const uint8_t *blob, size_t blob_n,
                              size_t *out_n, int use_gpu) {
    if (blob_n >= 4 && !memcmp(blob, OX2_MAGIC, 4))
        return ox_decompress_v2(blob, blob_n, out_n, use_gpu);
    return ox_decompress_v1(blob, blob_n, out_n);
}

/* ------------------------------------------------- параметрический rANS --
 * Режим для OX-Neuro (hyperprior): у каждого символа своя таблица (бакет).
 * Таблицы считает Python и передаёт как есть — расхождения невозможны.
 *
 *   ox prans c tables.bin data.bin out.bin
 *     tables.bin: [u32 K][K x 513 x u32 cum LE]
 *     data.bin:   [u32 n][u16 sym x n LE][u8 bucket x n]
 *     out.bin:    [u32 state BE][body]
 *   ox prans d tables.bin blob.bin bids.bin out.bin
 *     blob.bin:   [u32 state BE][body]
 *     bids.bin:   [u32 n][u8 bucket x n]
 *     out.bin:    [u16 sym x n LE][u32 consumed LE] — consumed != blob.bin size
 *                 означает, что таблицы не совпали с энкодером (десинхрон)
 */
static uint8_t *read_file(const char *path, size_t *n);
static void write_file(const char *path, const uint8_t *d, size_t n);

static int prans(int argc, char **argv) {
    if (argc < 2) return -1;
    size_t tn, dn, bn;
    uint8_t *td = read_file(argv[2], &tn);
    uint32_t K; memcpy(&K, td, 4);
    uint32_t *cums = (uint32_t *)(td + 4);
    if (tn < 4 + (size_t)K * 513 * 4) { fprintf(stderr, "prans: tables?\n"); return 1; }

    if (!strcmp(argv[1], "c") && argc == 5) {
        uint8_t *dd = read_file(argv[3], &dn);
        uint32_t n; memcpy(&n, dd, 4);
        uint16_t *sym = (uint16_t *)(dd + 4);
        uint8_t *bid = dd + 4 + (size_t)n * 2;
        if (dn < 4 + (size_t)n * 3) { fprintf(stderr, "prans: data?\n"); return 1; }
        uint8_t *tmp = malloc(2 * (size_t)n + 16);
        size_t tp = 0;
        uint32_t x = RANS_L;
        for (uint32_t i = n; i-- > 0; ) {
            uint32_t *cum = cums + (size_t)bid[i] * 513;
            uint32_t s = sym[i];
            uint32_t f = cum[s + 1] - cum[s], c = cum[s];
            uint32_t x_max = ((RANS_L >> SCALE) << 8) * f;
            while (x >= x_max) { tmp[tp++] = (uint8_t)(x & 0xFF); x >>= 8; }
            x = ((x / f) << SCALE) + (x % f) + c;
        }
        uint8_t *out = malloc(4 + tp);
        out[0] = (uint8_t)(x >> 24); out[1] = (uint8_t)(x >> 16);
        out[2] = (uint8_t)(x >> 8);  out[3] = (uint8_t)x;
        for (size_t i = 0; i < tp; i++) out[4 + i] = tmp[tp - 1 - i];
        write_file(argv[4], out, 4 + tp);
        printf("prans c: %u символов -> %zu байт\n", n, 4 + tp);
        free(tmp); free(out); free(dd); free(td);
        return 0;
    }
    if (!strcmp(argv[1], "d") && argc == 6) {
        uint8_t *bd = read_file(argv[3], &bn);
        uint8_t *bids_raw = read_file(argv[4], &dn);
        uint32_t n; memcpy(&n, bids_raw, 4);
        const uint8_t *bids = bids_raw + 4;
        if (dn < 4 + (size_t)n) { fprintf(stderr, "prans: bids?\n"); return 1; }
        /* LUT по слотам для каждого бакета (символы 0..511!) */
        uint16_t *lut = malloc((size_t)K * T * 2);
        for (uint32_t b = 0; b < K; b++) {
            uint32_t *cum = cums + (size_t)b * 513;
            for (uint32_t s = 0; s < 512; s++)
                for (uint32_t k = cum[s]; k < cum[s + 1]; k++)
                    lut[b * T + k] = (uint16_t)s;
        }
        uint8_t *out = malloc((size_t)n * 2 + 4);
        uint32_t x = ((uint32_t)bd[0] << 24) | ((uint32_t)bd[1] << 16) |
                     ((uint32_t)bd[2] << 8) | bd[3];
        size_t pos = 4;
        for (uint32_t i = 0; i < n; i++) {
            uint32_t b = bids[i];
            uint32_t *cum = cums + (size_t)b * 513;
            uint32_t slot = x & (T - 1);
            uint32_t s = lut[(size_t)b * T + slot];
            ((uint16_t *)out)[i] = (uint16_t)s;
            uint32_t f = cum[s + 1] - cum[s];
            x = f * (x >> SCALE) + slot - cum[s];
            while (x < RANS_L)
                x = (x << 8) | (pos < bn ? bd[pos] : 0), pos++;
        }
        /* сколько байт потока реально прочитано: != bn => бакеты не те */
        uint32_t consumed = (uint32_t)pos;
        memcpy(out + (size_t)n * 2, &consumed, 4);
        write_file(argv[5], out, (size_t)n * 2 + 4);
        printf("prans d: %u символов\n", n);
        free(out); free(lut); free(bd); free(bids_raw); free(td);
        return 0;
    }
    return -1;
}

/* ----------------------------------------------------------------- CLI -- */
static uint8_t *read_file(const char *path, size_t *n) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t *d = malloc(sz ? sz : 1);
    if (fread(d, 1, sz, f) != (size_t)sz) { perror("read"); exit(1); }
    fclose(f); *n = (size_t)sz; return d;
}
static void write_file(const char *path, const uint8_t *d, size_t n) {
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); exit(1); }
    fwrite(d, 1, n, f); fclose(f);
}
static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void bench_one(const uint8_t *d, size_t n, int use_gpu, const char *tag) {
    size_t on, dn;
    double t0 = now_s();
    uint8_t *o = ox_compress_v2(d, n, &on, use_gpu);
    double te = now_s() - t0;
    t0 = now_s();
    uint8_t *r = ox_decompress(o, on, &dn, use_gpu);
    double td = now_s() - t0;
    printf("  %-4s: %zu (%.1f%%)  enc %.3f с (%.1f МБ/с)  dec %.3f с (%.1f МБ/с)  roundtrip %s\n",
           tag, on, 100.0 * on / (n ? n : 1),
           te, n / (te > 0 ? te : 1e-9) / 1e6,
           td, n / (td > 0 ? td : 1e-9) / 1e6,
           (dn == n && !memcmp(d, r, n)) ? "OK" : "FAIL");
    free(o); free(r);
}

int main(int argc, char **argv) {
    crc_init();
    int use_gpu = 1, v1 = 0;
    const char *pos[8] = {0}; int np = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--cpu"))      use_gpu = 0;
        else if (!strcmp(argv[i], "--gpu")) use_gpu = 1;
        else if (!strcmp(argv[i], "--v1"))  v1 = 1;
        else if (np < 8) pos[np++] = argv[i];
    }
    if (np >= 2 && !strcmp(pos[0], "prans")) {
        int r = prans(np, (char **)pos);
        if (r < 0) fprintf(stderr, "ox prans c tables data out | ox prans d tables blob bids out\n");
        return r < 0 ? 1 : r;
    }
    if (np == 1 && !strcmp(pos[0], "info")) {
#ifdef OX_CUDA
        int cuda = ox_cuda_available();
        printf("{\"version\":\"2.0\",\"format\":\"OX2\",\"cuda_compiled\":true,"
               "\"cuda\":%s,\"gpu\":\"%s\"}\n",
               cuda ? "true" : "false", cuda ? ox_cuda_device_name() : "");
#else
        printf("{\"version\":\"2.0\",\"format\":\"OX2\",\"cuda_compiled\":false,"
               "\"cuda\":false,\"gpu\":\"\"}\n");
#endif
        return 0;
    }
    if (np == 3 && !strcmp(pos[0], "c")) {
        size_t n, on; uint8_t *d = read_file(pos[1], &n);
        double t0 = now_s();
        uint8_t *o = v1 ? ox_compress_v1(d, n, &on)
                        : ox_compress_v2(d, n, &on, use_gpu);
        double te = now_s() - t0;
        write_file(pos[2], o, on);
        printf("%zu -> %zu (%.1f%%) за %.3f с%s\n", n, on,
               100.0 * on / (n ? n : 1), te, v1 ? " (v1)" :
               (use_gpu && ox_cuda_available()) ? " (GPU)" : " (CPU)");
    } else if (np == 3 && !strcmp(pos[0], "d")) {
        size_t n, on; uint8_t *d = read_file(pos[1], &n);
        double t0 = now_s();
        uint8_t *o = ox_decompress(d, n, &on, use_gpu);
        printf("распаковано %zu байт за %.3f с\n", on, now_s() - t0);
        write_file(pos[2], o, on);
    } else if (np == 2 && !strcmp(pos[0], "bench")) {
        size_t n; uint8_t *d = read_file(pos[1], &n);
        printf("%s: %zu байт\n", pos[1], n);
        bench_one(d, n, 0, "CPU");
        if (ox_cuda_available()) bench_one(d, n, 1, "GPU");
        else printf("  GPU : недоступен (сборка без CUDA или нет устройства)\n");
    } else {
        fprintf(stderr,
            "ox c in out.ox [--cpu|--gpu] [--v1]\n"
            "ox d in.ox out [--cpu|--gpu]\n"
            "ox bench file [--cpu|--gpu]\n"
            "ox info\n");
        return 1;
    }
    return 0;
}
