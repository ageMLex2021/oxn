#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OX (objectx) — прототип «промышленного медленного компрессора».

Идея (из юношеской задумки автора):
  1. Файл сравнивается сам с собой: копия «скользит» по оригиналу,
     ищутся совпадающие непрерывные блоки.   -> стадия LZ77 (хэш-цепочки)
  2. Накапливается статистика неповторяющихся цепочек, строится словарь.
                                              -> токены (литерал/смещение/длина)
  3. «И так до битов».                        -> стадия rANS (энтропийное кодирование)

Экспериментально установлено (см. benchmark): повторный запуск OX на своём же
результате ничего не даёт — «до битов» достигается одним проходом rANS,
рекурсия поглощается энтропийной стадией.

Использование:
  python3 ox.py c файл файл.ox      # сжать
  python3 ox.py d файл.ox файл.out  # восстановить
  python3 ox.py bench файл          # сравнить с zlib/bz2/lzma
"""

import struct, sys, time

# ---------------------------------------------------------------- rANS -----
SCALE = 12                  # точность вероятностей: T = 4096
T = 1 << SCALE
RANS_L = 1 << 23            # нижняя граница состояния (байтовый rANS)

def _build_table(data):
    """Статические частоты, отмасштабированные к сумме T; freq >= 1 у всех встреченных."""
    cnt = [0] * 256
    for b in data: cnt[b] += 1
    n = len(data)
    if n == 0: return [0] * 256
    used = [i for i in range(256) if cnt[i]]
    freq = [0] * 256
    for i in used:
        freq[i] = max(1, (cnt[i] * T) // n)
    diff = T - sum(freq)
    order = sorted(used, key=lambda i: -cnt[i])
    k = 0
    while diff:
        i = order[k % len(order)]
        if diff > 0:   freq[i] += 1; diff -= 1
        elif freq[i] > 1: freq[i] -= 1; diff += 1
        k += 1
    return freq

def rans_encode(data):
    freq = _build_table(data)
    cum = [0] * 257
    for i in range(256): cum[i + 1] = cum[i] + freq[i]
    x = RANS_L
    out = bytearray()
    for s in reversed(data):                # rANS — стек: кодируем с конца
        f = freq[s]
        x_max = ((RANS_L >> SCALE) << 8) * f
        while x >= x_max:
            out.append(x & 0xFF); x >>= 8
        x = ((x // f) << SCALE) + (x % f) + cum[s]
    out += struct.pack('<I', x)
    out.reverse()
    hdr = struct.pack('<I', len(data)) + b''.join(struct.pack('<H', f) for f in freq)
    return hdr + bytes(out)

def rans_decode(blob):
    n = struct.unpack('<I', blob[:4])[0]
    freq = list(struct.unpack('<256H', blob[4:4 + 512]))
    cum = [0] * 257
    for i in range(256): cum[i + 1] = cum[i] + freq[i]
    body = blob[4 + 512:]
    x = struct.unpack('>I', body[:4])[0]    # после reverse() состояние лежит big-endian
    pos = 4
    res = bytearray()
    for _ in range(n):
        slot = x & (T - 1)
        lo, hi = 0, 255
        while lo < hi:                      # бинарный поиск символа по cum
            mid = (lo + hi + 1) // 2
            if cum[mid] <= slot: lo = mid
            else: hi = mid - 1
        s = lo
        res.append(s)
        x = freq[s] * (x >> SCALE) + slot - cum[s]
        while x < RANS_L:
            x = (x << 8) | (body[pos] if pos < len(body) else 0)
            pos += 1
    return bytes(res)

# ------------------------------------------------- стадия 1: самосравнение -
MIN_M, MAX_M, WINDOW = 4, 258, 65535

def lz_parse(d, max_chain=16, lazy=True):
    """Быстрый вариант «скольжения копии»: хэш-цепочки по 4 байтам +
    ленивое сопоставление. Возвращает ('lit', b) / ('match', offset, len)."""
    n = len(d)
    head = {}
    prev = [-1] * n
    ops = []
    def insert(p):
        if p + MIN_M <= n:
            h = int.from_bytes(d[p:p + 4], 'little')
            prev[p] = head.get(h, -1)
            head[h] = p
    def best_match(p):
        if p + MIN_M > n: return 0, 0
        h = int.from_bytes(d[p:p + 4], 'little')
        cand = head.get(h, -1)
        bl, bo = 0, 0
        limit = min(MAX_M, n - p)
        chain = max_chain
        while cand >= 0 and p - cand <= WINDOW and chain:
            if bl >= limit: break
            if d[cand + bl] == d[p + bl]:        # быстрый отсев
                l = MIN_M
                while l < limit and d[cand + l] == d[p + l]:
                    l += 1
                if l > bl: bl, bo = l, p - cand
            cand = prev[cand]
            chain -= 1
        return bl, bo
    i = 0
    while i < n:
        l, off = best_match(i)
        if l >= MIN_M:
            if lazy and i + 1 < n:
                insert(i)
                l2, _ = best_match(i + 1)
                if l2 > l:
                    ops.append(('lit', d[i])); i += 1
                    continue
            ops.append(('match', off, l))
            for p in range(i, min(i + l, n)):
                insert(p)
            i += l
        else:
            ops.append(('lit', d[i]))
            insert(i)
            i += 1
    return ops

def sliding_self_match(d, min_len=3):
    """Буквальная реализация исходной метафоры OX: копия файла скользит
    по оригиналу сдвиг за сдвигом, ищется самое длинное совпадение.
    O(n^2) — только для демонстрации на маленьких данных."""
    n = len(d)
    best = {}
    for s in range(1, n):
        i, best_l, best_p = 0, 0, 0
        while i + s < n:
            l = 0
            while i + s + l < n and d[i + l] == d[i + s + l]:
                l += 1
            if l > best_l: best_l, best_p = l, i
            i += max(l, 1)
        if best_l >= min_len:
            best[s] = (best_p, best_l)
    return best

# ------------------------------------------------------- контейнер OX ------
def _rans_dec_body(body, n, freq, cum, lut):
    """Декодирует один блок rANS: [u32 state BE][emission]."""
    x = struct.unpack('>I', body[:4])[0]
    pos = 4
    res = bytearray()
    for _ in range(n):
        slot = x & (T - 1)
        s = lut[slot]
        res.append(s)
        x = freq[s] * (x >> SCALE) + slot - cum[s]
        while x < RANS_L:
            x = (x << 8) | (body[pos] if pos < len(body) else 0)
            pos += 1
    return bytes(res)

def _ox_decompress_v2(blob):
    """Формат v2 (ox.c): magic, CRC32, u64, потоки нарезаны на блоки
    с общей таблицей частот на поток."""
    import zlib
    crc, n = struct.unpack('<IQ', blob[4:16])
    pos = 16
    streams = []
    for _ in range(5):
        count = struct.unpack('<I', blob[pos:pos + 4])[0]
        freq = list(struct.unpack('<256H', blob[pos + 4:pos + 516]))
        nch, chunk = struct.unpack('<II', blob[pos + 516:pos + 524])
        pos += 524
        blens = struct.unpack(f'<{nch}I', blob[pos:pos + 4 * nch])
        pos += 4 * nch
        cum = [0] * 257
        for i in range(256): cum[i + 1] = cum[i] + freq[i]
        lut = b''.join(bytes([s]) * freq[s] for s in range(256))
        out = bytearray()
        for c in range(nch):
            body = blob[pos:pos + blens[c]]
            pos += blens[c]
            out += _rans_dec_body(body, min(chunk, count - c * chunk),
                                  freq, cum, lut)
        streams.append(bytes(out))
    res = _lz_rebuild(n, streams)
    assert zlib.crc32(res) & 0xFFFFFFFF == crc, 'OX v2: ошибка CRC32'
    return res

def _lz_rebuild(n, streams):
    flags, lits, lens, oh, ol = streams
    out = bytearray(); li = mi = fi = 0; fb = 0; fc = 0
    while len(out) < n:
        if fc == 0:
            fb = flags[fi]; fi += 1; fc = 8
        if fb & 1:
            l = lens[mi] + MIN_M
            off = (oh[mi] << 8) | ol[mi]; mi += 1
            st = len(out) - off
            for k in range(l): out.append(out[st + k])
        else:
            out.append(lits[li]); li += 1
        fb >>= 1; fc -= 1
    return bytes(out)
def ox_compress(d):
    n = len(d)
    ops = lz_parse(d)
    flags = bytearray(); lits = bytearray(); lens = bytearray()
    oh = bytearray(); ol = bytearray()
    fb = fc = 0
    for op in ops:
        fb >>= 1
        if op[0] == 'match':
            fb |= 0x80
            _, off, l = op
            lens.append(l - MIN_M); oh.append(off >> 8); ol.append(off & 0xFF)
        else:
            lits.append(op[1])
        fc += 1
        if fc == 8: flags.append(fb); fb = fc = 0
    if fc: flags.append(fb >> (8 - fc))
    out = bytearray(struct.pack('<I', n))
    for stream in (bytes(flags), bytes(lits), bytes(lens), bytes(oh), bytes(ol)):
        blob = rans_encode(stream)
        out += struct.pack('<I', len(blob)) + blob
    return bytes(out)

def ox_decompress(blob):
    if blob[:4] == b'OX2\n':
        return _ox_decompress_v2(blob)
    n = struct.unpack('<I', blob[:4])[0]
    pos = 4; streams = []
    for _ in range(5):
        ln = struct.unpack('<I', blob[pos:pos + 4])[0]
        streams.append(rans_decode(blob[pos + 4:pos + 4 + ln]))
        pos += 4 + ln
    return _lz_rebuild(n, streams)

# ------------------------------------------------------------- CLI ---------
def _bench(path):
    import zlib, bz2, lzma
    d = open(path, 'rb').read()
    t0 = time.time(); ox = ox_compress(d); te = time.time() - t0
    t0 = time.time(); ok = ox_decompress(ox) == d; td = time.time() - t0
    print(f'{path}: {len(d)} байт')
    print(f'  OX    : {len(ox)} ({len(ox)/len(d)*100:.1f}%)  enc {te:.1f}с dec {td:.1f}с  roundtrip {ok}')
    print(f'  zlib-9: {len(zlib.compress(d, 9))}')
    print(f'  bz2   : {len(bz2.compress(d))}')
    print(f'  lzma  : {len(lzma.compress(d))}')

if __name__ == '__main__':
    if len(sys.argv) == 4 and sys.argv[1] == 'c':
        open(sys.argv[3], 'wb').write(ox_compress(open(sys.argv[2], 'rb').read()))
    elif len(sys.argv) == 4 and sys.argv[1] == 'd':
        open(sys.argv[3], 'wb').write(ox_decompress(open(sys.argv[2], 'rb').read()))
    elif len(sys.argv) == 3 and sys.argv[1] == 'bench':
        _bench(sys.argv[2])
    else:
        print(__doc__)
