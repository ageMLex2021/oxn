# Перенос обучения на arm64 — задача для Kimi CLI

> Этот файл — инструкция для новой сессии Kimi CLI на arm64-машине.
> Прочти его целиком, затем `README.md` и раздел «OXN5/OXN6/OXN7» в
> `ARCHITECTURE.md`, и только потом запускай что-либо.

## Проект

OX-Neuro — lossy нейрокодек изображений: transformer-автоэнкодер
(патчи 16×16, windowed attention) → скейлер-квантизатор `round(z*s)/s`
(одна модель на все битрейты) → hyperprior → параметрический rANS.
Весь кодек — один файл `ox_neuro.py`. Энтропийный C-ускоритель — `ox.c`
(бинарник `ox` в репо собран под x86-64; на arm64 пересобрать `make` или
он молча отвалится в чистый Python — работает, но медленнее).

## Состояние репозитория

Разработка велась в локальной ветке **`oxn5-context`**; готовый OXN7 и
обратносовместимый декодер перенесены в **`main`**. Ничего не пушить без
отдельной инструкции.

- `ox_neuro.pt` — основной кодек (16.9M параметров, latent=24, hyper).
- `ox_ctx.pt` — контекстная сеть OXN5/OXN6 (checkerboard, hidden=128).
- `ox_ctx7.pt` — контекстная сеть OXN7 (канальные группы 6,6,6,6 + checkerboard).
- Форматы: OXN1/OXN2 (legacy), OXN3 (hyper, плоский), OXN4 (адаптивная
  лесенка), OXN5 (OXN3+контекст), OXN6 (OXN4+контекст), OXN7 (ELIC-style
  канальные группы + контекст). Новая упаковка — только OXN7; декодер выбирает
  любой поддерживаемый формат по magic-байтам.

Достигнуто на holdout kodim21–24 (эти 4 файла **никогда не обучать**),
PSNR идентичен:

| Формат | Против | Экономия |
|---|---|---|
| OXN5 | OXN3 | −5.7…−18.6% (ср. −11.9%) |
| OXN6 | OXN4 | −1.5…−14.5% (ср. −9.5%) |
| OXN7 | OXN5 | −3…−4% (q15), −4…−5% (q50), −13…−16% (q92); суммарно против OXN3 на q92 до **−31%** |

## Setup на arm64

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt          # torch, numpy, pillow
```

- torch: если это DGX Spark / GB10 (aarch64 + CUDA) — ставь CUDA-сборку
  для aarch64 с https://pytorch.org (или NVIDIA-индекс). Если Mac —
  будет CPU (MPS в коде не задействован): работает, но полный ретрен
  кодека на CPU непрактичен; ctx-сети обучаются и на CPU.
- C-ускоритель: `make` (нужен gcc; `make gpu` — если есть nvcc).
- Данные: скопировать со старой машины папку `data/` из
  `/home/acer/Документы/python/ox/data/` **внутрь `oxn/`** (т.е.
  `oxn/data/train/` — 1422 img, 3.8 ГБ; `oxn/data/kodak/`;
  `oxn/data/test/` — holdout kodim21–24). В git она не попадёт:
  `data/` добавлена в `.gitignore` на ветке. Все команды ниже
  рассчитаны именно на это расположение.

### Проверка после переноса (обязательно)

1. `python3 ox_neuro.py bench <один из data/kodak>` — пайплайн живой.
2. Декодировать заранее сделанный на x86 файл OXN5/OXN7. Энтропийный
   канон — CPU fp64 с выключенным oneDNN; на arm64 oneDNN нет вовсе, и
   теоретически ulp-различия BLAS могут десинхронизировать **старые**
   файлы (`corrupt .oxh: entropy stream desync`). Новые файлы,
   сделанные на arm64, самосогласованы. Если старые не читаются — это
   ожидаемо, пережать заново.

## Железные правила (нарушение = битые файлы)

1. Всё, что задаёт энтропийные таблицы (hyperprior, контекстные сети),
   считается **на CPU в fp64 с `torch.backends.mkldnn.enabled = False`** —
   и на энкодере, и на декодере (`_hyper_pair`, `_ctx_params`).
2. Скейлер квантуется до использования в ctx-входах: OXN5 — `s` в float32,
   OXN6 — лесенка в float16, как в заголовке файла.
3. В контекстную сеть подаётся **декодируемая** версия латента (остаток
   зажат в ±256), не истинная — иначе энкодер и декодер разъезжаются на
   больших `s`.
4. Тесты всегда с `--verify` (энкодер сам декодирует энтропию обратно);
   `_dec_syms` проверяет `consumed == len(blob)`.
5. Результат измерять **только** на `data/test` (kodim21–24).

## Задача A (основная): полный ретрен кодека с SE-гейдами + OXN7-рейтом

Код готов и проверен (smoke 20 шагов проходит). В трансформ добавлены
SE-гейты на латенте (`--se`, attention-маски в духе NLAM — неявное
распределение бит по важности; zero-init, стартуют тождественными);
`--ctx7` подключает полную OXN7-энтропийную модель в лосс — энкодер
учится делать латенты, дружественные к контексту, а `ox_ctx7.pt`
дообучается совместно.

```bash
python3 ox_neuro.py train --steps 40000 --batch 8 --data data/train \
    --init ox_neuro.pt --dim 320 --depth 6 --latent 24 --refine --hyper \
    --window 8 --se --ctx7 --w_mse 300 --w_msssim 100 --w_hp 100 \
    --out ox_neuro_v11.pt
```

- Чекпоинты пишутся каждые 5000 шагов (`*.part`) — можно прерывать и
  продолжать теми же флагами + `--init ox_neuro_v11.pt.part`
  (и `--ctx_file ox_ctx7.pt.part`, если продолжаешь и ctx).
- Смотри по логу `bpp~` (должен падать при том же `psnr~`).

### Приёмка задачи A

```bash
for f in data/test/kodim2{1,2,3,4}.png; do python3 ox_neuro.py bench $f; done
```

- Сравнить OXN7-строки с новой моделью (`--ckpt` в bench не пробрасывается
  для v5/v6/v7-строк — см. код `cmd_bench`; при надобности поправь) против
  базовой `ox_neuro.pt`. Цель: тот же PSNR при меньшем bpp.
- Roundtrip новой модели с `--verify` на паре картинок, в т.ч. CPU-only
  (`CUDA_VISIBLE_DEVICES=""`).
- Если лучше — `ox_neuro_v11.pt` становится основным (UI грузит
  `ox_neuro.pt` по умолчанию: скопируй, старый переименуй, не удаляй).
- Результаты и решение записать в раздел «Лог» ниже.

## Задача B: масштабирование контекстной ветки

124K → 460K параметров уже дали прирост; потолок не найден. Эксперименты
(каждый — отдельным файлом `ox_ctx7_hN.pt`, в `trainctx` это
`--ctx_hidden` + сохранение рядом, не затирая рабочий):

```bash
python3 ox_neuro.py trainctx --steps 100000 --batch 8 --ctx_hidden 192 \
    --groups 6,6,6,6 --data data/train
```

- Варианты: hidden 192/256; неравномерные группы (ELIC: начало мельче,
  напр. `4,4,8,8`); второй пространственный conv 5×5 в `_build_ctx7`;
  больше кропов на изображение (сейчас 2, см. `cmd_trainctx`).
- После каждого: бенч OXN5/OXN7 на holdout, записать в «Лог». Победителя
  разложить по штатным `ox_ctx.pt` / `ox_ctx7.pt`.

## Полезные команды

```bash
python3 ox_neuro.py c in.png out.oxh --q 75 [--verify]  # всегда OXN7
python3 ox_neuro.py d in.oxh out.jpg                     # OXN1–OXN7 по magic
python3 ox_neuro.py trainctx --steps N --ctx_hidden 128 [--groups 6,6,6,6] \
    [--init ox_ctx7.pt.part] --data data/train
```

## Ссылки

- NLAIC (non-local attention + 3D-CNN контекст, исходная идея):
  https://arxiv.org/html/1910.06244v1
- ELIC (канальные группы + space-channel контекст):
  https://arxiv.org/abs/2203.10886
- Checkerboard context (двухпроходный параллельный декод):
  https://arxiv.org/abs/2103.15306
- Ballé et al. 2018 (hyperprior): https://arxiv.org/abs/1802.01436
- Minnen et al. 2018 (joint autoregressive + hyperprior):
  https://arxiv.org/abs/1809.02736
- Cheng et al. 2020 (attention modules + GMM):
  https://arxiv.org/abs/2004.09570
- CompressAI (эталонные реализации): https://arxiv.org/abs/2011.03029
- Учебный репо (только структура, заявленные цифры недостоверны):
  https://github.com/adimalkar/Neural-Image-Compressor

## Лог (дополнять по ходу работы)

- 2026-11-18 (x86, RTX 4060): OXN5 −11.9% vs OXN3, OXN6 −9.5% vs OXN4
  (среднее по kodim21–24, PSNR идентичен). Коммиты `a19a912`, `f73cbf6`.
- 2026-11-18 (x86): OXN7 работает, roundtrip с `--verify` чистый.
  **Важно: `ox_ctx7.pt` недообучен** (~25k из 40k шагов, обучение прервано
  для переноса на arm64). Первое действие на новой машине — дообучить:

  ```bash
  python3 ox_neuro.py trainctx --steps 40000 --batch 8 --ctx_hidden 128 \
      --groups 6,6,6,6 --init ox_ctx7.pt --data data/train
  ```

  Замер недообученного состояния (kodim21–24, PSNR идентичен):
  OXN7 vs OXN5: q15 −3.3%, q50 −4.3%, q92 −15.1% (средние);
  против OXN3 на q92 суммарно −27…−31%. После дообучения перемерить —
  цифры должны стать лучше.
- 2026-11-19 (arm64, NVIDIA GB10): C/CUDA-ускоритель пересобран, обязательный
  bench на `data/kodak/kodim01.png` прошёл. Старых `.oxh` для проверки
  совместимости с x86 в репозитории нет.
- 2026-11-19 (arm64): `trainctx` на 40000 шагов завершён (1273 с).
  OXN7 vs OXN5 на kodim21–24, PSNR идентичен: q15 −4.46%, q50 −5.24%,
  q92 −15.92% (средние экономии по bpp; до дообучения −3.3/−4.3/−15.1%).
  Для экспериментов `--ctx_file` теперь задаёт отдельный OXN7-чекпоинт в
  `trainctx`, `train`, `c`, `d` и `bench`.
