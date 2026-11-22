# PICO MoE: состояние работы и порядок обучения

Обновлено: 2026-11-22. Документ нужен для продолжения после compaction.
Основной исследовательский журнал: `RESEARCH_IDEAS.md`. Результаты: `BENCHMARKS.md`.

## Задача и ограничения

- Пользователь просит единую PICO MoE с общим пулом весов и low/mid/high experts.
- Цель последнего сообщения: превзойти Apple PICO по RD, если идеи не закончатся.
- Tiny не изменять. Старые рабочие checkpoints не перезаписывать.
- UI/viewer нужны с автоматическим выбором модели по magic и resident cache.
- Кандидаты принимать только после реального roundtrip, matched-bpp RD, LPIPS/MS-SSIM,
  скорости и памяти. Не выдавать промежуточные модели за финальные.

## Уже готовые базовые модели

- `oxn_pico_v4_soup50.pt`: high-rate baseline, UI `pico2`, OXP6.
- `oxn_pico_v5_sparse_frozen_7k.pt`: sparse low-rate, UI `pico5`, OXP7.
- `ox_tiny_v4.pt`, `ox_tiny_v4_ctx.pt`, `ox_tiny_v4_boost.pt`: UI `tiny2`, OXNV. Не трогать.
- `picoauto`: v5 при target<=0.25 или q<=10, иначе v4.
- `picomoe` в UI сейчас указывает `PICO_MOE_CKPT=oxn_pico_v8_moe6.pt`, version 9.

Корпус: 120799 изображений: COCO train2017 118287, DIV2K 800, text_synth 2300,
`data/train` и `/home/agemlex/Pictures`. Holdout исключены: IMG_20261015_111154.jpg,
Kodak21-24, официальные Apple-сцены и 3 screenshots.

## Ключевая MoE-архитектура

Файл `oxn_pico_v8_moe6.pt`, `PicoMoeModel` в `oxn_pico.py`:

- общий analysis/predictor/scale_decoder/context_decoder/hyperprior;
- 6 synthesis experts с одним активным expert на изображение;
- OXP9: sparse nonzero bitmap/zlib + обычный rANS только для nonzero;
- явный expert ID в outer header (`blob[29]`), target-bpp bins:
  - 0 ultra-low: <0.075;
  - 1 low: 0.075–0.15;
  - 2 mid-low: 0.15–0.3;
  - 3 mid: 0.3–0.75;
  - 4 high: 0.75–1.5;
  - 5 ultra-high: >1.5.
- Ultra-low expert 0 использует spatial base layer 2x2: entropy symbols только на
  checkerboard anchors, остальные позиции восстанавливаются predictor’ом.
- Все 6 experts и spatial roundtrips проверены. Initial expert bpp:
  ultra-low ~0.075, low ~0.178, mid-low ~0.264, mid ~0.488, high ~0.873,
  ultra-high ~1.477 на kodim21.

Параметры: 6-expert model ~11M? Точное число не зафиксировано; 3-expert был 8.87M.
Не писать точное число без повторного подсчёта.

## Исходные веса experts

- 0: `oxn_pico_v5_sparse_frozen_7k.pt`.
- 1: `oxn_pico_v7_moe_low_perc_fixed.pt.part`.
- 2: `oxn_pico_v6_multirate_5k.pt`.
- 3: `oxn_pico_v7_moe_mid.pt`.
- 4: `oxn_pico_v4_soup50.pt`.
- 5: `oxn_pico_v7_moe_high.pt`.

Общий analysis взят из v5. Важно: v4 и v5 shared analysis/entropy совпадают, поэтому
MoE можно собирать без полного retrain. Первый 3-expert MoE совпал low/high побайтово
и попиксельно.

## Специализация уже выполнена

1. Middle (`oxn_pico_v7_moe_tuned.pt`) обучался на full/half crops.
2. Общее pretraining с teacher augment (`oxn_pico_v7_moe_pretrain_2k.pt`):
   high-rate teacher reconstructions подаются на вход, оригинал — ground truth.
   High-rate достиг 29.62 dB на Kodak 1bpp, но low bpp вырос до 0.39.
3. Low specialization (`oxn_pico_v7_moe_low_free.pt`) уменьшил bpp до 0.253.
4. High expert (`oxn_pico_v7_moe_high.pt`): q100 LPIPS ~0.0655 при том же bpp.
5. Ultra-low spatial expert (`oxn_pico_v8_moe6_low.pt`): ultra-low LPIPS улучшен
   примерно 0.545 -> 0.476 при неизменном 0.075445 bpp.

Последние baseline RD ultra-low до дальнейшего fine-tune:

- 0.075445 bpp: PSNR 21.319, MS-SSIM 0.6896, LPIPS 0.47635.
- 0.178141 bpp: PSNR 24.133, MS-SSIM 0.8241, LPIPS 0.40480.

## Текущая активная задача

`bash-amvz67ay`: low expert 1, 5000 шагов, output `oxn_pico_v8_moe6_low1.pt`,
init `oxn_pico_v8_moe6_low.pt`, `moe=6`, `train_expert=1`, q7–28, multi_resolution,
w_lpips=30, w_msssim=10, w_gan=0.05, TAL/color/seam/text. Статус проверить через
TaskList/TaskOutput, не считать завершённым.

Ранний 1k результат low1: 0.178141 bpp, PSNR 24.215, MS-SSIM 0.8255, LPIPS 0.41047
(по сравнению с base 24.133/0.8241/0.4048). Финал не проверен.

## Что сделать дальше по порядку

1. Проверить финал `oxn_pico_v8_moe6_low1.pt` и сравнить с `oxn_pico_v8_moe6.pt`.
2. Собрать финальный checkpoint от `oxn_pico_v8_moe6_low1.pt`, заменив expert 3 на
   `oxn_pico_v7_moe_mid.pt`, expert 4 на `oxn_pico_v4_soup50.pt`, expert 5 на
   `oxn_pico_v7_moe_high.pt`. Experts 0/1/2 оставить из low1.
3. Проверить OXP9 все expert IDs и RD 0.05/0.1/0.2/0.35/0.5/1.0/2.0, затем Apple scenes
   через `bench_pico_reimpl.py --stream-version 9`.
4. Проверить screenshots и 201-Мп multi-tile roundtrip, UI/viewer, resident preload.
5. Только если RD лучше прежних профилей, назначить финал в `PICO_MOE_CKPT`.
   Иначе оставить `oxn_pico_v8_moe6.pt` и задокументировать отказ.
6. Обновить `RESEARCH_IDEAS.md`, `BENCHMARKS.md`, `README.md`, `ARCHITECTURE.md`.
7. Запустить make, py_compile, node --check inline UI, git diff --check. Удалить только
   свои rejected/duplicate part-файлы, не рабочие checkpoints.

## Исправленные ошибки, не повторять

- Sparse training rate ошибочно умножался на spatial mask после суммирования bits и был
  завышен примерно в 64 раза. Исправлено: `_sparse_rate_bits` принимает spatial mask.
- STE mask должен быть `hard + (soft - soft.detach())`; без скобок float-хвосты 0.99999994
  меняли entropy и мешали точному совпадению профилей.
- `mask_lr=0` раньше оставлял mask в optimizer. Теперь mask включается только при
  явном `mask_lr>0`.
- `torch.compile` predictor менял bitstream и не ускорял; thread-per-stream async rANS
  был ~3x медленнее. Не возвращаться.
- Dim64 Tiny был отклонён. Tiny вообще не трогать.
- Простое смешивание low/high weights ухудшало RD; MoE с synthesis experts правильнее.
- `picoauto` не заменяет MoE, это только практический fallback.

## Проверки и UI

- `picomoe` сейчас использует OXP9. Viewer выбирает MoE по OXP8/OXP9, v5 по OXP7,
  v4 по OXP6, Tiny по OXNV.
- UI/viewer OXP8 roundtrip уже проверен на 3-expert MoE. OXP9 нужно проверить заново
  после финала.
- UI/viewer процессы по просьбе пользователя закрывались. Не оставлять зависшие серверы.
- При проверке памяти использовать free -h; пользователь просил не держать лишние модели.

## Важное ожидание

Победа над Apple ещё не достигнута. Не объявлять её без matched benchmarks. Текущая
главная задача — завершить low1, собрать final 6-expert и доказать улучшение. Если финал
не превосходит профили, сохранить их и честно задокументировать исчерпание идей.

## Финальное состояние после завершения low1

`oxn_pico_v8_moe6_final.pt` собран и назначен в `PICO_MOE_CKPT`. Apple: 0.31932bpp,
26.924dB, 0.91442 MS-SSIM, 0.27475 LPIPS. Screenshots: 0.19990bpp, 27.409dB,
0.94073, 0.29265 — лучше v5 по всем метрикам. OXP9 UI/viewer и resident preload
прошли. `bash-amvz67ay` завершён. Оставалась только проверка 201-Мп и стандартные тесты.
Не писать, что Apple превзойдена: это улучшение прежних моделей, но не победа над Apple.

Финальная 201-Мп проверка: OXP9 0.08795bpp, 20.459dB, encode 27.30с, decode 21.08с,
peak RSS 3.17ГиБ, CUDA reserve 11.21ГиБ. make, py_compile, node --check и git diff --check
прошли. Фоновых тренировок не осталось. `oxn_pico_v8_moe6_final.pt` — принятая MoE.

## Оптимизация скорости после compaction

Веса и шесть rate experts не менялись. В `oxn_pico.py` принят GPU scale decoder на ARM/CUDA
для всех размеров, contiguous synthesis, strided spatial phases, параллельные независимые
entropy tasks, pinned transfers и GPU uint8 output. Подробности и отклонённые эксперименты:
конец `RESEARCH_IDEAS.md`; результаты: `BENCHMARKS.md` и `data/eval/pico_v8_moe6_speed_*`.

12-Мп q20: 1.366с encode, 0.946с decode. 201-Мп q10: 22.313/14.169с, прежние bpp/PSNR,
3.33ГиБ RSS. Apple streams побайтово совпали. Старые OXP6–OXP9 и все 6 experts читаются.
Apple speed всё ещё не достигнут; следующий кандидат — AOT/native integer/tensor-core
pipeline, а не изменение MoE или дополнительная параллелизация маленького rANS decode.
