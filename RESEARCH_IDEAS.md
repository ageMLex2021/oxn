# PICO и Tiny: план исследований и математические идеи

Снимок состояния: 2026-11-21. Документ сохраняет идеи и измерения, чтобы работу
можно было продолжить после компакции контекста.

## Цели

- PICO: сначала обойти OXN17 при одинаковом bpp, затем приближаться к Apple PICO
  по perceptual RD и скорости.
- Tiny: сохранить маленькую модель и скорость, уменьшить блочность и цветовой
  сдвиг, улучшить PSNR/LPIPS при сопоставимом bpp.
- Решения принимать только по фактическому rANS bitrate, RD-метрикам, скорости и
  памяти. Не принимать идею из-за красивой формулы или чужого результата.
- До появления лучших кандидатов практичный default — OXN17, PICO/tiny — research.

## Принятые финальные кандидаты

### PICO v4 soup50 / OXP6

Файлы: `oxn_pico_v4_soup50.pt`, UI model `pico2`. Старый `oxn_pico.pt` не заменён.

- Kodak 0.5000bpp: 27.535dB, MS-SSIM 0.93165, LPIPS 0.22066.
- Kodak 1.0008bpp: 29.500dB, MS-SSIM 0.96566, LPIPS 0.12815.
- Apple scenes: 0.32234bpp, 25.928dB, MS-SSIM 0.89650, LPIPS 0.32351. Apple всё ещё
  лидирует (0.290bpp, 29.548dB, 0.96974, 0.04118); гэп уменьшен, но не закрыт.
- Screenshots: +2.85dB и LPIPS 0.3735 против 0.5598 у старой модели; минимальный
  битрейт 0.2372 против 0.2041bpp, поэтому это качественный, не минимальный вариант.
- 201-Мп OXP6 q75 roundtrip: 33.835dB, encode 46.1с, decode 40.7с, RSS 3.12ГиБ,
  CUDA reserve 11.13ГиБ. UI/viewer/CLI и legacy OXP5 проверены.

### Tiny v4 + scale boost / OXNV

Файлы: `ox_tiny_v4.pt`, `ox_tiny_v4_ctx.pt`, `ox_tiny_v4_boost.pt`, UI model `tiny2`.
Старые Tiny checkpoints не заменены.

- Matched Kodak 0.12616bpp: 23.672dB, MS-SSIM 0.82790, LPIPS 0.45704; baseline
  0.12613bpp, 22.736dB, 0.85608, 0.59705. MS-SSIM ниже baseline на Kodak — сознательный
  perceptual tradeoff, а не полное доминирование.
- Boost на Kodak: encode 0.1257с против 0.2316с (1.84×), bpp немного меньше,
  LPIPS не хуже, PSNR −0.041dB.
- 201-Мп boost: 0.10360bpp, 24.617dB, MS-SSIM 0.99541, LPIPS 0.01653;
  encode 16.8с против 35.8с, decode 17.0с против 21.3с. CUDA reserve 6.43ГиБ.
- UI/viewer, resident preload, target bpp, legacy OXNT и переключение с PICO проверены.

Оставшиеся направления: Apple-level low-rate ниже 0.1bpp и perceptual LPIPS <0.1 на
общих сценах, рост MS-SSIM Tiny на маленьких изображениях без потери LPIPS, компиляция
и уменьшение CPU/GPU sync. Это следующий research этап, а не блокеры текущих кандидатов.

## Зафиксированная база

### Наш PICO против Apple

- На matched-bpp официальных сценах наш PSNR ниже на 3.1–11.4 dB, LPIPS примерно
  0.47–0.56 против 0.028–0.061 у Apple.
- 12 Мп: encode 2.60 с, decode 1.99 с. У Apple iPhone 17 Pro Max: 0.23/0.15 с.
- Пользовательский кадр 201 Мп:
  - OXN8: 5,957,803 bytes, 0.23674 bpp, 30.917 dB;
  - наш PICO q100: 23,422,236 bytes, 0.93072 bpp, 29.929 dB.
- PICO использует 3.93× больше бит OXN8 и всё равно хуже. Зелёный канал занижен
  примерно на 1.89 уровня, картинка темнее и менее насыщенная.

### Tiny

Пользовательский кадр 201 Мп:

- `(1).oxh`, старый checkpoint: 0.13785 bpp, 22.185 dB;
- `(2).oxh`, текущий checkpoint: 0.13162 bpp, 24.976 dB.
- Текущее дообучение уже дало +2.79 dB и −4.5% размера, но цвет и блочность
  остались. `(1)` требует старого context checkpoint, отсюда необходимость версий.
- 12 Мп q100: encode 2.34 с, decode 1.10 с, 0.14198 bpp, 19.727 dB.

### PICO-v2, пилот 20k на Kodak holdout при matched 0.5 bpp

| Модель | bpp | PSNR | MS-SSIM | LPIPS ↓ | ΔR | ΔG | ΔB |
|---|---:|---:|---:|---:|---:|---:|---:|
| PICO baseline | 0.49999 | 24.061 | 0.8934 | 0.4474 | +1.03 | −3.97 | +2.49 |
| PICO-v2 MSE/structure | 0.50017 | 27.021 | 0.9206 | 0.3455 | −0.04 | +0.14 | +0.02 |

Пилот даёт +2.96 dB и лучше все RD/цветовые метрики при одинаковом bpp. Он выбран
инициализацией полного retrain, но не заменяет рабочий checkpoint до большого корпуса
и финальных проверок. При q=75 его bpp выше, поэтому сравнение только по q невалидно.
Результаты: `data/eval/pico_v1_kodak_0p5bpp.json`,
`data/eval/pico_v2_mse_20k_kodak_0p5bpp.json`.

PICO-v3 после 12k полного корпуса: Kodak 0.5bpp 27.53 dB / 0.9271 MS-SSIM /
0.3144 LPIPS; после 25k: 27.78 dB / 0.9319 / 0.2863 при 0.4995bpp. Checkpoint
`oxn_pico_v3_coco_25k.pt` сохранён как текущий лучший MSE/structure candidate.
На официальных Apple-сценах гэп остаётся большим: LPIPS 0.37–0.45
против 0.016–0.061. Для Apple target 0.095bpp модель достигает лишь 0.195bpp даже
на q≈1, а target 0.217bpp — 0.232bpp. После основного retrain нужен log-uniform
low-rate/perceptual fine-tune; поддержка `--q_min/--q_max/--q_log` добавлена и прошла
training smoke. Результаты: `data/eval/pico/v3_coco_12k_apple.json`.

### Tiny-v2, промежуточный 5000-step candidate на Kodak holdout

| Модель | bpp | PSNR | MS-SSIM | LPIPS ↓ | Boundary MSE ↓ | ΔG |
|---|---:|---:|---:|---:|---:|---:|
| Tiny baseline | 0.12613 | 22.736 | 0.8561 | 0.5971 | 376.5 | −3.29 |
| Tiny-v2 5k | 0.11999 | 23.759 | 0.8364 | 0.6298 | 290.8 | −0.96 |

Плюсы: +1.02 dB, −4.9% bpp, −22.8% boundary MSE, меньше цветовой сдвиг.
Минусы: MS-SSIM/LPIPS немного хуже. Кандидат ещё не принят окончательно.
Результаты: `data/eval/tiny_v1_baseline.json`, `data/eval/tiny_v2_5k.json`.

### Tiny-v2, финальный пилот 20k

На Kodak q=75: 0.11938 bpp против 0.12613 (−5.4%), 23.777 против 22.736 dB
(+1.04 dB), boundary MSE 283.95 против 376.48 (−24.6%). Цветовой сдвиг почти устранён,
но MS-SSIM 0.8345 против 0.8561 и LPIPS 0.6215 против 0.5971 всё ещё хуже.
При matched 0.1261 bpp LPIPS улучшается лишь до 0.6127, поэтому общий Kodak RD/perceptual
критерий пока не пройден.

На пользовательском 201-Мп holdout результат противоположный по perceptual метрикам:
0.10442 bpp против 0.11585 (−9.9%), 25.085 против 24.844 dB, MS-SSIM 0.99399 против
0.99094, LPIPS 0.02676 против 0.06311. Encode/decode примерно те же (35.8/23.2 с),
пиковая CUDA reserve 6.43 ГиБ, process RSS 9.88 ГиБ. Кандидат перспективен для high-res,
но не принят как универсальный default. Следующий Tiny retrain должен усилить perceptual
часть loss и использовать большой корпус, не потеряв достигнутые PSNR/цвет/битрейт.

Результаты: `data/eval/tiny_v2_20k.json`, `data/eval/tiny_v2_20k_kodak_matched.json`,
`data/eval/tiny_v1_user201mp.json`, `data/eval/tiny_v2_20k_user201mp.json`.
Soup baseline/v2 (25% и 50%) отклонён: LPIPS 0.633/0.661 хуже обоих parents,
поэтому context interpolation здесь не работает. PICO soup50 в отличие от этого дал
ожидаемый промежуточный RD-баланс и сохранён как кандидат.

Tiny-v3 10k на полном корпусе: 0.12443bpp, 23.630dB, MS-SSIM 0.82269, LPIPS 0.45166;
matched 0.1261bpp у 5k даёт LPIPS 0.4543. Это намного лучше baseline perceptual, но
MS-SSIM ниже 0.8561. Запущен Tiny-v4 rebalance; на 5k 23.672dB, MS-SSIM 0.82589,
LPIPS 0.46002. Финальный Tiny-v4 10k: matched 0.12616bpp, 23.672dB, MS-SSIM 0.82790,
LPIPS 0.45704. Финальный scale booster (2048 samples/4000 steps): encode 0.1257с
против 0.2316с (1.84×), 0.12477bpp против 0.12508, LPIPS 0.45824 против 0.45886;
PSNR 23.635 против 23.676. OXNV интегрирован в UI/viewer, resident preload, target bpp
и legacy OXNT проверены. Trainer: `ox_scale_booster.py`.

## Что уже реализовано

- `bench_research.py`: bpp, PSNR, MS-SSIM, LPIPS, RGB delta, boundary MSE,
  encode/decode time, память.
- `ox_data.py`: ленивые EXIF-aware кропы, рекурсивные корни, holdout exclusions.
- Общие losses: multi-resolution TAL, per-patch color balance, seam continuity,
  gradient-saliency text/detail fidelity. PICO также подавляет GAN в salient areas.
- Новые checkpoints сохраняют args и размер корпуса.
- OXNT отделяет новые tiny checkpoints от legacy OXN9. CLI/UI/viewer поддерживают
  OXNT и старые OXN9.
- Target bpp для tiny/PICO через бинарный q-search, включая UI.
- PICO ConvScale сворачивается в обычные веса при runtime initialization.
- PICO training исправлен для batch >1: level broadcasting и нормировка bpp.
- UI/viewer освобождают CUDA workspace после операций, не выгружая resident weights.
- Модели из селектора UI резидентные при достатке памяти; decode-only контексты
  подгружаются/выгружаются. При OOM resident mode отключается.
- Viewer переключает модель по magic файла.
- PICO v4 soup50 интегрирован как `pico2`/OXP6 с явным fast flag; UI/viewer и legacy
  OXP5 roundtrip проверены. На 201-Мп holdout OXP6 FP64: encode 46.1 с, decode 40.7 с,
  peak RSS 3.12 ГиБ, CUDA reserve 11.13 ГиБ, 1.0949bpp, 33.835dB при q75. Зависания
  и роста RSS нет; matched low-rate сравнение остаётся отдельным тестом.

## Корпус

Пилот: 1026 изображений, недостаточно для Apple-scale результата. Не путать пилот с
полноценным retrain.

Holdout: пользовательский 201-Мп кадр `IMG_20261015_111154.jpg` и три benchmark
screenshots исключены из обучения. Kodak21–24 и официальные PICO-сцены не используются
в обучении/подборе модели.

Подготовлено/собирается:

- DIV2K train HR: 800 изображений, MD5 проверен, архив удалён после распаковки.
- `data/train` и `/home/agemlex/Pictures` рекурсивно. Pictures: 128 валидных
  изображений после holdout/size exclusions.
- `data/text_synth`: 2300 текстовых/UI изображений, seed 42.
- COCO train2017: 118287 разнообразных изображений, ZIP проверен, распакован и удалён.
- Полный проверенный корпус: 120799 изображений. PICO-v3 запущен на 100k шагов из
  PICO-v2 с теми же losses; это отделяет эффект корпуса от изменения архитектуры.
  После RD-кривой 0.2/0.5/1.0bpp MSE остановлен на 30k: high-rate улучшения подтверждены,
  а low-rate floor остался. Сохранён `oxn_pico_v3_coco_30k.pt`; ресурсы переключены на
  30k log-uniform perceptual fine-tune (`oxn_pico_v4_perceptual.pt`) вместо неэффективного
  продолжения одинакового uniform MSE. Perceptual остановлен на 6k: Kodak 0.5bpp
  LPIPS 0.2172 против 0.2816 у MSE, MS-SSIM 0.9319, PSNR 27.57. Apple LPIPS 0.2999
  против 0.3633. Low-rate floor остался 0.295bpp, поэтому запущен отдельный 15k этап
  с rate_weight=100 из сохранённого `oxn_pico_v4_perceptual_6k.pt`.

Цель корпуса: сначала 25–50k, затем 100k+ с отдельными текстовыми и high-res
группами. Для Apple указаны примерно 90k generic + 2.3k text + 28k high-res.

## Протокол экспериментов

1. Менять один существенный элемент за эксперимент, инициализировать из выбранной
   базы и не перезаписывать рабочие checkpoints.
2. Оценивать одинаковые изображения, q или target bpp, PSNR, MS-SSIM, LPIPS,
   RGB shift, seam/boundary error, скорость и память.
3. Не требовать улучшения буквально всего любой ценой: заранее задать допустимый
   bpp/runtime budget и обязательные качественные критерии.
4. Принимать только измеренное улучшение или явное доминирование baseline.
   Плохие кандидаты откатывать с записанной причиной.
5. Новые entropy/context несовместимости маркировать новым magic/version.
6. Проверять real rANS roundtrip, UI/viewer и CPU/GPU пути, не только forward/loss.

## Этап Tiny-v2 и Tiny-v3

Tiny-v2: текущая архитектура `dim=48`, `latent=8`, один block, refine. Больше данных,
TAL/color/seam/text losses. После 20k сравнить со baseline; MS-SSIM/LPIPS тоже нужно
контролировать.

Если блочность остаётся, Tiny-v3 проверяет по одному:

- `latent=12`;
- `dim=64`;
- два блока;
- более сильный conv/refine decoder или overlapped/conv patch transitions.

Принимать рост capacity только при доказанном RD-выигрыше (ориентир +0.5 dB при
сопоставимом bpp и приемлемой скорости).

Для скорости: predictor scale map вместо полного ladder, меньше reconstruction
passes, пакетные независимые этапы, native integer entropy и refine tiling.

## Этап PICO retrain

- Сначала MSE/structure, затем LPIPS + MS-SSIM + warm-started GAN + TAL +
  TextFidelityLoss. Контролировать, чтобы perceptual качество не покупалось ростом bpp.
- После корпуса: capacity/decoder experiments и небольшой NAS по реальному GB10 runtime.
- Deployment: репараметризация ConvScale/Haar, детерминированное квантование scale
  decoder, компиляция, меньше CPU↔GPU sync, нормальный CPU/GPU pipeline.
- Первая цель скорости 12 Мп: <1 с encode и <0.75 с decode. Apple 0.23/0.15 с —
  следующий ориентир, а не обещание первого эксперимента.
- PICO-v2 на 12-Мп official holdout: FP64/whole encode 1.87 с, decode 0.69 с;
  TF32/tiled encode 1.17 с, decode 1.07 с. TF32 не принят как универсальный default:
  суммарный выигрыш небольшой, decode хуже, bpp 1.67986 против 1.67965. Нужна оптимизация
  tile/sync overhead. Метрики: `data/eval/pico_v2_12mp_fast_comparison.json`.

## Математические идеи для Tiny и PICO

Базовая цель:

`L = D(x,x_hat) + lambda * R(z) + lambda_c * color + lambda_s * seam`

`R(z)` — фактические биты entropy coder, не просто L1 latent.

### 1. Ортогональные latent transforms

`u = Q z`, обратно `z = Q^T u`, `Q^T Q = I`.

Строить Q из Givens rotations или Householder reflections. Точная обратимость,
det(Q)=1, нет скрытого масштабного изменения энтропии. Для latent=8 стоимость мала.
Цель — повернуть каналы так, чтобы квантованные компоненты были менее коррелированы
и лучше распределяли энергию. Не гарантирует выигрыш при сильном context model.

### 2. Каузальное whitening/Cholesky-преобразование

`r_c = z_c - sum_{j<c} a_cj z_j`.

A нижнетреугольная и может зависеть от patch/context. Каналы кодируются по
возрастанию, восстанавливаются в обратном порядке без пространственной AR.

Это каузальное PCA/Cholesky-преобразование. Дополнительный candidate loss:

`L_cov = ||Cov(r) - diag(Cov(r))||_F^2`.

### 3. Companding / неравномерное квантование

Кандидаты:

- `u = asinh(z / a)`;
- `u = sign(z) * log(1 + |z| / a)`.

Квантуется u, декодер применяет обратную функцию. Параметр a обучается по каналу
или context. Больше уровней около нуля, меньше на хвостах. Согласуется с реальными
распределениями latent и learned quantization width.

### 4. Generalized Gaussian и mixtures

`p(r) ∝ exp(-|r/sigma|^beta)`, beta обучается по каналу/group.

- beta=1 — Laplace;
- beta=2 — Gaussian;
- промежуточные значения описывают реальные хвосты.

Следующий вариант — mixture of two Gaussians с обучаемой долей компонент. CDF
дискретизируется в canonical tables, encode/decode остаются синхронизированными.
Требуется новая версия bitstream и проверка фактического rANS gain.

### 5. Water-filling / rate allocation

Для Gaussian RD: `R_i = max(0, 0.5 * log2(sigma_i^2 / lambda))`.

Использовать variance/energy патчей и каналов для инициализации распределения бит,
затем запускать полный decoder только для 2–3 лучших кандидатов вместо всей ladder.
Это приближение для нелинейного декодера, но потенциально ускоряет encoder без
изменения bitstream и качества.

### 6. Reversible spatial/spectral transforms

- Haar/DCT/Laplacian pyramid для разделения низкочастотной основы и разреженных
  остатков;
- learned lifting с нормировкой и точной обратимостью;
- quantization-aware covariance/whitening остатка после context, а не обязательно
  исходного latent.

### 7. Что пока отложить

- E8/A16 lattice quantization: интересно, но сложность nearest-neighbor и coding
  indices пока не оправдана. Сначала companding + causal whitening.
- CLUB/mutual-information estimators: сначала простые covariance residuals.
- Сложные physics-inspired priors без явной обратимой операции и измеримого выигрыша.

## Идеи из физики

Использовать только как формальные priors/операции:

- scale-space/diffusion и Laplacian pyramid: низкие частоты отдельно от разреженных
  деталей;
- сохранение энергии через orthogonal/wavelet transforms;
- локальные поля соседства для плоских областей/UI с переключением по контенту;
- кодирование изменений/остаточной энергии вместо самого поля.

Никаких декоративных аналогий без проверки RD.

## Порядок математических экспериментов

1. Water-filling для скорости encoder без изменения bitstream. Первый pruning 7→3
   ближайших к q кандидатов проверен и отклонён: encode 0.244 с против 0.318 с,
   PSNR/MS-SSIM/LPIPS улучшились, но bpp вырос с 0.11938 до 0.17327 (+45%). Причина —
   глобальный shortlist исключил низкие scale для гладких патчей. Нужен content-aware
   predictor, а не фиксированные соседи q. Полный ladder побайтово не изменён.
   Существующий `ox_boost.pt` несовместим с Tiny latent=8 (ожидает 44 входных канала,
   Tiny даёт 28), поэтому не используется. Для content-aware ускорения booster нужно
   переобучить именно на принятом Tiny checkpoint.
2. Companding для Tiny. OXNU/asinh prototype проверен с a=0.25, 0.5, 1, 2 и
   отклонён без retrain: около 0.18 bpp против 0.11938, PSNR и LPIPS хуже во всей сетке.
   Нелинейный latent требует согласованного entropy/model обучения; текущий приоритет
   полный perceptual retrain, а не новый несовместимый bitstream.
3. Givens latent rotation для Tiny-v3.
4. Каузальная нижнетреугольная матрица/whitening.
5. Generalized Gaussian/mixture entropy с новой версией bitstream.
6. Laplacian/wavelet architecture, если блочность остаётся.

## Источники гипотез

- Apple PICO, architecture/training/TAL/TextFidelityLoss/NAS:
  https://arxiv.org/html/2605.05148v1
- Apple research summary:
  https://machinelearning.apple.com/research/compression
- Официальные PICO reconstructions/bpp:
  https://github.com/apple/ml-pico
- DCVC-RT: operational complexity (memory I/O, function calls), single low-res latent,
  integerization и module-bank rate control:
  https://arxiv.org/html/2502.20762v2
- Rate-controllable LIC Using Channel Attention: target bpp как side information,
  двухстадийное обучение:
  https://proceedings.mlr.press/v278/xie25b.html

Интернет используется как источник гипотез, решение всегда принимается по локальному
matched-bpp benchmark.

## Что не повторять

- BF16 PICO: был быстрым, но PSNR упал с 21.37 до 12.9–15.9 dB, поэтому отклонён.
- Halo 128: сильно уменьшил seam MSE, но почти не улучшил PSNR и резко увеличил bpp.
  Auto halo теперь 64, `--overlap 0` доступен для min bitrate.
- Не оценивать коэффициент сжатия только относительно уже сжатого JPEG/PNG.
  Сравнивать bpp и RD на одинаковых исходниках.
- Не запускать полноценный PICO retrain на 1026 изображениях: это пилот, не финальный
  корпус.

## Следующая серия: capacity и low-rate allocation

- Tiny dim=64: остановлен на 5k и отклонён — 0.14484bpp, 21.537dB, MS-SSIM 0.7491,
  LPIPS 0.6229 против 0.12508bpp, 23.676dB, 0.8265, 0.4589 у Tiny-v4; decode также
  медленнее. Переход к latent=12 поэтому не запускался.
- PICO channel mask: добавлена sigmoid level/channel mask поверх quantization gain.
  В отличие от inverse-only gain, masked каналы действительно удаляются из latent и не
  делятся обратно при synthesis. Инициализация ≈1, старые модели не меняются. Smoke
  training и roundtrip прошли. Гипотеза основана на target-rate channel attention:
  https://proceedings.mlr.press/v278/xie25b.html и module-bank rate control:
  https://arxiv.org/html/2502.20762v2. Принятие только по фактическому bpp/RD.

### Selective frozen-good / dynamic-LR fine-tune (предложение пользователя)

Гипотеза: сохранить хорошие веса принятой модели, а остальным позволить адаптироваться
с LR на 1–2 порядка меньше; после каждого батча снижать LR при ухудшении и осторожно
увеличивать при улучшении. Реализовывать не как жёсткую реакцию на шумный батч, а как:

- parameter groups: scale_decoder/context/entropy-sensitive веса заморожены или имеют
  сильный trust-region penalty к checkpoint, synthesis/predictor/allocation свободнее;
- EMA для PSNR/MS-SSIM/LPIPS/bpp и RD-score, dynamic LR только по сглашенному тренду;
- отдельный rollback к лучшему checkpoint, если matched validation деградирует;
- вариант EWC: `L_anchor = sum_i importance_i * (w_i - w_ref_i)^2`, importance можно
  оценить по чувствительности entropy/RD, а не назначать декоративно.

Это plausible training recipe, но не подтверждённый факт о внутреннем обучении Apple.
Проверять после channel allocation на PICO и, если потребуется, на Tiny MS-SSIM.

### Sparse residual bitstream (OXP7)

Gaussian zero residual стоит до 3.37 бита, поэтому channel mask без sparse entropy не
работал. OXP7 кодирует packed/zlib bitmap ненулевых residuals и обычный rANS только для
nonzero; для каждого stream выбирается меньший из plain/sparse вариантов. Плотные модели
почти не меняются (1 byte/stream overhead при выборе plain). Helper, empty/nonempty
roundtrips и полный OXP7 roundtrip проверены. Для раннего residual-mask candidate bpp
снизился с ~0.18 до 0.07658. Запущен новый `oxn_pico_v5_sparse.pt` с согласованной
sparse rate loss, mask_lr=0.001, умеренным w_mask=5 и freeze_entropy. Качество ещё нужно
оценить; формат не интегрирован в UI до принятия.

### Sparse + selective frozen analysis: первый RD результат

Первый вариант sparse training без заморозки analysis обвалил качество; freeze_entropy +
freeze_analysis стабилизировал его. `oxn_pico_v5_sparse_frozen_1k.pt` при OXP7:

- 0.13319bpp: 22.078dB / MS-SSIM 0.72196 / LPIPS 0.46923;
- 0.19655bpp: 23.809dB / 0.81350 / 0.40215;
- 0.49917bpp: 26.372dB / 0.90443 / 0.26199.

Low-rate улучшение над soup50 (0.246bpp) около 20% бит при −0.45dB и лучшем LPIPS.
High-rate пока хуже soup50, продолжается recovery до 10k. Mask пока полностью открыта,
поэтому выигрыш получен от selective synthesis и sparse entropy, не от удаления каналов.
`torch.compile` predictor и thread-per-stream async rANS отклонены: первый меняет bitstream,
второй замедляет encode примерно втрое.

### PICO v5 sparse frozen 7k — принятый low-rate профиль

`oxn_pico_v5_sparse_frozen_7k.pt` выбран вместо более слабого 10k и интегрирован как
`pico5`/OXP7 в UI/viewer. PICO v4 остаётся high-rate профилем.

- Apple b1400 target 0.095bpp: достигнуто 0.0947bpp, 30.342dB, LPIPS 0.2664; v4 soup50
  требовал 0.1996bpp для 30.507dB и 0.3255. Apple всё ещё лидирует по perceptual метрикам.
- Все Apple scenes: 0.28933bpp против 0.290 target, 25.581dB, MS-SSIM 0.88609,
  LPIPS 0.31565; v4: 0.32234bpp, 25.928dB, 0.89650, 0.32351.
- Kodak 0.20070bpp: 23.844dB, MS-SSIM 0.81713, LPIPS 0.40608.
- OXP7 UI/viewer target 0.1/0.2 и legacy переключение проверены. Sparse helper проверен
  на пустых и непустых потоках; plain/sparse выбор предотвращает регресс плотных streams.

Идея selective freeze от пользователя оказалась ключевой: без freeze_analysis sparse rate
обваливал reconstruction, с замороженными analysis/predictor/entropy получен рабочий RD.

Дополнительные проверки PICO v5:

- Screenshots: 0.20189bpp, 26.825dB, MS-SSIM 0.93630, LPIPS 0.29740 против v4
  0.23715bpp, 25.902dB, 0.91410, 0.37353 — улучшение сразу по всем метрикам.
- 201-Мп OXP7 q10: 0.16511bpp, 23.940dB, encode 28.1с, decode 27.2с,
  peak RSS 3.10ГиБ, CUDA reserve 11.17ГиБ. Multi-tile sparse roundtrip прошёл.

### Multi-rate v6 и исправление sparse rate loss

Перед объединением обнаружена ошибка в training-only sparse rate: scalar bits группы
ошибочно умножались на spatial mask, завышая оценку примерно в 64 раза. OXP7 bitstream
не был затронут; v5 остаётся валидным, но это объясняет прошлые обвалы quality. После
исправления v5 forward оценивает ~0.13bpp на q1 и ~2.29bpp на q100 вместо сотен bpp.

`oxn_pico_v6_multirate.pt`: старт из v5 7k, 50/25/25% low/mid/high batches, v4 teacher
для q>=14, anchor к v5 low и v4 high, freeze_analysis/entropy, mask_lr=1e-4,
w_anchor=w_distill=1000. Цель — общий sparse OXP7 без отдельных low/high моделей.

### Multi-rate v6: итог и auto profile

`oxn_pico_v6_multirate_5k.pt` с distillation/anchors сохранил low-rate (0.2069bpp,
23.741dB, LPIPS 0.4209), но high-rate остался 28.019dB при ~1bpp против 29.50 у v4.
Stage2 unfreeze увеличил low-rate bpp до 0.256 и не помог high-rate, поэтому остановлен.
Soup v4/v6 также не восстановил high-rate. Общий checkpoint не принят.

Практическое решение лучше: `picoauto` в UI автоматически выбирает v5 при target≤0.25bpp
(или q≤10) и v4 выше. Auto 0.2→OXP7, auto 0.5→OXP6 и roundtrips проверены. Это сохраняет
лучший measured RD каждого профиля вместо навязывания более слабой общей модели.

## PICO MoE v7: общий пул и synthesis experts

Собрана `oxn_pico_v7_moe.pt`: shared analysis/predictor/scale/context/hyper и 3 synthesis
experts low/mid/high. Gate детерминирован по level: 0–1 low, 2–4 mid, 5–7 high; одновременно
работает один expert, поэтому compute остаётся как у одной PICO, параметров 8.87M вместо
6.81M. Исходные low/high reconstruction совпадают с профилями побайтово и попиксельно.
Важное исправление STE: `hard + (soft - soft.detach())` исключает float-хвосты вместо 1.

`oxn_pico_v7_moe_tuned.pt` обучал middle на full/half crops и немного улучшил mid RD.
UI/viewer используют `picomoe`/OXP8; low/mid/high roundtrips прошли.

Дальше: `oxn_pico_v7_moe_pretrain.pt` — общее pretraining всех experts с multi-rate batches,
selective LR, anchors и teacher augmentation. High-rate teacher reconstruction подаётся как
входная синтетика, оригинал остаётся ground truth; 25% high/mid batches. После него будет
специализация experts и сравнение с Apple PICO. Tiny не изменяется.

## PICO v8 MoE final: 6 experts / OXP9

`oxn_pico_v8_moe6_final.pt` собран из low1, middle и high perceptual experts. Общий
analysis/entropy пул, один активный synthesis expert по target bpp. Ultra-low использует
2x2 spatial anchors. Все expert IDs и UI/viewer/resident roundtrips прошли.

- Apple scenes: 0.31932bpp, 26.924dB, MS-SSIM 0.91442, LPIPS 0.27475.
- Screenshots: 0.19990bpp, 27.409dB, MS-SSIM 0.94073, LPIPS 0.29265, лучше v5 по всем
  метрикам (0.20189bpp, 26.825dB, 0.93630, 0.29740).
- PICO_MOE_CKPT назначен на финал. Это улучшение прежних профилей, но не победа над
  perceptual RD Apple; не заявлять её без дополнительных результатов.

## Приоритетный план оптимизации скорости PICO/MoE

Скорость не определяется количеством параметров. В текущем OXP9 доминируют FP64 CPU,
последовательная entropy и CPU↔GPU синхронизации, а не активный synthesis expert.

Порядок работы:

1. Профилировать encode/decode 12 Мп по этапам: analysis, scale decoder, context/predictor,
   rANS, sparse bitmap, tiled assembly. Не оптимизировать вслепую.
2. Перевести scale decoder с CPU FP64 на integer/fixed-point native kernel. Сохранить
   каноническую синхронизацию encoder/decoder и побайтовый bitstream. Это главный кандидат.
3. Убрать CPU↔GPU синхронизации между 4 spatial context этапами; сделать batched native rANS
   и уменьшить количество/стоимость передач. Не нарушать последовательную зависимость.
4. Уменьшить tile overhead: однотайловый путь уже не должен создавать thread/stream;
   дальше оптимизировать assembly, выделения и упаковку sparse bitmap.
5. Компилировать только reconstruction/predictor, если численный результат и bitstream
   побайтово не меняются. Прежний torch.compile predictor изменил bitstream и был отклонён.
6. Не возвращаться к thread-per-stream async rANS: замедление было примерно втрое.
7. После каждого изменения проверять 12 Мп encode/decode, real OXP9 roundtrip, побайтовый
   entropy и RD. Цель Apple: 0.23/0.15 с; текущая финальная MoE: 2.07/1.61 с.

MoE добавляет память для experts, но только один активен; десятикратная просадка скорости
не связана с числом experts. Сначала integer scale decoder, затем native entropy/sync.

### Результат оптимизации OXP9 на GB10

Профиль 4000×3000 показал, что в старом default доминировали analysis (~0.90с), CPU FP64
scale (~0.55с), synthesis (~0.51с) и sparse/plain encode (~0.44с); четыре predictor вместе
занимали только ~0.03с, а entropy decode ~0.05с. Поэтому spatial dependency не была главным
узким местом и не потребовала новой архитектуры.

Принято без изменения весов и rate gate:

- ARM/CUDA fast с GPU scale decoder для всех размеров, не только маленьких кадров;
- contiguous synthesis вместо channels-last на GB10;
- checkerboard как strided slices, без mask CPU transfers и полноразмерных `where`;
- hyper entropy параллельно GPU scale/predictor, четыре независимых spatial entropy streams
  параллельно после GPU-части; pinned copies с одной синхронизацией;
- пустые spatial streams не кодируются дополнительно полным plain rANS;
- GPU RGB clip/quantization и перенос uint8 вместо float32;
- non-MoE OXP6/OXP7 снова вызывают quantization/synthesis без лишнего expert argument.

12-Мп q20: 1.366/0.946с encode/decode против 2.07/1.61с. 201-Мп q10: 22.313/14.169с
против 27.30/21.08с; bpp и PSNR совпали. Apple OXP9 streams побайтово совпали со старыми,
RD изменения только в последних знаках. FP64 old path побайтово сохранён.

Отклонено по измерениям: dense expansion grouped conv (почти нет выигрыша), HaarDown/HaarUp
fusion (нестабильный/нулевой выигрыш и численные изменения), cuDNN benchmark (~40с первого
поиска без ускорения), torch.compile analysis/synthesis (0.1–0.3с потенциально, но 43–72с
первой компиляции и численные изменения), два CUDA streams на 2048/1536/1024 tiles (не
быстрее whole, bpp 0.144–0.159 против 0.134). До Apple 0.23/0.15с всё ещё большой gap;
следующий реальный скачок требует AOT/native integer или tensor-core convolution pipeline,
а не дальнейшей оптимизации четырёх predictor/rANS этапов.
