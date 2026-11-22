# PICO/OXP9: бриф для оптимизации скорости

Дата состояния: 2026-11-22.

Инструкция воспроизведения и публикация: [`PICO_REPRODUCE.md`](PICO_REPRODUCE.md).

Этот документ описывает текущую рабочую реализацию PICO MoE, целевое железо, измеренную
производительность и узкие места. Он предназначен для выработки новых идей оптимизации.
Это независимая реализация архитектуры из статьи Apple PICO, а не код или веса Apple.

## 1. Цель и ограничения

Главная текущая задача — уменьшить encode/decode latency без ухудшения фактического bpp,
PSNR, MS-SSIM и LPIPS. Приоритет — скорость на ARM64 + NVIDIA GB10. Обратная совместимость
новых экспериментальных потоков не обязательна, но существующие OXP6–OXP9 должны оставаться
читаемыми. Tiny и другие кодеки не входят в эту задачу.

Текущая принятая модель: `oxn_pico_v8_moe6_final.pt`.
Реализация: `oxn_pico.py`.
Формат: OXP9, sparse residual coding, явный ID synthesis expert.
Менять веса или архитектуру допустимо, если реальный matched-bpp benchmark доказывает
улучшение. Простое увеличение скорости ценой заметного роста bpp или снижения качества не
считается успехом.

## 2. Целевое железо и программное окружение

- CPU: ARM64, 20 физических ядер:
  - 10 × Cortex-X925, до 3.9 GHz;
  - 10 × Cortex-A725, до 2.808 GHz.
- GPU: NVIDIA GB10, compute capability 12.1.
- 48 streaming multiprocessors, 24 MiB L2.
- GPU memory: около 124.6 GB; система сообщает 121 GiB RAM, память фактически общая для
  CPU/GPU на этой платформе.
- CUDA 13.0, PyTorch 2.14.1+cu130.
- ОС: Linux aarch64.
- Для CPU fallback желательно использовать параллелизм независимых задач, но не создавать
  дорогой thread-per-symbol или thread-per-tiny-stream overhead.

Apple PICO оптимизирована под собственное Apple/ARM/ANE окружение. Её цифры являются
ориентиром, а не гарантированно идентичным аппаратным сравнением с GB10.

## 3. Архитектура текущей модели

### 3.1. Общая схема

Вход RGB преобразуется analysis-сетью в latent `y` с 192 каналами и пространственным
разрешением примерно H/16 × W/16. Дополнительная hyper-ветвь создает представление с
разрешением H/64 × W/64. Из quantized hyper-latent восстанавливаются:

- `sigma` для entropy model через scale decoder;
- `prior` для predictor через context decoder.

Predictor получает `prior`, уже восстановленные spatial phases и 8-мерный level vector.
Он выдает residual center `mu` и quantization width для каждого latent-элемента.

Reconstruction synthesis восстанавливает RGB из деквантованного latent. Используется один
из шести synthesis experts; остальные experts в этот момент не вычисляются.

### 3.2. Analysis

- HaarDown 3 → 64.
- Level conditioning: 8 каналов добавляются к features.
- ConvScale311-блоки с GELU и learned scales.
- Последовательные HaarDown и feature blocks: 64 → 96 → 96 → latent 192.
- Hyper-ветвь: 96 → 128 → 192 → 192.
- ConvScale при runtime репараметризуется: input/output scales заранее вшиваются в веса.
- Часть 3×3 ConvScale — grouped convolutions, обычно 2–4 группы.

### 3.3. Spatial context

Есть четыре фазы 2×2 checkerboard:

1. `(even y, even x)`;
2. `(even y, odd x)`;
3. `(odd y, even x)`;
4. `(odd y, odd x)`.

Каждая следующая фаза использует уже восстановленные предыдущие. Поэтому четыре predictor
прохода причинно последовательны; их нельзя просто выполнить одновременно без изменения
entropy model и переобучения. Это четыре крупных пространственных прохода, а не
pixel-by-pixel autoregression.

В текущем codec-пути маски заменены strided slices `a::2, b::2`. Это устранило переносы
масок на CPU и полноразмерные `where`.

### 3.4. MoE rate control

- Общие analysis, predictor, scale decoder, context decoder и entropy tables.
- Шесть synthesis experts, активен только один.
- Expert выбирается явно по target bpp:
  - 0: ultra-low, `<0.075 bpp`;
  - 1: low, `0.075–0.15`;
  - 2: mid-low, `0.15–0.3`;
  - 3: mid, `0.3–0.75`;
  - 4: high, `0.75–1.5`;
  - 5: ultra-high, `>1.5`.
- Ultra-low expert дополнительно использует spatial anchor base layer 2×2: residual symbols
  записываются только на anchor-позициях, остальные позиции восстанавливаются predictor'ом.
- MoE увеличивает хранимые параметры, но не запускает шесть synthesis-сетей одновременно.

### 3.5. Entropy и sparse coding

- Параметрический rANS, нативная C-реализация с frozen canonical tables.
- Для OXP9 дополнительно хранится packed bitmap ненулевых residuals, сжатый zlib level 9.
- rANS кодирует только nonzero residuals.
- Для непустого stream вычисляются sparse и plain варианты, выбирается меньший.
- Для большого полностью нулевого stream plain rANS больше не вычисляется.
- Hyper entropy выполняется на CPU одновременно с GPU scale/predictor.
- Четыре независимых spatial entropy encode jobs запускаются параллельно после GPU-части.
- Decode rANS остается последовательным внутри каждого stream; это свойство текущего
  bitstream, а не случайное ограничение Python.

### 3.6. Tiling и память

- Если изображение помещается в вычисленный memory budget, оно обрабатывается целиком.
- Для больших изображений размер tile вычисляется по свободной RAM/VRAM, worker count и
  оценке working set.
- Максимальный tile размер масштабируется по SM count и clock GPU относительно GB10.
- По умолчанию halo до 64 пикселей; halo удаляется при сборке.
- Для multi-tile допускаются два CUDA streams.
- Практический результат: на 201-Мп используется tile 2048, но принудительная нарезка 12-Мп
  на маленькие tiles оказалась невыгодной.

## 4. Текущий runtime-путь

На ARM/CUDA fast mode теперь применяется для любого размера изображения:

- analysis использует TF32 channels-last convolutions;
- scale decoder работает на GPU FP32 вместо CPU FP64;
- synthesis использует contiguous layout, поскольку на GB10 он быстрее channels-last;
- spatial phases используют slices без CPU mask transfers;
- GPU/CPU copies для spatial symbols используют pinned buffers и одну синхронизацию;
- независимые entropy encode jobs выполняются параллельно GPU-работе или друг с другом;
- final RGB clip/quantization выполняется на GPU;
- на CPU копируется uint8 RGB, а не float32: для 12-Мп это 36 MB вместо 144 MB.

Старый FP64 path сохранен и проверен. Fast OXP-потоки не claims cross-device determinism.

## 5. Текущие измеренные показатели

### 5.1. Скорость до/после последней оптимизации

4000×3000, q20, OXP9, пять измеренных runs после warmup:

| Метрика | До | Сейчас |
|---|---:|---:|
| Encode | 2.07 s | 1.366 s |
| Decode | 1.61 s | 0.946 s |
| Bitrate | 0.134012 bpp | 0.134012 bpp |

201 Мп, 12288×16384, q10, OXP9:

| Метрика | До | Сейчас |
|---|---:|---:|
| Encode | 27.30 s | 22.313 s |
| Decode | 21.08 s | 14.169 s |
| Bitrate | 0.0879519 bpp | 0.0879519 bpp |
| PSNR | 20.459 dB | 20.459 dB |
| Peak process RSS | около 3.17 GiB | около 3.33 GiB |

Для 12-Мп benchmark: peak CUDA reserved около 10.85 GB, process peak RSS около 2.24 GB.
Для 201-Мп: peak CUDA reserved около 12.47 GB.

Ориентир Apple PICO для похожего 12-Мп сценария: около 0.23 s encode и 0.15 s decode.
Текущий gap все еще примерно 5.9× encode и 6.3× decode. Победа над Apple по скорости не
достигнута.

### 5.2. Качество принятой модели

Apple scenes, среднее:

- 0.31932 bpp;
- PSNR 26.924 dB;
- MS-SSIM 0.91442;
- LPIPS 0.27475.

Screenshots, среднее:

- 0.19990 bpp;
- PSNR 27.409 dB;
- MS-SSIM 0.94073;
- LPIPS 0.29265.

После runtime-оптимизации все четыре Apple-scene OXP9 streams остались побайтово теми же.
Изменения PSNR/MS-SSIM/LPIPS находятся только в последних отображаемых десятичных знаках.
Все шесть experts прошли реальный roundtrip. Старые OXP6/OXP7/OXP8/OXP9 декодируются.

### 5.3. Текущее сравнение с оригинальной Apple PICO

Источники оригинальных показателей: [статья Apple/CVPR 2026](https://arxiv.org/html/2605.05148v1)
и [официальные reconstruction images/bpp](https://github.com/apple/ml-pico). Apple не публикует
encoder/decoder или веса, поэтому локальный roundtrip оригинальной PICO невозможен. Метрики
качества OXP9 вычислены относительно тех же reference images; показатели Apple вычислены по
её официальным reconstruction PNG. Скорость измерена на разном железе и является ориентиром,
а не строгим like-for-like benchmark.

| Параметр | Apple PICO | Текущий OXP9 | Разница / комментарий |
|---|---:|---:|---|
| 12-Мп encode | 0.230 s, iPhone 17 Pro Max | 1.366 s, GB10 | OXP9 в 5.94× медленнее |
| 12-Мп decode | 0.150 s, iPhone 17 Pro Max | 0.946 s, GB10 | OXP9 в 6.31× медленнее |
| 12-Мп encode + decode, сумма | 0.380 s, производная сумма | 2.312 s, производная сумма | OXP9 в 6.08× медленнее |
| Всего параметров | не опубликовано | 11 969 456 | Shared pool + 6 experts |
| Активных параметров при reconstruction | не опубликовано | 6 809 396 | Shared pool + 1 expert |
| Размер checkpoint | не опубликовано | 48 282 977 байт, 46.05 MiB | `oxn_pico_v8_moe6_final.pt` |
| 12-Мп peak process RSS | не опубликовано | около 2.24 GB | Локальный benchmark |
| 12-Мп peak CUDA reserved | не применимо к iPhone | около 10.85 GB | Не равно фактически используемой памяти |
| Rate control | fine-grained, по статье | 8-мерный level + 6 target-bpp experts | У OXP9 один активный expert |
| Cross-platform entropy | заявлена bit-exact scale decoder | FP64 legacy path; fast не гарантирует cross-device determinism | Разные deployment guarantees |
| Корпус обучения | точный размер не использован в этом сравнении | 120 799 изображений | Holdout images исключены |
| Субъективное BD-rate исследование | 2.3–3× экономии против AV1/AV2/VVC/ECM/JPEG-AI | не проводилось | Нельзя переносить вывод на OXP9 |

Качество и bitrate по официальным сценам. Для LPIPS меньше лучше, для остальных quality
метрик больше лучше. OXP9 не везде достиг точно такого же bpp, поэтому таблица показывает
фактические точки, а не идеально matched-bpp BD-rate.

| Scene | bpp Apple | bpp OXP9 | PSNR Apple | PSNR OXP9 | MS-SSIM Apple | MS-SSIM OXP9 | LPIPS Apple | LPIPS OXP9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| philipp-reiner-207 | 0.341 | 0.35946 | 25.266 | 24.198 | 0.95424 | 0.88563 | 0.05968 | 0.29185 |
| b1400e99ef413c8fee7d5c0be41d7d1e | 0.095 | 0.11439 | 34.886 | 32.322 | 0.99003 | 0.97161 | 0.01624 | 0.25799 |
| alexander-shustov-73 | 0.507 | 0.50708 | 25.468 | 22.530 | 0.96119 | 0.88350 | 0.06107 | 0.29315 |
| kodim23 | 0.217 | 0.29635 | 32.572 | 28.647 | 0.97348 | 0.91693 | 0.02774 | 0.25599 |
| **Среднее** | **0.29000** | **0.31932** | **29.548** | **26.924** | **0.96973** | **0.91442** | **0.04118** | **0.27475** |

Средний gap: OXP9 использует на 10.11% больше бит, имеет на 2.624 dB меньший PSNR, на
0.05531 меньший MS-SSIM и на 0.23357 больший LPIPS. Apple PICO сейчас лучше по скорости и
по perceptual RD. Локальные результаты не следует представлять как победу над Apple.

## 6. Узкие места по убыванию важности

### 6.1. Analysis encoder: grouped FP32/TF32 ConvScale — главная проблема

Профиль 4000×3000 до последних мелких улучшений:

- analysis целиком: примерно 0.85–0.90 s;
- `e12`: около 0.206 s;
- `e22`: около 0.243 s;
- `e11`: около 0.108 s;
- `e21`: около 0.081 s;
- HaarDown и остальные analysis stages заметно меньше.

Таким образом, analysis занимает около двух третей текущего encode. Наиболее дорогие
операции — grouped 3×3 convolutions. Generic PyTorch/cuDNN grouped path плохо использует
возможности GB10/tensor cores. Наибольший ожидаемый выигрыш: специализированный native/AOT
grouped-convolution pipeline с разумным fusion и контролируемой численностью.

Простое превращение grouped conv в dense conv с нулевыми весами уже проверено и почти не
помогло: дополнительные FLOPs съедают преимущество dense kernel.

### 6.2. Synthesis decoder — главная проблема decode

До layout-оптимизации synthesis занимал примерно 0.51 s в FP64/default path и около 0.75 s
в неудачном channels-last fast path. После перехода synthesis на contiguous decode стал
около 0.90–0.95 s в стабильном теплом benchmark.

Synthesis содержит четыре HaarUp-уровня и ConvScale311 blocks. Это второй по значимости
кандидат для native fused kernels. Для общего ускорения encode+decode оптимизированный
convolution pipeline должен покрывать и analysis, и synthesis.

### 6.3. Sparse bitmap и encode-side entropy

Старый последовательный sparse/plain encode занимал около 0.44 s. Внутри него native rANS
занимал около 0.16 s CPU work, остальное — bitmap, zlib, nonzero selection и связанные
операции. После переноса hyper job и параллельного запуска четыре spatial jobs часть этой
работы перекрывается GPU или другими CPU jobs.

CPU work нескольких parallel jobs нельзя складывать и называть wall-clock. Тем не менее
bitmap/zlib и необходимость вычислять plain candidate для непустых streams остаются
следующим резервом. Нужны измерения wall-clock critical path, а не оптимизация по сумме
времени потоков.

Decode-side entropy намного меньше: около 0.05 s для sparse decode на 12-Мп профиле.
Поэтому GPU rANS или сложный parallel rANS decode не являются первым приоритетом.

### 6.4. Оставшиеся CPU↔GPU transfers и синхронизации

Раньше было восемь отдельных spatial transfers и переносы checkerboard masks. Теперь masks
не переносятся, используются pinned buffers и одна синхронизация. Это дало выигрыш, но
следующие переносы все еще необходимы текущему формату:

- quantized symbols и sigma на CPU для native rANS encode;
- decoded residuals обратно на GPU между четырьмя causal phases при decode;
- sigma на CPU для выбора rANS tables при decode.

Полное удаление этих переходов потребует GPU entropy implementation или другого entropy/
context дизайна. Из-за малой доли decode rANS сначала нужно доказать, что новый GPU path не
проиграет нативному C на launch/sync overhead.

### 6.5. Scale/context decoders и predictor

На 12-Мп fast profile:

- scale decoder: около 0.02 s;
- context decoder: около 0.02 s;
- четыре predictor прохода вместе: около 0.035 s.

Это уже не главные bottleneck. Их оптимизация не даст шестикратного приближения к Apple.

### 6.6. Tile assembly

Для whole-image 12-Мп лишних tiles нет. Для 201-Мп сборка остается частью runtime, но
основное время — повторные analysis/synthesis/entropy для tiles. Увеличение числа CUDA
streams или уменьшение tile size не гарантирует параллельного выигрыша и увеличивает bpp
из-за повторного контекста и halo.

## 7. Что уже пробовали и почему не приняли

### Успешные изменения, уже входящие в текущий baseline

- GPU FP32 scale decoder вместо CPU FP64 на ARM/CUDA: устраняет примерно 0.5 s на 12-Мп.
- Fast mode для всех размеров: 201-Мп decode ускорился с 21.08 до 14.169 s.
- Contiguous synthesis: быстрее channels-last на GB10.
- Strided spatial slices: тот же bitstream без mask transfers и полноразмерных `where`.
- Parallel independent entropy encode jobs: перекрытие CPU и GPU работы.
- Empty sparse streams без полного plain rANS: удаляет три дорогих прохода по нулям в
  типичном ultra-low случае.
- GPU uint8 quantization/output: меньше финальный transfer и CPU postprocessing.
- Pinned spatial copies с одной синхронизацией.
- Эти изменения сохранили Apple-scene streams побайтово и прошли 201-Мп roundtrip.

### Отклоненные или пока неинтегрированные идеи

1. **Dense convolution вместо grouped convolution с нулевыми межгрупповыми весами.**
   Математически эквивалентно, но дополнительные FLOPs почти полностью съедают выигрыш.
   В одном варианте наблюдалось лишь около 0.04 s; численность и bitstream менялись.

2. **HaarDown fusion в одну свёртку 2×2.**
   Корректная математическая факторизация реализована как эксперимент, но стабильного
   ускорения не дала. Менялась численность и несколько байт потока. Откачено.

3. **HaarUp через ConvTranspose2d или fused 1×1 + pixel shuffle.**
   Не дало устойчивого decode-ускорения, меняло reconstruction numerics. Fusion scale/context
   HaarUp нельзя применять к старому потоку: изменение sigma вызывает entropy desync.

4. **cuDNN benchmark/autotune.**
   Первый encode занял около 42.5 s, первый decode около 14.9 s из-за поиска алгоритмов.
   Теплого выигрыша не было; поток и reconstruction numerics менялись. Не принято.

5. **`torch.compile` analysis/synthesis/scale.**
   В отдельных прогонах encode улучшался примерно до 1.28 s, synthesis иногда давал небольшой
   выигрыш. Но первая компиляция занимала примерно 43–72 s, появлялись shape/expert
   specialization issues, numerics и bitstream менялись. Для интерактивного codec это не было
   принято. AOT с заранее подготовленными kernels остается отдельной возможной идеей.

6. **`torch.compile` predictor.**
   Ранее менял bitstream и не давал приемлемого результата. Predictor занимает всего около
   0.035 s, поэтому возвращаться к нему нецелесообразно.

7. **Thread-per-stream async rANS во время predictor.**
   Ранее был примерно втрое медленнее. Текущий принятый вариант отличается: hyper job
   перекрывается GPU, а четыре spatial jobs запускаются после завершения GPU causal work.

8. **Принудительная нарезка 12-Мп на tiles и два CUDA streams.**
   Tiles 2048/1536/1024 не ускорили стабильно whole-image path. Bitrate вырос с 0.134012 до
   примерно 0.144–0.159 bpp из-за halo и повторного контекста. Не принято.

9. **Channels-last для synthesis.**
   Оказался медленнее contiguous на GB10; также дорогой несвязный HWC transfer увеличивал
   decode. Сейчас channels-last оставлен прежде всего для analysis.

10. **BF16 PICO.**
    Ранее резко ухудшил качество: PSNR падал примерно с 21.37 до 12.9–15.9 dB. Без QAT или
    другого компенсирующего обучения не использовать.

11. **Простая попытка распараллелить causal phases.**
    Несовместима с текущей entropy dependency: следующая phase predictor зависит от decoded
    предыдущих. Нужна новая модель/bitstream и переобучение, а не только другой scheduler.

12. **Дальнейший MoE/retrain как способ решить текущий runtime gap.**
    Только один expert активен, поэтому шесть experts не являются причиной шестикратного
    замедления. Сейчас важнее convolution runtime, а не добавление experts.

## 8. Наиболее перспективные направления для новых идей

1. Специализированные GB10 kernels для grouped 3×3 ConvScale, прежде всего `e12` и `e22`.
2. AOT/native pipeline с заранее выбранными algorithms без 40–70-секундной runtime компиляции.
3. Kernel fusion ConvScale + GELU + residual/Haar там, где это доказуемо сокращает memory traffic.
4. Integer/QAT tensor-core inference с matched-bpp RD validation; обычный BF16 уже отклонен.
5. Профилирование энергии/частот и warmup: короткие isolated runs на GB10 иногда заметно
   медленнее устойчивого теплого состояния, поэтому нужны медианы нескольких runs.
6. Более дешевый sparse bitmap/plain-candidate selection только после измерения фактического
   encode critical path.
7. Новая parallel-friendly context архитектура имеет смысл только как отдельный retrain
   experiment; текущий four-phase predictor сам по себе уже не главный bottleneck.

Для каждого кандидата обязательны: реальный OXP9 roundtrip, все шесть experts, проверка
entropy consistency, matched-bpp PSNR/MS-SSIM/LPIPS, 12-Мп encode/decode medians, 201-Мп
memory/tiling regression и проверка старых streams. Нельзя оценивать успех только по импорту,
компиляции или времени одного isolated CUDA call.
