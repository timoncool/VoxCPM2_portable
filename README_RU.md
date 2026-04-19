<div align="center">

# VoxCPM2 Portable

<img src="docs/hero.png" alt="VoxCPM2 Portable" width="600"/>

**Портативная Windows-сборка VoxCPM2 — мультиязычный TTS с Voice Design, клонированием голоса и end-to-end тренировкой LoRA (видео/аудио → датасет → обучение).**

[![Stars](https://img.shields.io/github/stars/timoncool/VoxCPM2_portable?style=flat-square)](https://github.com/timoncool/VoxCPM2_portable/stargazers)
[![License](https://img.shields.io/github/license/timoncool/VoxCPM2_portable?style=flat-square)](LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/timoncool/VoxCPM2_portable?style=flat-square)](https://github.com/timoncool/VoxCPM2_portable/commits/main)
[![Downloads](https://img.shields.io/github/downloads/timoncool/VoxCPM2_portable/total?style=flat-square)](https://github.com/timoncool/VoxCPM2_portable/releases)

**[English version](README.md)**

</div>

Синтезируй натуральную речь на 30 языках, создавай уникальные голоса из текстового описания, клонируй любой голос по короткому референсу и **тренируй свою LoRA прямо из видео или аудио-файла** — бросил 8-минутный подкаст в приложение, и оно само нарезает на клипы, транскрибирует, подбирает оптимальные параметры и запускает тренировку. **100% локально**, без облака, без API-ключей. Установка в один клик на Windows, работает на любой NVIDIA GPU с 8+ GB VRAM.

Построен на [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) от OpenBMB — токенайзер-свободной 2B модели TTS (diffusion autoregressive), обученной на 2M+ часов речи.

## Почему VoxCPM2 Portable?

- **Бесплатно навсегда** — без API-ключей, кредитов и лимитов
- **Приватно** — твои голосовые данные никуда не уходят
- **Портативно** — всё в одной папке, скопируй на флешку, удалил = деинсталлировал
- **Один клик** — `install.bat` → `run.bat` → синтезируй
- **30 языков** — русский, английский, китайский, французский, немецкий, японский, корейский и другие
- **Авто-датасет из видео/аудио** — загружаешь длинный файл, приложение делает ffmpeg → ASR → VAD → нарезку по предложениям → подбор параметров → обучение, всё само

## Возможности

### Синтез речи (Text-to-Speech)
- **30 языков** — RU/EN/ZH (+9 китайских диалектов)/AR/FR/DE/HI/IT/JA/KO/PT/ES и другие — автоопределение языка
- **48 kHz студийное качество** через AudioVAE V2 super-resolution (16→48 kHz)
- **Естественная просодия** — tokenizer-free diffusion autoregressive архитектура
- Форматы вывода: **MP3** (по умолчанию), WAV, FLAC, OGG
- **Live-стриминг воспроизведения** — аудио начинает играть во время генерации (8 сек prebuffer + progressive чанки по 2 сек), не надо ждать окончания синтеза

### Дизайн голоса (Voice Design)
- Создавай голоса из **текстового описания** — пол, возраст, тон, эмоция, темп, акцент
- Zero-shot — без референс-аудио
- 6 готовых примеров (EN+RU) — клик и заполняется

### Клонирование голоса (с опциональным Ultimate-режимом)
- Клонируй любой голос по **5-50 секундам** референс-аудио
- **Пак голосов** из коробки (~100 голосов: RU/EN/FR/DE/JP/KR/AR)
- **743 дополнительных русских голоса** по запросу из `Slait/russia_voices`
- Контроль стиля: `чуть быстрее, бодрым тоном` / `шёпотом, интимно` / `медленно и драматично`
- **Ultimate-режим** — заполни транскрипт → модель использует `prompt_wav_path + prompt_text + reference_wav_path` для максимальной верности
- Опциональное **ZipEnhancer шумоподавление** для шумных референсов

### LoRA Fine-Tuning — Авто-пайплайн 🎬
Тренируй LoRA на целом видео или подкасте в один клик:
1. **Дропаешь видео/аудио** (mp4/mkv/webm/mov/mp3/wav/flac/…)
2. ffmpeg извлекает аудио → 16 kHz mono WAV
3. **Parakeet TDT 0.6B v3 INT8 ONNX** (NVIDIA NeMo, ~670 MB, 25 европейских языков включая русский) + **Silero VAD** транскрибируют с word-level таймстампами, держат любую длину
4. Слова группируются в **целые предложения** по пунктуации, короткие склеиваются, длинные режутся по запятым/паузам — **никогда не посреди слова**
5. Клипы сохраняются в `train_data/<имя>/audio/clip_NNNN.wav` + `transcripts.txt`
6. **Авто-тюнинг** подбирает `r / alpha / steps / lr` по минутам чистой речи (официальная таблица OpenBMB)
7. Обучение стартует в том же клике (если галочка стоит)

Типичный результат для 8 мин речи: ~69 клипов, 86 % извлечения, ~14 мин тренировки на RTX 4090.

### LoRA Fine-Tuning — Ручной режим
Для заранее подготовленных датасетов:
- Загрузи 5-50 WAV/MP3 (3-15 сек каждое) + транскрипты в формате `имя_файла.wav|текст`
- Дефолты соответствуют **официальному YAML OpenBMB** (`voxcpm_finetune_lora.yaml`): `r=32, alpha=32, steps=1000, lr=1e-4`
- Live-лог тренировки, автоочистка старых чекпоинтов при повторном запуске
- **Hot-swap** LoRA во всех режимах без перезапуска

### Все параметры модели выведены в UI
CFG Scale · Inference Steps · Min/Max длина · Retry при плохой генерации · Макс. повторов · Порог плохой генерации · Нормализация текста (wetext) · Денойзинг референса · Streaming (live-прогресс) · Seed + Lock

### Интерфейс
- **i18n RU/EN** — кнопки RU/EN в шапке для мгновенного переключения
- **Тёмная тема** с gradient header
- **FFmpeg в комплекте** (для кодирования MP3/OGG)
- **Автозагрузка** — VoxCPM2 (~4-5 GB) + пак голосов + ASR-модель (~670 MB, лениво) при первом использовании
- **Auto-port, auto-browser** — открывается на `localhost` автоматически

### GPU-ускорители (из коробки)
| GPU | Flash Attention 2 | SDPA flash | bfloat16 | AMP (тренировка) | ONNX CUDA (ASR) |
|-----|:---:|:---:|:---:|:---:|:---:|
| RTX 30xx / 40xx / 50xx | ✅ | ✅ | ✅ | ✅ | ✅ |
| GTX 10xx / RTX 20xx | ❌ | ✅ | ✅ | ✅ | ✅ |

## Системные требования

| Компонент | Минимум | Рекомендуется |
|-----------|---------|---------------|
| GPU VRAM | 8 GB | 12+ GB |
| RAM | 16 GB | 32 GB |
| Диск | 15 GB | 30 GB (с пакетами голосов и LoRA) |
| ОС | Windows 10/11 | Windows 11 |
| GPU | RTX 2080 / RTX 3060 | RTX 4070+ |

CPU-режим поддерживается, но **очень медленный** (минуты на фразу).

## Быстрый старт

### 1. Клонируй
```bash
git clone https://github.com/timoncool/VoxCPM2_portable.git
cd VoxCPM2_portable
```

### 2. Установи
```
install.bat
```
Выбери тип GPU (6 вариантов). Устанавливает портативный Python 3.12 + PyTorch 2.7.1 + voxcpm + Flash Attention 2 + FFmpeg + onnx-asr + дефолтный пак голосов. Ничего в системе.

### 3. Запусти
```
run.bat
```
Браузер откроется сам. Модель скачается при первом запуске (~4-5 GB в `models/`). Parakeet ASR (~670 MB) тянется только при первом клике на **Авто-подготовку**.

## Батники

| Скрипт | Описание |
|--------|----------|
| `install.bat` | Установщик в один клик — Python + PyTorch + voxcpm + ускорители + FFmpeg + onnx-asr + пак голосов |
| `run.bat` | Запуск Gradio UI с полной изоляцией окружения |
| `update.bat` | Обновление портативки + voxcpm package |

## LoRA тренировка — полный гайд

### Вариант A — Авто из видео/аудио (рекомендуется)

1. Открой таб **LoRA**
2. Введи **Имя датасета / LoRA** (используется и для `train_data/<имя>/`, и для `lora/<имя>/`)
3. Перейди на под-таб **🎬 Авто**
4. Брось видео или аудио-файл (mp4, mkv, webm, mp3, wav, flac, m4a, ogg, opus, …)
5. Оставь **🤖 Авто-тюнинг** включённым (подберёт r / α / steps / lr по минутам чистой речи)
6. Поставь галочку **Запустить обучение после подготовки**, если хочешь чтобы ещё и обучение стартануло без второго клика
7. Жми **🎬 Авто-подготовка**

Приложение стримит лог по каждому этапу: ffmpeg-извлечение → распознавание Parakeet → сегментация → сохранение клипов → тренировка. Финальный чекпоинт ложится в `lora/<имя>/step_XXXX/` и сразу появляется в dropdown'е LoRA на всех табах.

### Вариант B — Ручной режим

1. Подготовь клипы (5-50 WAV/MP3, 3-15 сек каждое) + транскрипты в формате `имя_файла.wav|текст`, по строке
2. Переключись на под-таб **🎓 Ручное обучение** в табе LoRA
3. Загрузи файлы и вставь транскрипты
4. При желании подкрути слайдеры r / α / steps / lr (дефолты — официальные значения OpenBMB)
5. Жми **🎓 Начать обучение**

### Рекомендуемые настройки — авто-тюнинг по минутам чистой речи

Приложение считает `steps = target_epochs × n_clips / effective_batch`, округляет до 50, и выбирает `grad_accum` адаптивно (4 для <200 клипов, 8 для <500, 16 для 500+). Таблица target-эпох:

| Чистой речи | target эпох | r / α | lr |
|---|---|---|---|
| < 2 мин | 25 (⚠ ниже минимума OpenBMB) | 32 / 32 | 1e-4 |
| 2-5 мин | 20 | 32 / 32 | 1e-4 |
| **5-10 мин** (сладкая зона) | **15** | **32 / 32** | **1e-4** |
| 10-20 мин | 12 | 32 / 32 | 1e-4 |
| 20-60 мин | 8 | 32 / 32 | 1e-4 |
| 60-120 мин | 5 | 64 / 64 | 5e-5 |
| 120+ мин | 3 | 64 / 64 | 5e-5 |

Для 69 клипов / 7.6 мин (типичный 8-минутный подкаст) → 250 шагов → **≈3-4 мин тренировки на RTX 4090**.

### Активация обученной LoRA

- В любом табе (TTS / Voice Design / Voice Cloning) раскрой **Расширенные настройки**
- Выбери свою LoRA в dropdown'е **🧬 LoRA**
- Первая активация занимает ~30-60 сек (модель перезагружается со структурой LoRA r/α)
- Последующие переключения — мгновенный hot-swap

## Структура

```
VoxCPM2_portable/
├── app.py              # Gradio UI (4 таба: TTS / Voice Design / Клонирование / LoRA)
├── install.bat         # Выбор GPU + установщик
├── run.bat             # Запуск с изоляцией окружения
├── update.bat          # Обновление
├── requirements.txt    # Python-зависимости
├── training/
│   ├── scripts/        # Официальные OpenBMB train & inference скрипты (встроены)
│   └── conf/           # Шаблоны YAML-конфигов
├── python/             # Портативный Python 3.12 (создаётся install.bat)
├── models/             # HuggingFace кэш (VoxCPM2 ~4-5 GB, Parakeet ~670 MB, Silero VAD, …)
├── voices/             # Пак голосов (дефолтные ~100 + пользовательские)
├── lora/               # Обученные LoRA чекпоинты (lora/<имя>/step_XXXX/)
├── train_data/         # Пользовательские датасеты для LoRA (аудио + транскрипты)
├── ffmpeg/             # Портативный FFmpeg (для кодирования MP3/OGG)
├── output/             # Сгенерированные аудио с таймстампами
├── cache/ / temp/      # Общий кэш / tempdir
```

## Обновление

```
update.bat
```

## Ссылки

- [OpenBMB / VoxCPM](https://github.com/OpenBMB/VoxCPM) — оригинальный проект
- [VoxCPM2 карточка модели](https://huggingface.co/openbmb/VoxCPM2) — веса
- [Демо-страница с примерами](https://openbmb.github.io/voxcpm2-demopage/)
- [Гайд по Fine-tuning](https://voxcpm.readthedocs.io/en/latest/finetuning/finetune.html)
- [Parakeet TDT 0.6B v3 (мультиязычный ASR)](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
- [onnx-asr](https://github.com/istupakov/onnx-asr) — Python-обёртка использующаяся для Parakeet

## Другие портативные нейросети

| Проект | Описание |
|--------|----------|
| [ACE-Step Studio](https://github.com/timoncool/ACE-Step-Studio) | Локальная AI-студия для генерации музыки |
| [Foundation Music Lab](https://github.com/timoncool/Foundation-Music-Lab) | Генерация музыки + редактор таймлайна |
| [VibeVoice ASR](https://github.com/timoncool/VibeVoice_ASR_portable_ru) | Распознавание речи (ASR) |
| [LavaSR](https://github.com/timoncool/LavaSR_portable_ru) | Улучшение качества звука |
| [Qwen3-TTS](https://github.com/timoncool/Qwen3-TTS_portable_rus) | Синтез речи от Qwen |
| [SuperCaption Qwen3-VL](https://github.com/timoncool/SuperCaption_Qwen3-VL) | Описание изображений |
| [VideoSOS](https://github.com/timoncool/videosos) | AI-видеопродакшн |
| [RC Stable Audio Tools](https://github.com/timoncool/RC-stable-audio-tools-portable) | Генерация музыки и звуков |

## Авторы

Собрал **[Nerual Dreming](https://t.me/nerual_dreming)** — основатель **[ArtGeneration.me](https://artgeneration.me)**, техноблогер и нейро-евангелист. Канал **[Нейро-Софт](https://t.me/neuroport)** — репаки и портативки полезных нейросетей.

## Благодарности

- **[Команда OpenBMB / VoxCPM](https://github.com/OpenBMB/VoxCPM)** — открытая модель VoxCPM2
- **[NVIDIA NeMo / Parakeet TDT](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)** — мультиязычный ASR
- **[istupakov/onnx-asr](https://github.com/istupakov/onnx-asr)** — ONNX-обёртка для Parakeet + Silero VAD
- **[Slait/russia_voices](https://huggingface.co/datasets/Slait/russia_voices)** — 743 русских голоса
- **[lldacing/flash-attention-windows-wheel](https://huggingface.co/lldacing/flash-attention-windows-wheel)** — Windows wheels для Flash Attention 2
- **[Gradio](https://gradio.app/)** — UI фреймворк
- **[FFmpeg](https://ffmpeg.org/)** — кодирование аудио

## Поддержать опенсорс

Привет! Я Илья ([Nerual Dreming](https://t.me/nerual_dreming)), я создаю AI-инструменты, которые работают локально — бесплатно, без облака, без подписок. Ваш донат позволяет фокусироваться на исследовании и создании новых открытых проектов, а не на выживании. Спасибо!

**[Карта / PayPal / Apple Pay](https://dalink.to/nerual_dreming)** | **[Ежемесячная подписка на Boosty](https://boosty.to/neuro_art)**

- **BTC:** `1E7dHL22RpyhJGVpcvKdbyZgksSYkYeEBC`
- **ETH (ERC20):** `0xb5db65adf478983186d4897ba92fe2c25c594a0c`
- **USDT (TRC20):** `TQST9Lp2TjK6FiVkn4fwfGUee7NmkxEE7C`

---

## Star History

<a href="https://www.star-history.com/?repos=timoncool%2FVoxCPM2_portable&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=timoncool/VoxCPM2_portable&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=timoncool/VoxCPM2_portable&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=timoncool/VoxCPM2_portable&type=date&legend=top-left" />
 </picture>
</a>
</content>
</invoke>