<div align="center">

# VoxCPM2 Portable

**Portable Windows build of VoxCPM2 — multilingual TTS with Voice Design & Cloning.**

[![Stars](https://img.shields.io/github/stars/timoncool/VoxCPM2_portable?style=flat-square)](https://github.com/timoncool/VoxCPM2_portable/stargazers)
[![License](https://img.shields.io/github/license/timoncool/VoxCPM2_portable?style=flat-square)](LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/timoncool/VoxCPM2_portable?style=flat-square)](https://github.com/timoncool/VoxCPM2_portable/commits/main)
[![Downloads](https://img.shields.io/github/downloads/timoncool/VoxCPM2_portable/total?style=flat-square)](https://github.com/timoncool/VoxCPM2_portable/releases)

**[Русская версия](README_RU.md)**

</div>

Generate natural multilingual speech, design brand-new voices from text descriptions, or clone any voice from a short reference clip — **100% local**, no cloud, no API keys, no subscriptions. One-click install on Windows, runs on any NVIDIA GPU with 8+ GB VRAM.

Built on [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) by OpenBMB — a tokenizer-free 2B-parameter diffusion autoregressive TTS model trained on 2M+ hours of speech.

## Why VoxCPM2 Portable?

- **Free forever** — no API keys, no credits, no usage limits
- **Private** — your voice data never leaves your machine
- **Portable** — everything in one folder, copy to USB, delete = uninstall
- **One-click** — `install.bat` → `run.bat` → generate speech
- **30 languages** — Russian, English, Chinese, French, German, Japanese, Korean and more

## Features

### Text-to-Speech
- **30 languages** out of the box — Russian, English, Chinese (plus 9 dialects), Arabic, French, German, Hindi, Italian, Japanese, Korean, Portuguese, Spanish and many more
- **48 kHz studio-quality output** — AudioVAE V2 super-resolution from 16 kHz input
- **Natural prosody** — tokenizer-free diffusion autoregressive architecture
- **Language auto-detection** — no language tags needed, paste text in any language

### Voice Design
- **Create voices from text** — describe gender, age, tone, emotion, pace, accent
- **Zero-shot** — no reference audio required
- **Examples** — "A young woman with a soft voice" / "An elderly man with a deep baritone"

### Voice Cloning
- **Clone any voice** from 5-30 seconds of clean audio (up to 50 seconds)
- **Style control** — steer emotion/pace while preserving timbre: "slightly faster, cheerful tone"
- **Microphone recording** — record reference directly in the browser
- **Denoise with ZipEnhancer** — automatic cleanup of noisy references

### Ultimate Cloning
- **Maximum fidelity** — reference audio + exact transcript
- **Continuation mode** — model "continues" reference audio preserving every vocal nuance
- **Best for production** — audiobooks, voiceovers, character consistency

### Interface
- **Bilingual UI** — Russian + English, auto-detected from browser language
- **Dark theme** — forced dark mode for comfortable work
- **Seed + Lock** — reproducible generations
- **Advanced settings** — CFG Scale, Inference Steps, text normalization (wetext), denoise, retry
- **Auto-download** — model (~4-5 GB) downloads automatically on first run
- **Auto-port, auto-browser** — no manual port config, opens in browser automatically

### Out-of-the-box GPU Accelerators
All compatible accelerators installed automatically based on your GPU:

| GPU | Triton | Flash Attention 2 | xformers |
|-----|--------|-------------------|----------|
| GTX 10xx (Pascal) | ✅ | ❌ | ✅ |
| RTX 20xx (Turing) | ✅ | ❌ | ✅ |
| RTX 30xx (Ampere) | ✅ | ✅ | ✅ |
| RTX 40xx (Ada) | ✅ | ✅ | ✅ |
| RTX 50xx (Blackwell) | ✅ | ✅ | ✅ |

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 8 GB | 12+ GB |
| RAM | 16 GB | 32 GB |
| Disk | 15 GB | 30 GB |
| OS | Windows 10/11 | Windows 11 |
| GPU | RTX 2080 / RTX 3060 | RTX 4070+ |

CPU-only mode works but is experimentally slow (minutes per short phrase).

## Quick Start

### 1. Clone

```bash
git clone https://github.com/timoncool/VoxCPM2_portable.git
cd VoxCPM2_portable
```

### 2. Install

```
install.bat
```

Select your GPU type (6 options). Installs portable Python 3.12, PyTorch 2.7, voxcpm, plus all compatible GPU accelerators — nothing system-wide.

### 3. Run

```
run.bat
```

Browser opens automatically. On first run the model downloads to `models/` (~4-5 GB).

## Launchers

| Script | Description |
|--------|-------------|
| `install.bat` | One-click installer — Python + PyTorch + voxcpm + accelerators + FFmpeg |
| `run.bat` | Launch the Gradio UI with full environment isolation |
| `update.bat` | Update portable wrapper and voxcpm package |

## Architecture

```
VoxCPM2_portable/
├── app.py              # Gradio UI (RU/EN, 4 tabs, dark theme)
├── install.bat         # GPU selector + installer
├── run.bat             # Launcher with env isolation
├── update.bat          # Updater
├── requirements.txt    # voxcpm + gradio + soundfile + numpy
├── python/             # Portable Python 3.12 (created by install.bat)
├── models/             # HuggingFace cache (VoxCPM2 weights ~4-5 GB)
├── ffmpeg/             # Portable FFmpeg (for wide audio format support)
├── output/             # Generated WAV files with timestamps
├── cache/              # General cache
└── temp/               # Temporary files (Gradio)
```

## How to Use

### Text-to-Speech
Just enter text in any supported language — the model auto-detects the language.

### Voice Design
Describe the voice you want in "Voice description" (e.g. "A young woman, gentle and sweet voice"), then enter what it should say. Results vary between runs — try 1-3 times and lock the seed when you find the voice you want.

### Voice Cloning
Upload a 5-30 second clean recording. Optionally set a style: "slightly faster, cheerful tone" / "slow and dramatic" / "whispering, intimate".

### Ultimate Cloning
Best quality mode. Provide reference audio **and** its exact transcript. Use the **same** file as both reference and prompt for maximum similarity.

### Advanced Settings
- **CFG Scale** (0.5–5.0) — how closely to follow the prompt/reference. 2.0 is recommended.
- **Inference Steps** (5–30) — diffusion steps. 10 balances quality and speed.
- **Seed + Lock Seed** — reproducible generations
- **Text normalization** — auto-processes numbers, dates, abbreviations via wetext
- **Denoise reference** — ZipEnhancer cleanup before cloning

## Updating

```
update.bat
```

Pulls latest wrapper code, upgrades `voxcpm` package.

## Links

- [OpenBMB / VoxCPM](https://github.com/OpenBMB/VoxCPM) — original project
- [VoxCPM2 model card](https://huggingface.co/openbmb/VoxCPM2) — model weights
- [Demo page with audio samples](https://openbmb.github.io/voxcpm2-demopage/)
- [Official documentation](https://voxcpm.readthedocs.io/)

## Other Portable Neural Networks

| Project | Description |
|---------|-------------|
| [ACE-Step Studio](https://github.com/timoncool/ACE-Step-Studio) | Local AI music generation studio |
| [Foundation Music Lab](https://github.com/timoncool/Foundation-Music-Lab) | Music generation + timeline editor |
| [VibeVoice ASR](https://github.com/timoncool/VibeVoice_ASR_portable_ru) | Speech recognition (ASR) |
| [LavaSR](https://github.com/timoncool/LavaSR_portable_ru) | Audio quality enhancement |
| [Qwen3-TTS](https://github.com/timoncool/Qwen3-TTS_portable_rus) | Text-to-speech by Qwen |
| [SuperCaption Qwen3-VL](https://github.com/timoncool/SuperCaption_Qwen3-VL) | Image captioning |
| [VideoSOS](https://github.com/timoncool/videosos) | AI video production |
| [RC Stable Audio Tools](https://github.com/timoncool/RC-stable-audio-tools-portable) | Music and audio generation |

## Authors

- **Nerual Dreming** — [Telegram](https://t.me/nerual_dreming) | [neuro-cartel.com](https://neuro-cartel.com) | [ArtGeneration.me](https://artgeneration.me)
- **Neiro-Soft** — [Telegram](https://t.me/neuroport) | portable neural network builds

## Acknowledgments

- **[OpenBMB / VoxCPM team](https://github.com/OpenBMB/VoxCPM)** — open source VoxCPM2 model
- **[AIQuest Academy](https://github.com/TeamAIQ/Colab-notebooks)** — Colab notebook reference for the 4-tab UI structure
- **[mjun0812](https://github.com/mjun0812/flash-attention-prebuild-wheels)** — Windows wheels for Flash Attention 2
- **[woct0rdho / triton-windows](https://pypi.org/project/triton-windows/)** — Triton port for Windows
- **[Gradio](https://gradio.app/)** — ML model UI framework
- **[FFmpeg](https://ffmpeg.org/)** — audio processing

## Support This Project

I build software and do research in AI. Most of what I create is free and open source.

**[All donation methods](https://dalink.to/nerual_dreming)** | **[boosty.to/neuro_art](https://boosty.to/neuro_art)**

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
