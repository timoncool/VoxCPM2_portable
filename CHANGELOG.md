# Changelog

## 2026-04-23

### Added
- **Pinokio launcher** — one-click cross-platform install via [Pinokio](https://pinokio.computer/). Works on Windows / Linux (x64 & aarch64) / macOS, NVIDIA / AMD / CPU. No manual `install.bat` required — Pinokio manages CUDA, Python 3.12, PyTorch, Flash-Attn wheels, triton, xformers, and FFmpeg automatically. Launcher repo: [timoncool/VoxCPM2_portable-pinokio](https://github.com/timoncool/VoxCPM2_portable-pinokio).
- **`NO_AUTO_BROWSER` env var** — when set to `true`/`1`/`yes`, `app.py` skips Gradio's `inbrowser=True` auto-opening of the system browser. Prevents a duplicate Chrome tab when launched from Pinokio (or any other launcher that opens its own embedded UI). Default behavior unchanged when the env var is unset.
