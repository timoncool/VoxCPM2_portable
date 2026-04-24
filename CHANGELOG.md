# Changelog

## 2026-04-25

### Fixed
- **Tab hangs in Gradio UI** — outer tabs (TTS / Design / Clone / LoRA) now use `render_children=False`, so Gradio 6.10 only mounts the active tab's components instead of rendering all of them eagerly. Eliminates stutters and freezes when switching tabs. Inside LoRA, the nested `gr.Tabs()` (Auto / Manual modes) was replaced with two `gr.Accordion` blocks (Auto open by default, Manual collapsed) — nested Tabs is a known source of UI freezes in Gradio 6. Credit: fix proposed by Aaron Lee.

## 2026-04-23

### Added
- **Pinokio launcher** — one-click cross-platform install via [Pinokio](https://pinokio.computer/). Works on Windows / Linux (x64 & aarch64) / macOS, NVIDIA / AMD / CPU. No manual `install.bat` required — Pinokio manages CUDA, Python 3.12, PyTorch, Flash-Attn wheels, triton, xformers, and FFmpeg automatically. Launcher repo: [timoncool/VoxCPM2_portable-pinokio](https://github.com/timoncool/VoxCPM2_portable-pinokio).
- **`NO_AUTO_BROWSER` env var** — when set to `true`/`1`/`yes`, `app.py` skips Gradio's `inbrowser=True` auto-opening of the system browser. Prevents a duplicate Chrome tab when launched from Pinokio (or any other launcher that opens its own embedded UI). Default behavior unchanged when the env var is unset.
