"""VoxCPM2 Portable — Multilingual TTS (RU/EN).

Портативная русскоязычная сборка VoxCPM2 от Nerual Dreming + Нейро-Софт.
Поддерживает 30 языков, включая русский. Режимы: TTS / Voice Design / Voice Cloning / Ultimate Cloning.
"""

# === КРИТИЧЕСКИЙ ПАТЧ: отключение torch._dynamo ДО импорта voxcpm ===
import os
import sys
import asyncio
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# === Windows retry_open patch для anyio/aiofiles (PermissionError от антивируса) ===
if sys.platform == "win32":
    try:
        import anyio
        import anyio._core._fileio
        _original_anyio_open = anyio._core._fileio.open_file

        async def _retry_anyio_open(file, *args, **kwargs):
            max_retries = 20
            delay = 0.2
            for attempt in range(max_retries):
                try:
                    return await _original_anyio_open(file, *args, **kwargs)
                except PermissionError:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(delay)
                        delay *= 1.2
                    else:
                        raise

        anyio._core._fileio.open_file = _retry_anyio_open
        anyio.open_file = _retry_anyio_open
        try:
            import starlette.responses
            starlette.responses.anyio.open_file = _retry_anyio_open
        except Exception:
            pass
    except ImportError:
        pass

    try:
        import aiofiles
        import aiofiles.threadpool
        _original_aiofiles_open = aiofiles.threadpool._open

        async def _retry_aiofiles_open(*args, **kwargs):
            max_retries = 20
            delay = 0.2
            for attempt in range(max_retries):
                try:
                    return await _original_aiofiles_open(*args, **kwargs)
                except PermissionError:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(delay)
                        delay *= 1.2
                    else:
                        raise

        aiofiles.threadpool._open = _retry_aiofiles_open
    except ImportError:
        pass

import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True

# === Стандартные импорты ===
import random
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

import gradio as gr
import numpy as np
import soundfile as sf

from voxcpm import VoxCPM

# === Конфигурация ===
SCRIPT_DIR = Path(__file__).parent.absolute()
OUTPUT_DIR = SCRIPT_DIR / "output"
VOICES_DIR = SCRIPT_DIR / "voices"
LORA_DIR = SCRIPT_DIR / "lora"
TRAIN_DATA_DIR = SCRIPT_DIR / "train_data"
TRAINING_DIR = SCRIPT_DIR / "training"  # bundled train scripts (in repo)
OUTPUT_DIR.mkdir(exist_ok=True)
VOICES_DIR.mkdir(exist_ok=True)
LORA_DIR.mkdir(exist_ok=True)
TRAIN_DATA_DIR.mkdir(exist_ok=True)

MODEL_REF = "openbmb/VoxCPM2"

# === Voice pack (русские голоса с HuggingFace) ===
CLOUD_VOICES_REPO = "Slait/russia_voices"
CLOUD_VOICES_BASE_URL = "https://huggingface.co/datasets/Slait/russia_voices/resolve/main"
_CLOUD_VOICES_CACHE: list[str] = []


def scan_local_voices() -> list[str]:
    """Список локальных голосов в voices/ (.mp3/.wav/.flac)."""
    exts = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}
    names = set()
    for p in VOICES_DIR.iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            names.add(p.stem)
    return sorted(names)


def voice_audio_path(name: str) -> Optional[str]:
    """Путь к аудиофайлу голоса в voices/ (любое расширение из поддерживаемых)."""
    for ext in (".mp3", ".wav", ".flac", ".m4a", ".ogg"):
        p = VOICES_DIR / f"{name}{ext}"
        if p.exists():
            return str(p)
    return None


def get_first_ru_voice() -> Optional[str]:
    """Первый RU_ голос из локального пака (дефолт)."""
    for name in scan_local_voices():
        if name.upper().startswith("RU_"):
            return name
    return None


def voice_transcript(name: str) -> str:
    """Транскрипт голоса из voices/{name}.txt или .lab, если есть."""
    for ext in (".txt", ".lab"):
        p = VOICES_DIR / f"{name}{ext}"
        if p.exists():
            try:
                return p.read_text(encoding="utf-8").strip()
            except Exception:
                try:
                    return p.read_text(encoding="cp1251").strip()
                except Exception:
                    return ""
    return ""


def fetch_cloud_voices_list() -> list[str]:
    """Получить список mp3-голосов из HF dataset."""
    global _CLOUD_VOICES_CACHE
    try:
        from huggingface_hub import list_repo_files
        files = list(list_repo_files(CLOUD_VOICES_REPO, repo_type="dataset"))
        voices = sorted(f[:-4] for f in files if f.endswith(".mp3"))
        _CLOUD_VOICES_CACHE = voices
        return voices
    except Exception as exc:
        print(f"[voices] fetch list error: {exc}")
        return []


def download_cloud_voice(name: str) -> bool:
    """Скачать один голос (mp3+txt) в voices/."""
    import requests
    try:
        mp3_url = f"{CLOUD_VOICES_BASE_URL}/{name}.mp3?download=true"
        r = requests.get(mp3_url, timeout=60, stream=True)
        r.raise_for_status()
        with open(VOICES_DIR / f"{name}.mp3", "wb") as f:
            for chunk in r.iter_content(8192):
                if chunk:
                    f.write(chunk)
        # txt необязателен
        try:
            txt_url = f"{CLOUD_VOICES_BASE_URL}/{name}.txt?download=true"
            r2 = requests.get(txt_url, timeout=30)
            if r2.status_code == 200:
                (VOICES_DIR / f"{name}.txt").write_text(r2.text, encoding="utf-8")
        except Exception:
            pass
        return True
    except Exception as exc:
        print(f"[voices] download '{name}' error: {exc}")
        return False


def load_cloud_list():
    """Загрузить список голосов с HF (как в Qwen3-TTS). Возвращает (status, checkbox_update)."""
    voices = fetch_cloud_voices_list()
    if voices:
        return (
            f"Найдено {len(voices)} голосов. Репозиторий: {CLOUD_VOICES_REPO}",
            gr.update(choices=voices, value=[]),
        )
    return (
        "Не удалось загрузить список голосов.",
        gr.update(choices=[], value=[]),
    )


# ====================================================================
# === LoRA: fine-tuning + hot-swap ===
# ====================================================================

_ACTIVE_LORA: Optional[str] = None  # имя загруженной LoRA или None


def scan_local_loras() -> list[str]:
    """Список локальных LoRA чекпоинтов в lora/ (по наличию lora_config.json)."""
    result = []
    for p in LORA_DIR.iterdir():
        if p.is_dir() and (p / "lora_config.json").exists():
            result.append(p.name)
    # также checkpoint-папки внутри step_XXXX
    for p in LORA_DIR.iterdir():
        if p.is_dir():
            for sub in p.iterdir():
                if sub.is_dir() and (sub / "lora_config.json").exists():
                    result.append(f"{p.name}/{sub.name}")
    return sorted(set(result))


def lora_attach(name: str) -> str:
    """Подключить LoRA.
    Если модель ещё не грузилась с LoRA — полная reinit с lora_weights_path.
    Иначе hot-swap через unload + load_lora."""
    global _ACTIVE_LORA, _model
    if not name or name == "-- Без LoRA --":
        return lora_detach()
    path = LORA_DIR / name
    if not path.exists() or not (path / "lora_config.json").exists():
        return f"❌ LoRA '{name}' не найдена (нет lora_config.json)"
    try:
        # Проверяем — есть ли у модели LoRA-структура
        has_lora_structure = (
            _model is not None and getattr(_model, "lora_enabled", False) is not False
            and hasattr(_model, "load_lora")
        )
        # Если модель загружена но без LoRA-структуры — полный reload
        if _model is None or _ACTIVE_LORA is None:
            print(f"[lora] reloading model with LoRA structure: {path}")
            model = get_model(lora_weights_path=str(path), force_reload=(_model is not None))
        else:
            model = _model
            try:
                model.unload_lora()
            except Exception:
                pass
            model.load_lora(str(path))
        try:
            model.set_lora_enabled(True)
        except Exception:
            pass
        _ACTIVE_LORA = name
        return f"✅ LoRA '{name}' активна"
    except Exception as exc:
        traceback.print_exc()
        return f"❌ Ошибка загрузки LoRA: {exc}"


def lora_detach() -> str:
    """Отключить активную LoRA (set_lora_enabled(False), модель остаётся в памяти)."""
    global _ACTIVE_LORA
    try:
        if _ACTIVE_LORA and _model is not None:
            try:
                _model.set_lora_enabled(False)
            except Exception as e:
                print(f"[lora] detach warning: {e}")
        _ACTIVE_LORA = None
        return "LoRA отключена"
    except Exception as exc:
        return f"Ошибка отключения: {exc}"


def lora_active_status() -> str:
    return f"Активна: {_ACTIVE_LORA}" if _ACTIVE_LORA else "LoRA не активна"


def get_training_script() -> Optional[Path]:
    """Возвращает путь к bundled train script."""
    script = TRAINING_DIR / "scripts" / "train_voxcpm_finetune.py"
    return script if script.exists() else None


def prepare_train_data(name: str, files: list, transcripts_text: str) -> tuple[Path, int]:
    """
    Подготовить датасет для обучения:
    - files: список временных путей к wav/mp3/flac из gr.File
    - transcripts_text: построчно — имя_файла|транскрипт
    Возвращает (путь_к_manifest.jsonl, количество_сэмплов)
    """
    import json, shutil
    ds_dir = TRAIN_DATA_DIR / name
    ds_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = ds_dir / "audio"
    audio_dir.mkdir(exist_ok=True)

    # Map transcript_text "filename.wav|текст"
    tr_map = {}
    for line in (transcripts_text or "").splitlines():
        line = line.strip()
        if not line or "|" not in line:
            continue
        fn, tx = line.split("|", 1)
        tr_map[fn.strip()] = tx.strip()

    manifest = []
    for src in files or []:
        src = Path(src)
        if not src.exists():
            continue
        dst = audio_dir / src.name
        if not dst.exists():
            shutil.copy2(str(src), str(dst))
        tx = tr_map.get(src.name) or tr_map.get(src.stem) or ""
        if not tx:
            continue  # без транскрипта пропускаем
        manifest.append({"audio": str(dst), "text": tx})

    manifest_path = ds_dir / "train.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for it in manifest:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    return manifest_path, len(manifest)


def train_lora(name, files, transcripts, r, alpha, steps, lr, progress=gr.Progress()):
    """Запустить обучение LoRA. Yield-генератор для live log."""
    import subprocess, yaml, json
    if not name or not name.strip():
        yield "❌ Укажи имя LoRA (имя папки-результата)"
        return
    name = name.strip().replace(" ", "_")

    if not files:
        yield "❌ Загрузите аудио-файлы для тренировки"
        return

    train_script = get_training_script()
    if train_script is None:
        yield "❌ training/scripts/train_voxcpm_finetune.py не найден"
        return

    progress(0.1, desc="Готовлю датасет...")
    manifest, n = prepare_train_data(name, files, transcripts)
    if n == 0:
        yield "❌ Нет валидных сэмплов. Проверь транскрипты (формат: имя_файла.wav|текст)"
        return
    yield f"✓ Dataset: {n} сэмплов → {manifest}"

    save_path = LORA_DIR / name
    save_path.mkdir(exist_ok=True)
    config_path = TRAIN_DATA_DIR / name / "train_config.yaml"

    # Путь к уже скачанному VoxCPM2 (в models/ через HF cache)
    # Ищем snapshot
    from huggingface_hub import snapshot_download
    try:
        pretrained = snapshot_download("openbmb/VoxCPM2", local_files_only=True)
    except Exception:
        pretrained = snapshot_download("openbmb/VoxCPM2")

    cfg = {
        "pretrained_path": pretrained,
        "train_manifest": str(manifest),
        "sample_rate": 16000,  # AudioVAE encodes 16kHz (issue #202 fix)
        "out_sample_rate": 48000,
        "batch_size": 1,
        "grad_accum_steps": 16,
        "num_workers": 0,
        "num_iters": int(steps),
        "log_interval": 10,
        "valid_interval": max(100, int(steps) // 2),
        "save_interval": max(100, int(steps) // 2),
        "learning_rate": float(lr),
        "weight_decay": 0.01,
        "warmup_steps": min(100, int(steps) // 10),
        "max_steps": int(steps),
        "max_batch_tokens": 8192,
        "save_path": str(save_path),
        "lambdas": {"loss/diff": 1.0, "loss/stop": 1.0},
        "lora": {
            "enable_lm": True,
            "enable_dit": True,
            "enable_proj": False,
            "r": int(r),
            "alpha": int(alpha),
            "dropout": 0.0,
        },
    }
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)
    yield f"✓ Config: {config_path}"
    yield f"Запуск тренировки (steps={steps}, r={r}, α={alpha}, lr={lr})..."

    progress(0.15, desc="Старт тренировки...")

    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.Popen(
        [sys.executable, str(train_script), "--config_path", str(config_path)],
        cwd=str(TRAINING_DIR),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env=env, text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    log_lines = []
    try:
        for line in proc.stdout:
            line = line.rstrip()
            log_lines.append(line)
            # Парсим прогресс из логов (iter XX/YYY)
            import re
            m = re.search(r'iter[:\s]+(\d+)\s*/\s*(\d+)', line, re.I) or re.search(r'step[:\s]+(\d+)\s*/\s*(\d+)', line, re.I)
            if m:
                cur, total = int(m.group(1)), int(m.group(2))
                progress(0.15 + 0.8 * cur / max(total, 1), desc=f"Step {cur}/{total}")
            # yield последние 40 строк как log
            yield "\n".join(log_lines[-40:])
    except Exception as exc:
        log_lines.append(f"[exception] {exc}")
        yield "\n".join(log_lines[-40:])
    proc.wait()
    if proc.returncode == 0:
        progress(1.0, desc="Готово")
        log_lines.append(f"\n✅ Готово! LoRA сохранена в {save_path}")
    else:
        log_lines.append(f"\n❌ Тренировка завершилась с кодом {proc.returncode}")
    yield "\n".join(log_lines[-60:])


def download_selected_voices(selected):
    """Скачать выбранные голоса. Возвращает (status, updated_dropdown_choices)."""
    if not selected:
        return (
            "Выберите голоса для скачивания.",
            gr.update(),
        )
    ok, fail = 0, 0
    for v in selected:
        if download_cloud_voice(v):
            ok += 1
        else:
            fail += 1
    return (
        f"Скачано: {ok} из {len(selected)}. Ошибок: {fail}.",
        gr.update(choices=["-- Свой файл --"] + scan_local_voices()),
    )

# === Определение устройства ===
def _detect_device() -> tuple[str, str]:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        return "cuda", f"{name} | VRAM: {vram:.1f} GB"
    return "cpu", "CPU (экспериментально / experimental — very slow)"

DEVICE, DEVICE_INFO = _detect_device()
print(f"[VoxCPM2] Device: {DEVICE_INFO}")

# === Ленивая загрузка модели ===
_model = None

def get_model(lora_weights_path: Optional[str] = None, force_reload: bool = False) -> VoxCPM:
    """Загружает модель. При первом lora_weights_path — модель перезагружается с LoRA-структурой
    (нужно для load_lora hot-swap в дальнейшем)."""
    global _model
    if _model is not None and not force_reload and lora_weights_path is None:
        return _model
    if force_reload and _model is not None:
        del _model
        import gc; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _model = None
    print(f"[VoxCPM2] Loading {MODEL_REF}" + (f" + LoRA {lora_weights_path}" if lora_weights_path else ""))
    kwargs = dict(load_denoiser=True, optimize=False)
    if lora_weights_path:
        kwargs["lora_weights_path"] = lora_weights_path
    _model = VoxCPM.from_pretrained(MODEL_REF, **kwargs)
    print(f"[VoxCPM2] Model loaded. Sample rate: {_model.tts_model.sample_rate} Hz")
    return _model


# === Вспомогательные утилиты ===
def _resolve_seed(seed, locked: bool) -> int:
    if locked and seed is not None and int(seed) >= 0:
        s = int(seed)
    else:
        s = random.randint(0, 2**31 - 1)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    return s


def _collect_audio(result) -> np.ndarray:
    """Обрабатывает как generator (потоковая генерация), так и np.ndarray."""
    if isinstance(result, np.ndarray):
        return result
    chunks = []
    for chunk in result:
        arr = np.asarray(chunk)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        chunks.append(arr)
    if not chunks:
        raise gr.Error("Модель не вернула аудио / Model returned no audio.")
    return np.concatenate(chunks)


def _generate_audio(model, kwargs: dict, streaming: bool, progress) -> np.ndarray:
    """Единый роутер: streaming → generate_streaming (с прогрессом по чанкам),
    иначе обычный generate."""
    if not streaming:
        progress(0, desc="Генерация...")
        return _collect_audio(model.generate(**kwargs))
    # streaming mode
    chunks = []
    gen = model.generate_streaming(**kwargs)
    for i, chunk in enumerate(gen):
        arr = np.asarray(chunk)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        chunks.append(arr)
        try:
            progress(min(0.99, i / 120), desc=f"Chunk {i+1}")
        except Exception:
            pass
    if not chunks:
        raise gr.Error("Модель не вернула аудио / Model returned no audio.")
    return np.concatenate(chunks)


def _save_wav(wav: np.ndarray, sr: int, prefix: str = "tts", fmt: str = "mp3") -> str:
    """Сохранить аудио в output/ в формате fmt (mp3/wav/flac/ogg)."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fmt = (fmt or "mp3").lower()

    # WAV и FLAC умеет напрямую soundfile
    if fmt in ("wav", "flac"):
        out_path = OUTPUT_DIR / f"{prefix}_{ts}.{fmt}"
        sf.write(str(out_path), wav, sr, format=fmt.upper())
        return str(out_path)

    # MP3/OGG — через ffmpeg (WAV -> сжатый формат)
    import subprocess
    tmp_wav = OUTPUT_DIR / f"_tmp_{prefix}_{ts}.wav"
    out_path = OUTPUT_DIR / f"{prefix}_{ts}.{fmt}"
    sf.write(str(tmp_wav), wav, sr)
    try:
        codec_args = {
            "mp3": ["-codec:a", "libmp3lame", "-b:a", "192k"],
            "ogg": ["-codec:a", "libvorbis", "-q:a", "5"],
        }.get(fmt, [])
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(tmp_wav), *codec_args, str(out_path)],
            check=True, capture_output=True, timeout=60,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
        # Fallback: возвращаем WAV если ffmpeg не сработал
        print(f"[save] ffmpeg {fmt} encode failed, returning WAV: {exc}")
        return str(tmp_wav)
    finally:
        try:
            if tmp_wav.exists() and out_path.exists():
                tmp_wav.unlink()
        except Exception:
            pass
    return str(out_path)


def _build_kwargs(
    *,
    text: str,
    cfg: float,
    steps: int,
    normalize: bool,
    retry: bool,
    retry_max: int = 3,
    retry_ratio: float = 6.0,
    min_len: int = 2,
    max_len: int = 4096,
    reference_wav_path: Optional[str] = None,
    prompt_wav_path: Optional[str] = None,
    prompt_text: Optional[str] = None,
    denoise: Optional[bool] = None,
) -> dict:
    kwargs = {
        "text": text,
        "cfg_value": float(cfg),
        "inference_timesteps": int(steps),
        "normalize": bool(normalize),
        "retry_badcase": bool(retry),
        "retry_badcase_max_times": int(retry_max),
        "retry_badcase_ratio_threshold": float(retry_ratio),
        "min_len": int(min_len),
        "max_len": int(max_len),
    }
    if reference_wav_path:
        kwargs["reference_wav_path"] = reference_wav_path
    if prompt_wav_path:
        kwargs["prompt_wav_path"] = prompt_wav_path
    if prompt_text:
        kwargs["prompt_text"] = prompt_text
    if denoise is not None and (reference_wav_path or prompt_wav_path):
        kwargs["denoise"] = bool(denoise)
    return kwargs


# === Функции генерации ===
def tts_generate(text, cfg, steps, fmt, retry_max, retry_ratio, min_len, max_len, streaming, seed, locked, normalize, retry, progress=gr.Progress()):
    if not (text or "").strip():
        raise gr.Error("Введите текст / Please enter text.")
    try:
        model = get_model()
        used_seed = _resolve_seed(seed, locked)
        kwargs = _build_kwargs(
            text=text.strip(), cfg=cfg, steps=steps,
            normalize=normalize, retry=retry,
            retry_max=retry_max, retry_ratio=retry_ratio,
            min_len=min_len, max_len=max_len,
        )
        wav = _generate_audio(model, kwargs, bool(streaming), progress)
        return _save_wav(wav, model.tts_model.sample_rate, "tts", fmt), used_seed
    except gr.Error:
        raise
    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"Ошибка генерации / Generation error: {e}") from e


def voice_design(description, text, cfg, steps, fmt, retry_max, retry_ratio, min_len, max_len, streaming, seed, locked, normalize, retry, progress=gr.Progress()):
    if not (text or "").strip():
        raise gr.Error("Введите текст / Please enter text.")
    if not (description or "").strip():
        raise gr.Error("Введите описание голоса / Please enter voice description.")
    try:
        model = get_model()
        combined = f"({description.strip()}){text.strip()}"
        used_seed = _resolve_seed(seed, locked)
        kwargs = _build_kwargs(
            text=combined, cfg=cfg, steps=steps,
            normalize=normalize, retry=retry,
            retry_max=retry_max, retry_ratio=retry_ratio,
            min_len=min_len, max_len=max_len,
        )
        wav = _generate_audio(model, kwargs, bool(streaming), progress)
        return _save_wav(wav, model.tts_model.sample_rate, "design", fmt), used_seed
    except gr.Error:
        raise
    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"Ошибка генерации / Generation error: {e}") from e


def _numpy_to_tempfile(ref_audio):
    """Сохранить numpy-аудио (sr, wav) во временный wav и вернуть путь.
    Если на вход path (str) — вернуть как есть."""
    if ref_audio is None:
        return None
    if isinstance(ref_audio, str):
        return ref_audio
    sr, wav = ref_audio
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    sf.write(tmp.name, wav, sr)
    return tmp.name


def voice_clone(text, ref_audio, style, transcript, cfg, steps, fmt, retry_max, retry_ratio, min_len, max_len, streaming, seed, locked, normalize, denoise, retry, progress=gr.Progress()):
    """Voice Cloning. Если transcript заполнен — автоматически Ultimate-режим."""
    if not (text or "").strip():
        raise gr.Error("Введите текст / Please enter text.")
    if ref_audio is None:
        raise gr.Error("Загрузите референс-аудио / Please upload reference audio.")
    ref_audio = _numpy_to_tempfile(ref_audio)
    try:
        model = get_model()
        final_text = text.strip()
        if style and style.strip():
            final_text = f"({style.strip()}){final_text}"
        used_seed = _resolve_seed(seed, locked)

        transcript_clean = (transcript or "").strip()
        common_kwargs = dict(
            text=final_text, cfg=cfg, steps=steps,
            normalize=normalize, retry=retry,
            retry_max=retry_max, retry_ratio=retry_ratio,
            min_len=min_len, max_len=max_len,
            denoise=denoise,
        )
        if transcript_clean:
            kwargs = _build_kwargs(
                **common_kwargs,
                prompt_wav_path=ref_audio, prompt_text=transcript_clean,
                reference_wav_path=ref_audio,
            )
        else:
            kwargs = _build_kwargs(
                **common_kwargs,
                reference_wav_path=ref_audio,
            )
        wav = _generate_audio(model, kwargs, bool(streaming), progress)
        return _save_wav(wav, model.tts_model.sample_rate, "clone", fmt), used_seed
    except gr.Error:
        raise
    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"Ошибка генерации / Generation error: {e}") from e




# === Примеры (gr.Examples) ===
TTS_EXAMPLES = [
    ["Привет! Это портативная версия VoxCPM2 от Nerual Dreming и Нейро-Софт."],
    ["Hello! This is the portable VoxCPM2 build."],
    ["Сегодня прекрасная погода, солнце светит ярко, и птицы поют в парке."],
    ["The quick brown fox jumps over the lazy dog."],
]

DESIGN_EXAMPLES = [
    ["Молодая женщина, нежный и мягкий голос", "Привет, добро пожаловать в VoxCPM2!"],
    ["Пожилой мужчина с глубоким баритоном, говорит медленно и внушительно", "Давным-давно, в далёкой галактике..."],
    ["Весёлый ребёнок, энергично и быстро", "Ура! Сегодня выходной!"],
    ["A young woman with a soft gentle voice", "Hello, welcome to VoxCPM2!"],
    ["An elderly British man, deep and authoritative", "Once upon a time, in a galaxy far far away..."],
    ["A cheerful child, energetic and playful", "Yay! It's weekend today!"],
]

CLONE_STYLE_EXAMPLES = [
    "чуть быстрее, бодрым тоном",
    "медленно и драматично",
    "шёпотом, интимно",
    "slightly faster, cheerful tone",
    "slow and dramatic",
    "whispering, intimate",
]


# === CSS (тёмная тема + gradient header) ===
_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
* { font-family: 'Inter', 'Segoe UI', sans-serif !important; }
.gradio-container { max-width: 1400px !important; margin: auto !important; width: 100% !important; }
.gradio-container > *, .gradio-container .row { width: 100% !important; }

.brand-header {
  text-align: center;
  background: linear-gradient(135deg, #4c1d95 0%, #6d28d9 50%, #7e22ce 100%);
  padding: 28px 20px;
  border-radius: 16px;
  margin: 8px 0 16px 0;
  box-shadow: 0 10px 30px rgba(109, 40, 217, 0.35);
  color: white;
}
.brand-title { font-size: 1.9em; font-weight: 700; margin: 0 0 6px 0; }
.brand-subtitle { font-size: 1em; opacity: 0.9; margin-bottom: 14px; }
.brand-credits { font-size: 0.9em; opacity: 0.95; }
.brand-credits a { color: #fbbf24; text-decoration: none; font-weight: 600; }
.brand-credits a:hover { text-decoration: underline; }
.device-badge {
  display: inline-block;
  background: rgba(255,255,255,0.15);
  padding: 4px 12px;
  border-radius: 999px;
  font-size: 0.85em;
  margin-top: 10px;
}
/* Tabs — равные по ширине */
.tabs > div[role="tablist"] > button,
.tab-nav > button {
  flex: 1 !important;
  text-align: center !important;
}
.lang-switcher-top {
  text-align: right; padding: 8px 12px 0 0;
}
.brand-box {
  background: linear-gradient(135deg, #4c1d95 0%, #6d28d9 50%, #7e22ce 100%);
  padding: 24px 28px;
  border-radius: 16px;
  margin: 8px 0 16px 0;
  box-shadow: 0 10px 30px rgba(109, 40, 217, 0.35);
  color: white;
  text-align: center;
}
.brand-box h1 { color: white !important; margin: 0 0 8px 0 !important; font-size: 1.9em !important; }
.brand-box p { color: rgba(255,255,255,0.92) !important; margin: 4px 0 !important; }
.brand-box a { color: #fbbf24 !important; text-decoration: none !important; font-weight: 600 !important; }
.brand-box a:hover { text-decoration: underline !important; }
.lang-switcher {
  position: absolute; top: 12px; right: 16px;
  display: flex; gap: 6px;
}
.lang-btn {
  background: rgba(255,255,255,0.18);
  color: white !important;
  padding: 5px 10px;
  border-radius: 8px;
  font-size: 0.82em;
  text-decoration: none !important;
  font-weight: 600;
}
.lang-btn:hover { background: rgba(255,255,255,0.3); }
.brand-header { position: relative; }

button.primary {
  background: linear-gradient(135deg, #6d28d9 0%, #7e22ce 100%) !important;
  color: white !important;
  font-weight: 600 !important;
  border-radius: 10px !important;
}
"""

# === JS: принудительно тёмная тема (IIFE — одноразовый редирект с __theme=dark) ===
_HEAD_JS = """
() => {
  const url = new URL(window.location);
  if (!url.searchParams.has('__theme')) {
    url.searchParams.set('__theme', 'dark');
    window.location.replace(url.toString());
  }
}
"""

def _brand_html(subtitle: str, credits_label: str) -> str:
    return f"""
<div class="brand-header">
  <div class="lang-switcher">
    <a href="?__lang=ru&__theme=dark" class="lang-btn">
      <img src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f1f7-1f1fa.svg" width="16" height="16" style="vertical-align:-3px;margin-right:4px"/>RU
    </a>
    <a href="?__lang=en&__theme=dark" class="lang-btn">
      <img src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f1ec-1f1e7.svg" width="16" height="16" style="vertical-align:-3px;margin-right:4px"/>EN
    </a>
  </div>
  <div class="brand-title">🎙️ VoxCPM2 — Multilingual TTS</div>
  <div class="brand-subtitle">{subtitle}</div>
  <div class="brand-credits">
    {credits_label}:
    <a href="https://t.me/nerual_dreming" target="_blank">Nerual Dreming</a> ·
    <a href="https://neuro-cartel.com" target="_blank">neuro-cartel.com</a> ·
    <a href="https://t.me/neuroport" target="_blank">Нейро-Софт</a>
  </div>
  <div class="device-badge">💻 {DEVICE_INFO}</div>
</div>
"""

_BRAND_HTML_RU = _brand_html(
    "2B параметров · 30 языков · 48 kHz · Voice Design &amp; Cloning",
    "Портативная сборка",
)
_BRAND_HTML_EN = _brand_html(
    "2B parameters · 30 languages · 48 kHz · Voice Design &amp; Cloning",
    "Portable build",
)


# === UI Builder ===
def _seed_row(prefix: str):
    with gr.Row():
        seed = gr.Number(value=-1, label=I18N("label_seed"), precision=0, scale=3, elem_id=f"{prefix}_seed")
        locked = gr.Checkbox(value=False, label=I18N("label_lock"), scale=1, elem_id=f"{prefix}_locked")
    return seed, locked


def _advanced_block(prefix: str, show_denoise: bool = False):
    """Accordion с расширенными параметрами."""
    with gr.Accordion(label=I18N("label_advanced"), open=False, elem_id=f"{prefix}_advanced"):
        # LoRA — просто dropdown. '-- Без LoRA --' = detach. Любое другое = attach.
        _LORA_NONE = "-- Без LoRA --"
        _lora_choices = [_LORA_NONE] + scan_local_loras()
        with gr.Row():
            lora_sel = gr.Dropdown(
                label="🧬 LoRA (fine-tuned веса из папки lora/)",
                choices=_lora_choices,
                value=_ACTIVE_LORA if _ACTIVE_LORA in _lora_choices else _LORA_NONE,
                interactive=True, scale=4, elem_id=f"{prefix}_lora",
            )
            lora_refresh = gr.Button("🔄", size="sm", scale=0, elem_id=f"{prefix}_lora_refresh")

        def _on_lora_change(name):
            if not name or name == _LORA_NONE:
                lora_detach()
            else:
                lora_attach(name)

        lora_sel.change(_on_lora_change, inputs=[lora_sel], outputs=[], show_progress="hidden")
        lora_refresh.click(
            fn=lambda: gr.update(choices=[_LORA_NONE] + scan_local_loras()),
            outputs=[lora_sel],
        )
        with gr.Row():
            cfg = gr.Slider(0.5, 5.0, value=2.0, step=0.1, label=I18N("label_cfg"), info="Выше = ближе к промпту, ниже = больше креатива", elem_id=f"{prefix}_cfg")
            steps = gr.Slider(5, 30, value=10, step=1, label=I18N("label_steps"), info="Больше = качественнее, но медленнее (5-10 для скорости)", elem_id=f"{prefix}_steps")
        with gr.Row():
            min_len = gr.Slider(1, 100, value=2, step=1, label=I18N("label_min_len"), info="Минимальная длина аудио (токенов)", elem_id=f"{prefix}_min_len")
            max_len = gr.Slider(512, 8192, value=4096, step=256, label=I18N("label_max_len"), info="Максимальная длина генерации", elem_id=f"{prefix}_max_len")
        fmt = gr.Radio(choices=["mp3", "wav", "flac", "ogg"], value="mp3", label=I18N("label_format"), elem_id=f"{prefix}_fmt")
        normalize = gr.Checkbox(value=True, label=I18N("label_normalize"), info="Обработка чисел, дат, сокращений", elem_id=f"{prefix}_normalize")
        retry = gr.Checkbox(value=False, label=I18N("label_retry"), info="Перегенерировать если качество плохое", elem_id=f"{prefix}_retry")
        with gr.Row():
            retry_max = gr.Slider(1, 10, value=3, step=1, label=I18N("label_retry_max"), info="Макс. число попыток повтора", elem_id=f"{prefix}_retry_max")
            retry_ratio = gr.Slider(2.0, 20.0, value=6.0, step=0.5, label=I18N("label_retry_ratio"), info="Порог детекции плохой генерации", elem_id=f"{prefix}_retry_ratio")
        streaming = gr.Checkbox(value=True, label=I18N("label_streaming"), info="Потоковая генерация с live-прогрессом", elem_id=f"{prefix}_streaming")
        denoise = None
        if show_denoise:
            denoise = gr.Checkbox(value=False, label=I18N("label_denoise"), info="ZipEnhancer денойзинг референс-аудио", elem_id=f"{prefix}_denoise")
    return cfg, steps, fmt, normalize, retry, retry_max, retry_ratio, min_len, max_len, streaming, denoise


# === i18n — официальный паттерн Gradio 6 ===
# https://gradio.app/guides/internationalization
I18N = gr.I18n(
    en={
        "tab_tts": "Text-to-Speech",
        "tab_design": "Voice Design",
        "tab_clone": "Voice Cloning",
        "tab_ultimate": "Ultimate Cloning",
        "tts_instructions": "Enter any text, 30 languages supported.",
        "design_instructions": "Create a voice from a text description — no reference needed.",
        "clone_instructions": "Upload a reference or pick from the pack. Transcript is optional.",
        "ultimate_instructions": "Maximum fidelity — audio + transcript required.",
        "label_text": "Text",
        "label_description": "Voice description",
        "label_content": "Text to speak",
        "label_reference": "Reference audio",
        "label_transcript": "Reference transcript",
        "label_style": "Style (optional)",
        "label_preset": "Voice preset",
        "label_cfg": "CFG Scale",
        "label_steps": "Inference Steps",
        "label_seed": "Seed",
        "label_lock": "Lock Seed",
        "label_advanced": "Advanced settings",
        "label_normalize": "Text normalization (wetext)",
        "label_denoise": "Denoise reference",
        "label_retry": "Retry on bad case",
        "label_output": "Generated audio",
        "btn_tts": "Synthesize",
        "btn_design": "Design Voice",
        "btn_clone": "Clone Voice",
        "btn_ultimate": "Ultimate Clone",
        "btn_refresh": "Refresh list",
        "btn_load_cloud": "Load list",
        "btn_download": "Download selected",
        "accordion_cloud": "Download voices from server",
        "cloud_status_initial": "Click 'Load list' to fetch available voices",
        "cloud_voices_label": "Available voices",
        "download_status_label": "Download result",
        "brand_header_html": _BRAND_HTML_EN,
        "transcript_info": "Fill in to enable Ultimate mode (max fidelity). Auto-fills from voice pack.",
        "label_format": "Output format",
        "label_min_len": "Min audio length",
        "label_max_len": "Max generation length",
        "label_retry_max": "Max retry attempts",
        "label_retry_ratio": "Bad-case ratio threshold",
        "label_streaming": "Streaming mode (live progress)",
        "tab_lora": "LoRA",
        "lora_attach_title": "🔌 Attach trained LoRA",
        "lora_train_title": "🎓 Train new LoRA",
    },
    ru={
        "tab_tts": "Текст в речь",
        "tab_design": "Дизайн голоса",
        "tab_clone": "Клонирование",
        "tab_ultimate": "Ultimate-клонирование",
        "tts_instructions": "Введите любой текст, поддерживается 30 языков.",
        "design_instructions": "Создайте голос из текстового описания — без референса.",
        "clone_instructions": "Загрузите референс или выберите из пака. Транскрипт опционален.",
        "ultimate_instructions": "Максимальная верность — аудио + транскрипт обязательны.",
        "label_text": "Текст",
        "label_description": "Описание голоса",
        "label_content": "Текст для озвучки",
        "label_reference": "Референс-аудио",
        "label_transcript": "Транскрипт референса",
        "label_style": "Стиль (опционально)",
        "label_preset": "Пресет голоса",
        "label_cfg": "CFG Scale",
        "label_steps": "Шаги диффузии",
        "label_seed": "Seed",
        "label_lock": "Зафиксировать Seed",
        "label_advanced": "Расширенные настройки",
        "label_normalize": "Нормализация текста (wetext)",
        "label_denoise": "Шумоподавление референса",
        "label_retry": "Повтор при плохой генерации",
        "label_output": "Результат",
        "btn_tts": "Синтезировать",
        "btn_design": "Создать голос",
        "btn_clone": "Клонировать",
        "btn_ultimate": "Ultimate-клонировать",
        "btn_refresh": "Обновить список",
        "btn_load_cloud": "Загрузить список",
        "btn_download": "Скачать выбранные",
        "accordion_cloud": "Скачать голоса с сервера",
        "cloud_status_initial": "Нажмите 'Загрузить список' для получения доступных голосов",
        "cloud_voices_label": "Доступные голоса",
        "download_status_label": "Результат загрузки",
        "brand_header_html": _BRAND_HTML_RU,
        "transcript_info": "Заполните для Ultimate-режима (макс. качество). Автозаполняется из пака.",
        "label_format": "Формат вывода",
        "label_min_len": "Мин. длина аудио",
        "label_max_len": "Макс. длина генерации",
        "label_retry_max": "Макс. попыток повтора",
        "label_retry_ratio": "Порог плохой генерации",
        "label_streaming": "Потоковая генерация (live)",
        "tab_lora": "LoRA",
        "lora_attach_title": "🔌 Подключить готовую LoRA",
        "lora_train_title": "🎓 Обучить новую LoRA",
    },
)


_HEAD_SCRIPT = """
<script>
(function(){
  try {
    var u = new URL(window.location);
    var lang = u.searchParams.get('__lang');
    if (lang) {
      Object.defineProperty(navigator, 'language', {value: lang, configurable: true});
      Object.defineProperty(navigator, 'languages', {value: [lang], configurable: true});
    }
  } catch(e) {}
})();
</script>
"""


def build_ui():
    # Gradio 6.10: theme/css/head passed to launch(), not Blocks()
    with gr.Blocks(
        title="VoxCPM2 — Multilingual TTS",
        delete_cache=(300, 3600),
    ) as demo:
        gr.HTML(I18N("brand_header_html"))

        # === Таб 1: TTS ===
        with gr.Tab(label=I18N("tab_tts")):
            gr.Markdown(I18N("tts_instructions"))
            with gr.Row(equal_height=False):
                with gr.Column(scale=1):
                    tts_text = gr.Textbox(label=I18N("label_text"), placeholder="Введите текст на любом из 30 поддерживаемых языков...", lines=4)
                    tts_cfg, tts_steps, tts_fmt, tts_norm, tts_retry, tts_retry_max, tts_retry_ratio, tts_min_len, tts_max_len, tts_stream, _ = _advanced_block("tts", show_denoise=False)
                    tts_seed, tts_locked = _seed_row("tts")
                    tts_btn = gr.Button(I18N("btn_tts"), variant="primary", size="lg")
                with gr.Column(scale=1):
                    tts_out = gr.Audio(label=I18N("label_output"), type="filepath")
            gr.Examples(examples=TTS_EXAMPLES, inputs=[tts_text], label="Примеры / Examples", examples_per_page=10)
            tts_btn.click(
                tts_generate,
                inputs=[tts_text, tts_cfg, tts_steps, tts_fmt, tts_retry_max, tts_retry_ratio, tts_min_len, tts_max_len, tts_stream, tts_seed, tts_locked, tts_norm, tts_retry],
                outputs=[tts_out, tts_seed],
            )

        # === Таб 2: Voice Design ===
        with gr.Tab(label=I18N("tab_design")):
            gr.Markdown(I18N("design_instructions"))
            with gr.Row(equal_height=False):
                with gr.Column(scale=1):
                    vd_desc = gr.Textbox(label=I18N("label_description"), placeholder="например: Молодая женщина, нежный и мягкий голос", lines=2)
                    vd_text = gr.Textbox(label=I18N("label_content"), placeholder="Привет, добро пожаловать в VoxCPM2!", lines=3)
                    vd_cfg, vd_steps, vd_fmt, vd_norm, vd_retry, vd_retry_max, vd_retry_ratio, vd_min_len, vd_max_len, vd_stream, _ = _advanced_block("vd", show_denoise=False)
                    vd_seed, vd_locked = _seed_row("vd")
                    vd_btn = gr.Button(I18N("btn_design"), variant="primary", size="lg")
                with gr.Column(scale=1):
                    vd_out = gr.Audio(label=I18N("label_output"), type="filepath")
            # Voice Design examples — кнопки, заполняющие оба поля (gr.Examples не резолвит gr.I18n в заголовках)
            gr.Markdown("**Примеры описаний голоса / Voice design examples** (кликни, чтобы подставить):")
            with gr.Row():
                for i, (desc, txt) in enumerate(DESIGN_EXAMPLES):
                    ex_btn = gr.Button(desc[:40] + ("…" if len(desc) > 40 else ""), size="sm")
                    ex_btn.click(
                        fn=lambda d=desc, t=txt: (d, t),
                        inputs=[],
                        outputs=[vd_desc, vd_text],
                    )
            vd_btn.click(
                voice_design,
                inputs=[vd_desc, vd_text, vd_cfg, vd_steps, vd_fmt, vd_retry_max, vd_retry_ratio, vd_min_len, vd_max_len, vd_stream, vd_seed, vd_locked, vd_norm, vd_retry],
                outputs=[vd_out, vd_seed],
            )

        _initial_voices = scan_local_voices()
        # Limit to first 50 — Gradio 6 Dropdown has reactive issues with very large lists
        _voice_choices = ["-- Свой файл --"] + _initial_voices[:50]

        # === Таб 3: Voice Cloning ===
        with gr.Tab(label=I18N("tab_clone")):
            gr.Markdown(I18N("clone_instructions"))
            with gr.Row(equal_height=False):
                with gr.Column(scale=1):
                    vc_voice_pick = gr.Dropdown(
                        label=I18N("label_preset"),
                        choices=_voice_choices,
                        value="-- Свой файл --",
                        interactive=True,
                        elem_id="vc_voice_pick",
                    )
                    vc_refresh_btn = gr.Button(I18N("btn_refresh"), size="sm", elem_id="vc_refresh_btn")
                    vc_ref = gr.Audio(
                        label=I18N("label_reference"),
                        type="numpy",
                        sources=["upload", "microphone"],
                        elem_id="vc_ref",
                    )
                    vc_transcript = gr.Textbox(
                        label=I18N("label_transcript"),
                        info=I18N("transcript_info"),
                        placeholder="Точный текст того что говорится в референс-аудио",
                        lines=2,
                        elem_id="vc_transcript",
                    )
                    vc_text = gr.Textbox(label=I18N("label_content"), placeholder="Привет, добро пожаловать в VoxCPM2!", lines=3, elem_id="vc_text")
                    vc_style = gr.Textbox(
                        label=I18N("label_style"),
                        placeholder="например: чуть быстрее, бодрым тоном",
                        lines=1,
                        elem_id="vc_style",
                    )
                    vc_cfg, vc_steps, vc_fmt, vc_norm, vc_retry, vc_retry_max, vc_retry_ratio, vc_min_len, vc_max_len, vc_stream, vc_denoise = _advanced_block("vc", show_denoise=True)
                    vc_seed, vc_locked = _seed_row("vc")
                    vc_btn = gr.Button(I18N("btn_clone"), variant="primary", size="lg", elem_id="vc_btn")
                with gr.Column(scale=1):
                    vc_out = gr.Audio(label=I18N("label_output"), type="filepath", elem_id="vc_out")
                    with gr.Accordion(I18N("accordion_cloud"), open=False, elem_id="vc_download_accordion"):
                        gr.Markdown(f"*Репозиторий: `{CLOUD_VOICES_REPO}`*")
                        vc_cloud_status = gr.Textbox(
                            label="Статус",
                            interactive=False,
                            value=I18N("cloud_status_initial"),
                            elem_id="vc_cloud_status",
                        )
                        vc_load_cloud_btn = gr.Button(I18N("btn_load_cloud"), variant="secondary", elem_id="vc_load_cloud_btn")
                        vc_cloud_voices = gr.CheckboxGroup(
                            label=I18N("cloud_voices_label"),
                            choices=[],
                            interactive=True,
                            elem_id="vc_cloud_voices",
                        )
                        vc_download_btn = gr.Button(I18N("btn_download"), variant="primary", elem_id="vc_download_btn")
                        vc_download_status = gr.Textbox(label=I18N("download_status_label"), interactive=False, elem_id="vc_download_status")

            vc_btn.click(
                voice_clone,
                inputs=[vc_text, vc_ref, vc_style, vc_transcript, vc_cfg, vc_steps, vc_fmt, vc_retry_max, vc_retry_ratio, vc_min_len, vc_max_len, vc_stream, vc_seed, vc_locked, vc_norm, vc_denoise, vc_retry],
                outputs=[vc_out, vc_seed],
            )

            def _vc_load_preset(name):
                if not name or name == "-- Свой файл --":
                    return None, ""
                path = voice_audio_path(name)
                if not path:
                    return None, ""
                wav, sr = sf.read(path)
                return (sr, wav), voice_transcript(name)

            def _vc_refresh():
                return gr.update(choices=["-- Свой файл --"] + scan_local_voices())

            vc_voice_pick.change(_vc_load_preset, inputs=[vc_voice_pick], outputs=[vc_ref, vc_transcript])
            vc_refresh_btn.click(_vc_refresh, outputs=[vc_voice_pick])
            vc_load_cloud_btn.click(load_cloud_list, outputs=[vc_cloud_status, vc_cloud_voices])
            vc_download_btn.click(download_selected_voices, inputs=[vc_cloud_voices], outputs=[vc_download_status, vc_voice_pick])

        # === Таб 4: Обучение LoRA ===
        with gr.Tab(label=I18N("tab_lora")):
            gr.Markdown(f"### {I18N('lora_train_title')}")
            gr.Markdown(
                "**Как подготовить датасет:** загрузите 5-50 аудио (wav/mp3/flac, 3-15 сек каждое), "
                "ниже в поле транскриптов построчно: `имя_файла.wav|точный текст`.\n\n"
                "**Минимум**: 5-10 минут аудио. Оптимально — чистая запись одного голоса."
            )
            lora_name = gr.Textbox(label="Имя LoRA (папка-результат)", placeholder="my_voice_v1", value="")
            lora_files = gr.Files(
                label="Аудиофайлы (wav/mp3/flac/m4a)",
                file_types=[".wav", ".mp3", ".flac", ".m4a", ".ogg"],
                file_count="multiple",
            )
            lora_transcripts = gr.Textbox(
                label="Транскрипты (формат: имя_файла|текст)",
                placeholder="clip_001.wav|Здравствуйте, меня зовут Иван.\nclip_002.wav|Сегодня отличная погода.",
                lines=8,
            )
            with gr.Row():
                lora_r = gr.Slider(8, 128, value=16, step=8, label="LoRA rank (r)", info="Больше = больше capacity (16 для 10-20 клипов, 32 для 50+, 64 для 500+)")
                lora_alpha = gr.Slider(8, 128, value=16, step=8, label="LoRA alpha", info="Обычно = r")
            with gr.Row():
                lora_steps = gr.Slider(100, 5000, value=300, step=50, label="Training steps", info="300 для 10-20 клипов, 500-1000 для 50-100, 2000+ для 500+")
                lora_lr = gr.Slider(0.00001, 0.001, value=0.0001, step=0.00001, label="Learning rate", info="0.0001 стандарт")
            lora_train_btn = gr.Button("🎓 Начать обучение", variant="primary", size="lg")
            lora_train_log = gr.Textbox(label="Лог тренировки", interactive=False, lines=15)

            lora_train_btn.click(
                train_lora,
                inputs=[lora_name, lora_files, lora_transcripts, lora_r, lora_alpha, lora_steps, lora_lr],
                outputs=[lora_train_log],
            )

    return demo


# === Точка входа ===
if __name__ == "__main__":
    demo = build_ui()
    demo.queue(default_concurrency_limit=1).launch(
        server_port=None,
        inbrowser=True,
        i18n=I18N,
        theme=gr.themes.Soft(primary_hue="purple"),
        css=_CSS,
        head=_HEAD_SCRIPT,
        show_error=True,
    )
