"""VoxCPM2 Portable — Multilingual TTS (RU/EN).

Портативная русскоязычная сборка VoxCPM2 от Nerual Dreming + Нейро-Софт.
Поддерживает 30 языков, включая русский. Режимы: TTS / Voice Design / Voice Cloning / Ultimate Cloning.
"""

# === КРИТИЧЕСКИЙ ПАТЧ: отключение torch._dynamo ДО импорта voxcpm ===
import os
import sys
import asyncio
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# Windows: форсим UTF-8 stdout/stderr иначе print('❌ ...') ломается
# с 'charmap' codec can't encode character
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

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
# Embedded/portable Python can omit the application directory from sys.path.
# Keep bundled helper modules importable when app.py is launched through an
# alternate entry point (for example the Gradio API smoke-test launcher).
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
OUTPUT_DIR = SCRIPT_DIR / "output"
VOICES_DIR = SCRIPT_DIR / "voices"
LORA_DIR = SCRIPT_DIR / "lora"
TRAIN_DATA_DIR = SCRIPT_DIR / "train_data"
TRAINING_DIR = SCRIPT_DIR / "training"  # bundled train scripts (in repo)

from training.audio_splitting import (
    extract_absolute_words_from_timestamped_segment,
    partition_transcript,
    plan_no_drop_splits,
    slice_audio_file,
    validate_split_plan,
)
from training.local_media import (
    LocalMediaError,
    read_transcripts,
    resolve_audio_files,
    resolve_single_media_file,
)
from training.vram_profiles import (
    VRAM_PROFILES,
    build_training_command,
    load_vram_preference,
    profile_for_vram_gib,
    resolve_vram_profile,
    save_vram_preference,
    validate_lora_hyperparameters,
)

VRAM_PREFERENCE_PATH = SCRIPT_DIR / "cache" / "vram_profile.json"

# === Local model cache setup (always next to app.py) ===
VOXCPM2_CACHE_DIR = Path(__file__).parent.absolute() / "models" / "voxcpm2"
ZIPENHANCER_REPO = "iic/speech_zipenhancer_ans_multiloss_16k_base"
ZIPENHANCER_CACHE_DIR = Path(__file__).parent.absolute() / "models" / "zipenhancer"


def _ensure_voxcpm2_local() -> str:
    """
    Ensure VoxCPM2 model is available locally (offline mode).
    Checks local cache folder first; if missing, downloads via ModelScope/HF.
    Returns the absolute local path to the model directory.
    """
    local_path = VOXCPM2_CACHE_DIR / "openbmb_VoxCPM2"

    # Check if already cached locally (look for config.json or model.safetensors)
    if local_path.exists() and (
        (local_path / "config.json").exists() or 
        (local_path / "model.safetensors").exists() or
        (local_path / "pytorch_model.bin").exists()
    ):
        print(f"[voxcpm2] Using local cache: {local_path}")
        return str(local_path)

    # Need to download
    print(f"[voxcpm2] Model not found locally. Downloading to {VOXCPM2_CACHE_DIR}...")
    VOXCPM2_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Try ModelScope first (better for China/Russia, often faster)
    try:
        from modelscope import snapshot_download
        downloaded_path = snapshot_download(
            "openbmb/VoxCPM2",
            local_dir=str(local_path),
        )
        print(f"[voxcpm2] Downloaded via ModelScope: {downloaded_path}")
        return downloaded_path
    except ImportError:
        print("[voxcpm2] modelscope not installed. Run: pip install modelscope")
    except Exception as e:
        print(f"[voxcpm2] ModelScope download failed: {e}")

    # Fallback: HuggingFace
    try:
        from huggingface_hub import snapshot_download
        downloaded_path = snapshot_download(
            "openbmb/VoxCPM2",
            local_dir=str(local_path),
            local_files_only=False,
        )
        print(f"[voxcpm2] Downloaded via HuggingFace: {downloaded_path}")
        return downloaded_path
    except ImportError:
        print("[voxcpm2] huggingface_hub not installed.")
    except Exception as e:
        print(f"[voxcpm2] HuggingFace download failed: {e}")

    # Final fallback: check if modelscope already downloaded it to its default cache
    try:
        from modelscope.hub.utils.utils import get_cache_dir
        ms_cache = Path(get_cache_dir()) / "hub" / "openbmb_VoxCPM2"
        if ms_cache.exists() and (ms_cache / "config.json").exists():
            print(f"[voxcpm2] Found in ModelScope default cache, copying to {local_path}...")
            import shutil
            shutil.copytree(ms_cache, local_path, dirs_exist_ok=True)
            return str(local_path)
    except Exception:
        pass

    raise RuntimeError(
        f"Could not download VoxCPM2 model. Please manually download it:\n"
        f"  1) Install modelscope: pip install modelscope\n"
        f"  2) Download: modelscope download --model openbmb/VoxCPM2 --local_dir \"{local_path}\"\n"
        f"Or place the model files directly in: {local_path}\n"
        f"Model files should include: config.json, model.safetensors, audiovae.pth, etc."
    )




def _ensure_zipenhancer_local() -> str:
    """
    Ensure ZipEnhancer model is available locally (offline mode).
    Checks local cache folder first; if missing, downloads via ModelScope/HF.
    Returns the absolute local path to the model directory.
    """
    local_path = ZIPENHANCER_CACHE_DIR / "speech_zipenhancer_ans_multiloss_16k_base"

    # Check if already cached locally
    if local_path.exists() and (local_path / "configuration.json").exists():
        print(f"[zipenhancer] Using local cache: {local_path}")
        return str(local_path)

    # Need to download
    print(f"[zipenhancer] Model not found locally. Downloading to {ZIPENHANCER_CACHE_DIR}...")
    ZIPENHANCER_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Try ModelScope first (original source for this model)
    try:
        from modelscope import snapshot_download
        downloaded_path = snapshot_download(
            ZIPENHANCER_REPO,
            local_dir=str(local_path),
        )
        print(f"[zipenhancer] Downloaded via ModelScope: {downloaded_path}")
        return downloaded_path
    except ImportError:
        print("[zipenhancer] modelscope not installed. Run: pip install modelscope")
    except Exception as e:
        print(f"[zipenhancer] ModelScope download failed: {e}")

    # Fallback: try HuggingFace (some mirrors host it)
    try:
        from huggingface_hub import snapshot_download
        downloaded_path = snapshot_download(
            ZIPENHANCER_REPO,
            local_dir=str(local_path),
            local_files_only=False,
        )
        print(f"[zipenhancer] Downloaded via HuggingFace: {downloaded_path}")
        return downloaded_path
    except ImportError:
        print("[zipenhancer] huggingface_hub not installed.")
    except Exception as e:
        print(f"[zipenhancer] HuggingFace download failed: {e}")

    # Final fallback: check if modelscope already downloaded it to its default cache
    try:
        from modelscope.hub.utils.utils import get_cache_dir
        ms_cache = Path(get_cache_dir()) / "hub" / ZIPENHANCER_REPO.replace("/", "_")
        if ms_cache.exists() and (ms_cache / "configuration.json").exists():
            print(f"[zipenhancer] Found in ModelScope default cache, copying to {local_path}...")
            import shutil
            shutil.copytree(ms_cache, local_path, dirs_exist_ok=True)
            return str(local_path)
    except Exception:
        pass

    raise RuntimeError(
        f"Could not download ZipEnhancer model. Please manually download it:\n"
        f"  1) Install modelscope: pip install modelscope\n"
        f"  2) Download: modelscope download --model {ZIPENHANCER_REPO} --local_dir \"{local_path}\"\n"
        f"Or place the model files directly in: {local_path}\n"
        f"Model files should include: configuration.json, model.pt, etc."
    )


OUTPUT_DIR.mkdir(exist_ok=True)
VOICES_DIR.mkdir(exist_ok=True)
LORA_DIR.mkdir(exist_ok=True)
TRAIN_DATA_DIR.mkdir(exist_ok=True)

MODEL_REF = "openbmb/VoxCPM2"  # HF repo ID; actual path resolved via _ensure_voxcpm2_local()

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


def detected_vram_gib() -> Optional[float]:
    """Return physical VRAM without letting CUDA probing break WebUI startup."""
    try:
        if not torch.cuda.is_available():
            return None
        return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    except Exception as exc:
        print(f"[VRAM] CUDA memory detection failed: {exc}")
        return None


def recommend_vram_profile(total_vram_gib: Optional[float] = None) -> str:
    """Recommend a preset from physical VRAM without locking the UI choice."""
    total = detected_vram_gib() if total_vram_gib is None else total_vram_gib
    if total is None:
        return "8gb"
    try:
        return profile_for_vram_gib(total)
    except ValueError as exc:
        # There is no sub-8 GB training profile.  Keep the UI renderable, while
        # explicit manual values are rejected by _resolve_vram_profile().
        print(f"[VRAM] {exc}; showing the minimum 8 GB profile")
        return "8gb"


def _resolve_vram_profile(profile) -> tuple[str, dict]:
    """Resolve aliases/manual GiB amounts using the shared trainer contract."""
    if profile is None or (isinstance(profile, str) and not profile.strip()):
        profile = recommend_vram_profile()
    return resolve_vram_profile(profile)


def load_selected_vram_profile() -> str:
    """Use the last explicit selection, falling back to hardware detection."""
    return load_vram_preference(VRAM_PREFERENCE_PATH, recommend_vram_profile())


def persist_vram_profile(profile) -> str:
    """Persist a dropdown profile or a manually entered VRAM amount."""
    return save_vram_preference(VRAM_PREFERENCE_PATH, profile)


# Gradio 6.11 does not resolve I18nData nested inside ``(label, value)``
# dropdown choices. It renders the serialized ``__i18n__{...}`` payload
# literally. Keep the labels human-readable while preserving stable profile
# codes for the backend and public API.
VRAM_PROFILE_DROPDOWN_CHOICES = [
    ("8 GB — safe", "8gb"),
    ("12 GB — balanced", "12gb"),
    ("16 GB — balanced", "16gb"),
    (">16 GB — max speed (unrestricted)", "over_16gb"),
]


def _persist_vram_profile_from_ui(profile) -> None:
    """Best-effort persistence for the dropdown change event."""
    try:
        persist_vram_profile(profile)
    except (OSError, ValueError) as exc:
        # Selection should still work when the portable directory is read-only.
        print(f"[VRAM] Could not persist profile selection: {exc}")


def _normalize_lora_run_name(value) -> str:
    """Validate a result-folder name before any directory can be removed."""
    name = str(value or "").strip().replace(" ", "_")
    if not name:
        raise ValueError("Specify a LoRA name")
    if len(name) > 100:
        raise ValueError("LoRA name is too long (maximum 100 characters)")
    invalid = '<>:"/\\|?*'
    if name in {".", ".."} or name.endswith((".", " ")) or any(
        character in invalid or ord(character) < 32 for character in name
    ):
        raise ValueError("LoRA name must be a single safe folder name")
    if name.upper().split(".", 1)[0] in {
        "CON", "PRN", "AUX", "NUL",
        *(f"COM{index}" for index in range(1, 10)),
        *(f"LPT{index}" for index in range(1, 10)),
    }:
        raise ValueError("LoRA name is reserved by Windows")
    return name


def release_training_gpu_memory() -> str:
    """Unload WebUI inference/ASR state before the trainer claims the GPU.

    ONNX Runtime owns its CUDA allocations outside PyTorch, so dropping the
    final ASR adapter reference and running GC is as important as empty_cache().
    The inference model is lazy and will be loaded again on the next TTS call.
    """
    global _model, _asr_model, _ACTIVE_LORA
    import gc

    released = []
    if _model is not None:
        _model = None
        _ACTIVE_LORA = None
        released.append("TTS")
    if _asr_model is not None:
        _asr_model = None
        released.append("ASR")

    gc.collect()
    if not torch.cuda.is_available():
        return "Freed " + (" + ".join(released) if released else "no loaded GPU models")

    try:
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        free_gib = free_bytes / (1024 ** 3)
        total_gib = total_bytes / (1024 ** 3)
        allocated_gib = torch.cuda.memory_allocated(0) / (1024 ** 3)
        reserved_gib = torch.cuda.memory_reserved(0) / (1024 ** 3)
        what = " + ".join(released) if released else "no loaded WebUI models"
        return (
            f"Freed {what}; GPU free {free_gib:.2f}/{total_gib:.2f} GB "
            f"(PyTorch allocated {allocated_gib:.2f} GB, reserved {reserved_gib:.2f} GB)"
        )
    except Exception as exc:
        return f"Freed WebUI references; CUDA memory status unavailable: {exc}"


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


# ====================================================================
# === АВТО-ПОДГОТОВКА ДАТАСЕТА (Parakeet TDT 0.6B v3 INT8 ONNX) ====
# ====================================================================

# Ленивая загрузка Parakeet — модель (~670 MB) качается только когда юзер жмёт «Авто-подготовку»
_asr_model = None


def get_asr_model(language: Optional[str] = None):
    """Lazy-load Parakeet TDT 0.6B v3 INT8 ONNX + Silero VAD для длинного аудио.
    Модель скачается в HF cache при первом вызове (~670 MB + ~2 MB VAD).
    Автоматически использует GPU (CUDA) если доступен, с фоллбеком на CPU."""
    global _asr_model
    if _asr_model is not None:
        return _asr_model
    print("[asr] Загружаю Parakeet TDT 0.6B v3 INT8 + Silero VAD (при первом запуске ~670 MB)...")
    import onnx_asr

    # Выбираем best provider: CUDA если есть, иначе CPU
    providers = []
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            providers.append(("CUDAExecutionProvider", {"device_id": 0}))
            print("[asr] Использую CUDAExecutionProvider")
        providers.append(("CPUExecutionProvider", {}))
    except Exception as exc:
        print(f"[asr] onnxruntime check failed, используем default providers: {exc}")
        providers = None

    kwargs = {"quantization": "int8"}
    if providers is not None:
        kwargs["providers"] = providers

    # Silero VAD для разбиения длинного аудио на речевые сегменты
    # (Parakeet сам по себе держит только ~20-30 сек за проход)
    vad = onnx_asr.load_vad("silero", providers=providers if providers else None)

    _asr_model = (
        onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3", **kwargs)
        .with_vad(vad)
        .with_timestamps()
    )
    print("[asr] Модель загружена (с VAD для длинного аудио).")
    return _asr_model


def _ffmpeg_bin() -> str:
    """Return the portable ffmpeg path, supporting both shipped layouts."""
    executable = "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"
    for portable in (
        SCRIPT_DIR / "ffmpeg" / executable,
        SCRIPT_DIR / "ffmpeg" / "bin" / executable,
    ):
        if portable.exists():
            return str(portable)
    return "ffmpeg"


def _ffprobe_duration(path: str) -> float:
    """Длительность медиафайла в секундах (через ffmpeg -i parse)."""
    import subprocess, re
    try:
        res = subprocess.run(
            [_ffmpeg_bin(), "-i", path],
            capture_output=True, text=True, timeout=30,
        )
        m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", res.stderr)
        if m:
            return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    except Exception:
        pass
    return 0.0


def extract_audio_16k_mono(src_path: str, dst_path: Path) -> bool:
    """ffmpeg: любой видео/аудио → 16kHz mono WAV. Возвращает True при успехе."""
    import subprocess
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [_ffmpeg_bin(), "-y", "-i", src_path,
             "-vn", "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le",
             str(dst_path)],
            check=True, capture_output=True, timeout=3600,
        )
        return dst_path.exists() and dst_path.stat().st_size > 0
    except subprocess.CalledProcessError as exc:
        print(f"[ffmpeg] extract error: {exc.stderr.decode('utf-8', errors='replace')[:500]}")
        return False
    except Exception as exc:
        print(f"[ffmpeg] extract error: {exc}")
        return False


def _extract_words_from_result(result) -> list[dict]:
    """Распарсить результат onnx-asr .with_timestamps() в список {text, start, end}.

    Основной формат — TimestampedSegmentResult(start, end, text, timestamps, tokens, logprobs):
      - tokens — список BPE-субтокенов (строки); начало нового слова обозначается лидирующим пробелом
      - timestamps[i] — время старта tokens[i] (в секундах, абсолютные от начала поданного аудио)

    Алгоритм: склеиваем подряд идущие субтокены в слова по маркеру-пробелу.
    Возвращаем [{text, start, end}] на каждое слово.
    """
    words = []

    tokens = getattr(result, "tokens", None)
    timestamps = getattr(result, "timestamps", None)
    seg_start = float(getattr(result, "start", 0.0) or 0.0)
    seg_end = float(getattr(result, "end", 0.0) or 0.0)

    # --- Основной путь: TimestampedSegmentResult с параллельными tokens + timestamps ---
    if tokens is not None and timestamps is not None and len(tokens) == len(timestamps):
        # onnx-asr's VAD wrapper reports segment bounds in full-file time but
        # token timestamps in the sliced segment's local time.  Detect that
        # representation and put every boundary in full-file coordinates.
        raw_timestamps = [float(ts) for ts in timestamps]
        seg_duration = max(0.0, seg_end - seg_start)
        relative_timestamps = bool(raw_timestamps) and (
            min(raw_timestamps) >= -0.05
            and max(raw_timestamps) <= seg_duration + 0.25
            and seg_start > 0.0
        )
        timestamp_offset = seg_start if relative_timestamps else 0.0
        final_token_end = (
            seg_start + seg_duration if relative_timestamps else seg_end
        )
        cur = None
        for i, (tok, raw_ts) in enumerate(zip(tokens, raw_timestamps)):
            tok_s = str(tok)
            ts = raw_ts + timestamp_offset
            next_ts = (
                raw_timestamps[i + 1] + timestamp_offset
                if (i + 1) < len(raw_timestamps)
                else (final_token_end or ts)
            )
            is_new_word = tok_s.startswith(" ") or i == 0
            piece = tok_s.lstrip(" ")
            if is_new_word:
                if cur and cur["text"]:
                    words.append(cur)
                cur = {"text": piece, "start": ts, "end": float(next_ts)}
            else:
                if cur is None:
                    cur = {"text": piece, "start": ts, "end": float(next_ts)}
                else:
                    cur["text"] += piece
                    cur["end"] = float(next_ts)
        if cur and cur["text"]:
            words.append(cur)
        return words

    # --- Фоллбек: старые форматы (dict/list/tuple в tokens) ---
    if tokens is None and isinstance(result, (list, tuple)) and len(result) >= 2:
        tokens = result[1]
    if tokens is None:
        return words
    for t in tokens:
        if hasattr(t, "text"):
            text = (t.text or "").strip()
            start = float(getattr(t, "start", 0.0))
            end = float(getattr(t, "end", start))
        elif isinstance(t, dict):
            text = (t.get("text") or t.get("word") or "").strip()
            start = float(t.get("start", 0.0))
            end = float(t.get("end", start))
        elif isinstance(t, (list, tuple)) and len(t) >= 3:
            text = str(t[0]).strip()
            start = float(t[1])
            end = float(t[2])
        else:
            continue
        if text:
            words.append({"text": text, "start": start, "end": end})
    return words


def _split_asr_segment_no_drop(
    *,
    seg_start: float,
    seg_end: float,
    seg_text: str,
    words: list[dict],
    max_sec: float,
    target_max: float,
) -> tuple[list[dict], dict]:
    """Split one VAD segment without silently losing speech or transcript.

    Word timestamps are preferred.  Some ASR backends omit them; in that case
    deterministic synthetic word spans preserve the complete ASR transcript
    while still producing bounded audio windows.
    """
    source_words = list(words or [])
    if not source_words:
        tokens = [token for token in (seg_text or "").split() if token]
        if not tokens:
            return [], {
                "source_duration_sec": max(0.0, seg_end - seg_start),
                "planned_duration_sec": 0.0,
                "textless_duration_sec": 0.0,
                "part_count": 0,
            }
        span = max(0.0, seg_end - seg_start)
        source_words = []
        for index, token in enumerate(tokens):
            start = seg_start + span * index / len(tokens)
            end = seg_start + span * (index + 1) / len(tokens)
            source_words.append({"text": token, "start": start, "end": end})

    plan = plan_no_drop_splits(
        source_words,
        source_start=seg_start,
        source_end=seg_end,
        max_part_sec=max_sec,
        target_part_sec=min(target_max, max_sec),
    )
    audit = validate_split_plan(plan, allow_textless_parts=True)
    segments = [
        {
            "start": part.start,
            "end": part.end,
            "text": part.text,
            "_exact_bounds": True,
        }
        for part in plan.parts
        if part.has_text
    ]
    return segments, {
        "source_duration_sec": audit.source_duration_sec,
        "planned_duration_sec": audit.planned_duration_sec,
        "textless_duration_sec": audit.textless_duration_sec,
        "part_count": audit.part_count,
    }


# Пунктуация конца предложения (включая русские/англ/CJK)
_SENT_END_CHARS = set(".!?…。？！")
# Мягкие разделители внутри длинного предложения
_SOFT_SPLIT_CHARS = set(",;：、，；:")


def segment_by_sentences(
    words: list[dict],
    min_sec: float = 2.0,
    max_sec: float = 15.0,
    target_min: float = 4.0,
    target_max: float = 10.0,
) -> tuple[list[dict], dict]:
    """
    Сгруппировать слова в осмысленные клипы (целые предложения / абзацы).
    Возвращает (segments, stats).

    stats = {
      "raw_sentences": int,       # сколько было предложений после парсинга
      "after_merge": int,         # сколько блоков после склейки коротких
      "after_split": int,         # сколько блоков после разрезки длинных
      "kept": int,                # сколько осталось после финальной фильтрации
      "dropped_too_short": int,   # отброшено за min_sec
      "dropped_too_long": int,    # отброшено за max_sec (защита от бага split)
      "dropped_empty": int,       # отброшено без букв/цифр
      "dropped_sec": float,       # суммарная длительность отброшенного
    }

    Алгоритм:
    1) Предложения по `.` `!` `?` `…`
    2) Жадная склейка коротких: пока cur < target_min — приклеиваем следующее
       (независимо от паузы, лишь бы итог ≤ max_sec). Для маленьких видео
       это гарантирует, что ничего не теряем.
    3) Разрезка длинных (> max_sec) по запятым / паузам
    4) Фильтр: < min_sec, пустые, > max_sec
    """
    stats = {
        "raw_sentences": 0, "after_merge": 0, "after_split": 0,
        "kept": 0, "dropped_too_short": 0, "dropped_too_long": 0,
        "dropped_empty": 0, "dropped_sec": 0.0,
    }
    if not words:
        return [], stats

    # Шаг 1: группируем слова в предложения по терминаторам
    sentences: list[dict] = []
    buf: list[dict] = []
    for w in words:
        buf.append(w)
        last = w["text"][-1] if w["text"] else ""
        if last in _SENT_END_CHARS:
            sentences.append({
                "start": buf[0]["start"], "end": buf[-1]["end"],
                "text": " ".join(x["text"] for x in buf).strip(),
                "_words": buf,
            })
            buf = []
    if buf:
        sentences.append({
            "start": buf[0]["start"], "end": buf[-1]["end"],
            "text": " ".join(x["text"] for x in buf).strip(),
            "_words": buf,
        })
    stats["raw_sentences"] = len(sentences)

    # Шаг 2: жадная склейка — пока клип < target_min, приклеиваем следующий
    #         (тут НЕ смотрим на gap — важнее не потерять данные)
    merged: list[dict] = []
    i = 0
    while i < len(sentences):
        cur = dict(sentences[i])
        cur_words = list(cur.get("_words", []))
        dur = cur["end"] - cur["start"]
        while dur < target_min and (i + 1) < len(sentences):
            nxt = sentences[i + 1]
            new_dur = nxt["end"] - cur["start"]
            if new_dur > max_sec:
                break  # иначе лопнет и придётся пилить
            cur["end"] = nxt["end"]
            cur["text"] = (cur["text"] + " " + nxt["text"]).strip()
            cur_words.extend(nxt.get("_words", []))
            dur = cur["end"] - cur["start"]
            i += 1
        cur["_words"] = cur_words
        merged.append(cur)
        i += 1
    stats["after_merge"] = len(merged)

    # Шаг 3: режем слишком длинные
    final: list[dict] = []
    for s in merged:
        dur = s["end"] - s["start"]
        if dur <= max_sec:
            final.append(s)
            continue
        ws = s.get("_words", [])
        if not ws:
            final.append(s)
            continue
        final.extend(_split_long(ws, target_max=target_max, max_sec=max_sec, min_sec=min_sec))
    stats["after_split"] = len(final)

    # Шаг 4: финальная фильтрация
    out: list[dict] = []
    for s in final:
        dur = s["end"] - s["start"]
        if not s["text"] or not any(ch.isalnum() for ch in s["text"]):
            stats["dropped_empty"] += 1
            stats["dropped_sec"] += max(dur, 0)
            continue
        if dur < min_sec:
            stats["dropped_too_short"] += 1
            stats["dropped_sec"] += dur
            continue
        if dur > max_sec:
            stats["dropped_too_long"] += 1
            stats["dropped_sec"] += dur
            continue
        out.append({"start": s["start"], "end": s["end"], "text": s["text"]})
    stats["kept"] = len(out)
    return out, stats


def _make_segment(ws: list[dict]) -> dict:
    return {
        "start": ws[0]["start"],
        "end": ws[-1]["end"],
        "text": " ".join(x["text"] for x in ws).strip(),
        "_words": ws,
    }


def _find_best_cut(ws: list[dict], min_sec: float) -> int:
    """Найти индекс слова, ПОСЛЕ которого резать. Требования:
    - обе части должны иметь длительность >= min_sec (сбалансированный шов)
    - среди таких валидных швов: ближайший к середине
    - приоритет: запятая/двоеточие → большая пауза → ближайший к середине
    Возвращает -1 если валидного шва нет (блок некратен).
    """
    if len(ws) < 2:
        return -1
    total_start = ws[0]["start"]
    total_end = ws[-1]["end"]
    mid = len(ws) // 2

    # Фильтруем кандидатов: оба куска должны быть >= min_sec
    valid = []
    for i in range(len(ws) - 1):
        left_dur = ws[i]["end"] - total_start
        right_dur = total_end - ws[i + 1]["start"]
        if left_dur >= min_sec and right_dur >= min_sec:
            valid.append(i)
    if not valid:
        return -1

    # Сортируем валидных по удалению от середины
    valid.sort(key=lambda i: abs(i - (mid - 1)))

    # Проход 1: запятые/двоеточия (первый найденный = ближайший к середине)
    for i in valid:
        last = ws[i]["text"][-1] if ws[i]["text"] else ""
        if last in _SOFT_SPLIT_CHARS:
            return i

    # Проход 2: большие паузы (>0.2 сек), ближайший к середине
    for i in valid:
        gap = ws[i + 1]["start"] - ws[i]["end"]
        if gap > 0.2:
            return i

    # Проход 3: просто ближайший к середине валидный шов
    return valid[0]


def _split_long(ws: list[dict], target_max: float, max_sec: float, min_sec: float = 2.0) -> list[dict]:
    """Рекурсивно резать длинную последовательность слов на части ≤ max_sec.
    Гарантия: ни одна итоговая часть не превысит max_sec. Может вернуть куски < min_sec
    (они отфильтруются выше), но суммарно покрывает всё исходное аудио."""
    if not ws:
        return []
    total = ws[-1]["end"] - ws[0]["start"]
    if total <= max_sec:
        return [_make_segment(ws)]
    if len(ws) <= 1:
        return []  # одно «слово» длиннее max_sec — аномалия ASR, пропускаем

    # Приоритет 1: сбалансированный шов (обе части >= min_sec) по запятой
    cut = _find_best_cut(ws, min_sec=min_sec)

    # Приоритет 2: если балансированного нет — берём шов, ПОСЛЕ которого
    # левая часть ближе всего к target_max (но не превышает max_sec)
    if cut < 0:
        start_t = ws[0]["start"]
        best_i = -1
        best_score = float("inf")
        for i in range(len(ws) - 1):
            left_dur = ws[i]["end"] - start_t
            if left_dur > max_sec:
                break  # дальше только больше
            # score: насколько близко левая часть к target_max (но не выше max_sec)
            score = abs(target_max - left_dur)
            if score < best_score:
                best_score = score
                best_i = i
        cut = best_i if best_i >= 0 else len(ws) // 2 - 1

    if cut < 0 or cut >= len(ws) - 1:
        cut = len(ws) // 2 - 1

    left = ws[: cut + 1]
    right = ws[cut + 1 :]

    out: list[dict] = []
    out.extend(_split_long(left, target_max=target_max, max_sec=max_sec, min_sec=min_sec))
    out.extend(_split_long(right, target_max=target_max, max_sec=max_sec, min_sec=min_sec))
    return out


def _pick_grad_accum(n_clips: int) -> int:
    """Общий выбор grad_accum по размеру датасета (используется и в train_lora, и в recommender)."""
    if n_clips < 200:
        return 4
    if n_clips < 500:
        return 8
    return 16


def recommend_lora_settings(n_clips: int, total_speech_sec: float) -> tuple[int, int, int, float, str]:
    """
    Рекомендации по настройкам LoRA. Шаги считаются через target_epochs × n_clips / eff_batch,
    чтобы цикл обучения давал реалистичное число эпох (10-20 для малых датасетов),
    а не 60+ как при слепом копировании YAML-дефолта.

    База (HF карточка VoxCPM2 + YAML voxcpm_finetune_lora.yaml + FAQ):
    - r/alpha: 32 (официальный дефолт). Для >60 мин — 64.
    - lr: 1e-4 (5e-5 для 60+ мин).
    - eff_batch = 1 × grad_accum (grad_accum из `_pick_grad_accum`).

    target_epochs по минутам речи (эмпирика для speaker LoRA):
    - < 2 мин   → 25
    - 2-5 мин   → 20
    - 5-10 мин  → 15  (сладкая зона)
    - 10-20 мин → 12
    - 20-60 мин → 8
    - 60-120    → 5
    - 120+ мин  → 3
    """
    minutes = total_speech_sec / 60.0
    eff_batch = 1 * _pick_grad_accum(n_clips)  # batch_size=1 хардкод в train_lora

    if minutes < 2:
        target_epochs = 25; r, a, lr = 32, 32, 1e-4
        note = f"⚠ very little speech ({minutes:.1f} min), below OpenBMB's 5-minute minimum"
    elif minutes < 5:
        target_epochs = 20; r, a, lr = 32, 32, 1e-4
        note = f"minimum-size LoRA dataset ({minutes:.1f} min)"
    elif minutes < 10:
        target_epochs = 15; r, a, lr = 32, 32, 1e-4
        note = f"standard speaker-cloning dataset ({minutes:.1f} min)"
    elif minutes < 20:
        target_epochs = 12; r, a, lr = 32, 32, 1e-4
        note = f"medium dataset ({minutes:.1f} min)"
    elif minutes < 60:
        target_epochs = 8; r, a, lr = 32, 32, 1e-4
        note = f"large dataset ({minutes:.1f} min)"
    elif minutes < 120:
        target_epochs = 5; r, a, lr = 64, 64, 5e-5
        note = f"very large dataset ({minutes:.1f} min), using r=64 and a lower learning rate"
    else:
        target_epochs = 3; r, a, lr = 64, 64, 5e-5
        note = f"style/language adaptation dataset ({minutes:.1f} min), using r=64"

    # steps = epochs × n_clips / effective_batch, округляем к кратному 50, минимум 100
    steps = max(100, int(round(target_epochs * n_clips / max(eff_batch, 1) / 50.0)) * 50)
    reason = (
        f"{note} → target {target_epochs} epochs × {n_clips} clips / "
        f"effective batch={eff_batch} = {steps} steps"
    )
    return r, a, steps, lr, reason


def auto_prepare_dataset(
    name: str,
    input_file: str,
    min_sec: float,
    max_sec: float,
    target_min: float,
    target_max: float,
    start_training: bool,
    auto_tune: bool,
    r: int, alpha: int, steps: int, lr: float,
    local_media_path: str = "",
    progress=gr.Progress(),
):
    """
    Генератор: обрабатывает видео/аудио → клипы + транскрипты → (опц.) старт тренировки LoRA.
    Yields (log_text, transcripts_text, files_gallery_update) для UI.
    """
    import shutil, json, subprocess
    log = []

    def emit(msg: str):
        log.append(msg)
        print(f"[auto] {msg}")

    # Validate before creating or clearing any dataset directory.
    try:
        name = _normalize_lora_run_name(name)
    except ValueError as exc:
        emit(f"❌ {exc}")
        yield "\n".join(log), "", None, gr.update(), gr.update(), gr.update(), gr.update()
        return

    if input_file:
        src_path = Path(input_file)
        source_kind = "uploaded file"
    elif local_media_path and str(local_media_path).strip():
        try:
            src_path = resolve_single_media_file(
                local_media_path,
                base_dir=SCRIPT_DIR,
            )
        except LocalMediaError as exc:
            emit(f"❌ {exc}")
            yield "\n".join(log), "", None, gr.update(), gr.update(), gr.update(), gr.update()
            return
        source_kind = "local path"
    else:
        emit("❌ Upload a video/audio file or paste a local media path")
        yield "\n".join(log), "", None, gr.update(), gr.update(), gr.update(), gr.update()
        return

    if not src_path.exists():
        emit(f"❌ File not found: {src_path}")
        yield "\n".join(log), "", None, gr.update(), gr.update(), gr.update(), gr.update()
        return

    progress(0.02, desc="Checking input file...")
    duration = _ffprobe_duration(str(src_path))
    emit(f"▶ Input ({source_kind}): {src_path.name} ({duration:.1f} sec)")
    if duration > 0 and duration < min_sec * 3:
        emit(
            f"⚠ The file is very short ({duration:.1f} sec). "
            "Use at least 1–2 minutes for a useful dataset."
        )

    ds_dir = TRAIN_DATA_DIR / name
    audio_dir = ds_dir / "audio"
    if audio_dir.exists():
        # Чистим старые клипы (но сохраняем source_info)
        for f in audio_dir.glob("*.wav"):
            try: f.unlink()
            except: pass
    audio_dir.mkdir(parents=True, exist_ok=True)
    yield "\n".join(log), "", None, gr.update(), gr.update(), gr.update(), gr.update()

    # --- 1. ffmpeg → 16kHz mono ---
    progress(0.05, desc="Extracting audio with ffmpeg...")
    emit("▶ Converting audio to 16 kHz mono WAV...")
    full_wav = ds_dir / "_full_16k.wav"
    if not extract_audio_16k_mono(str(src_path), full_wav):
        emit("❌ ffmpeg could not extract the audio. Check the file format.")
        yield "\n".join(log), "", None, gr.update(), gr.update(), gr.update(), gr.update()
        return
    emit(f"✓ Created {full_wav.name}")
    yield "\n".join(log), "", None, gr.update(), gr.update(), gr.update(), gr.update()

    # --- 2. Parakeet ASR с таймстампами ---
    progress(0.15, desc="Loading the ASR model (first run: ~670 MB)...")
    emit("▶ Loading Parakeet TDT 0.6B v3 INT8 (first run downloads ~670 MB)...")
    yield "\n".join(log), "", None, gr.update(), gr.update(), gr.update(), gr.update()
    try:
        asr = get_asr_model()
    except Exception as exc:
        emit(f"❌ Could not load the ASR model: {exc}")
        yield "\n".join(log), "", None, gr.update(), gr.update(), gr.update(), gr.update()
        return

    progress(0.30, desc="Transcribing with Parakeet + VAD...")
    emit("▶ Running Silero VAD and Parakeet transcription...")
    yield "\n".join(log), "", None, gr.update(), gr.update(), gr.update(), gr.update()
    try:
        result_iter = asr.recognize(str(full_wav))
    except Exception as exc:
        emit(f"❌ Transcription failed: {exc}")
        yield "\n".join(log), "", None, gr.update(), gr.update(), gr.update(), gr.update()
        return

    # VAD-driven: один VAD-сегмент = один клип (если > max_sec, режем по внутренним словам)
    segments = []
    vad_seg_count = 0
    try:
        for seg in result_iter:
            vad_seg_count += 1
            seg_start = float(getattr(seg, "start", 0.0) or 0.0)
            seg_end = float(getattr(seg, "end", seg_start))
            seg_text = (getattr(seg, "text", "") or "").strip()
            seg_dur = seg_end - seg_start
            if not seg_text or not any(c.isalnum() for c in seg_text):
                continue
            # Убираем мусорные токены <unk>
            seg_text = seg_text.replace("<unk>", "").replace("  ", " ").strip()
            if not seg_text:
                continue

            if seg_dur <= max_sec:
                # VAD-сегмент целиком — это 1 клип
                segments.append({"start": seg_start, "end": seg_end, "text": seg_text})
            else:
                # onnx-asr VAD token timestamps are segment-relative. Convert
                # them to full-file time before building a contiguous no-drop
                # split plan. If timestamps are unavailable, the helper uses
                # deterministic synthetic spans for every transcript token.
                try:
                    timed_words = list(extract_absolute_words_from_timestamped_segment(seg))
                except Exception:
                    timed_words = []
                sub_segs, split_audit = _split_asr_segment_no_drop(
                    seg_start=seg_start,
                    seg_end=seg_end,
                    seg_text=seg_text,
                    words=timed_words,
                    max_sec=max_sec,
                    target_max=target_max,
                )
                if split_audit.get("textless_duration_sec", 0.0) > 0:
                    emit(
                        f"  ℹ {split_audit['textless_duration_sec']:.2f} sec silence "
                        "was accounted for outside text-conditioned clips"
                    )
                segments.extend(sub_segs)
    except Exception as exc:
        emit(f"❌ Could not parse the transcription result: {exc}")
        import traceback; traceback.print_exc()
        yield "\n".join(log), "", None, gr.update(), gr.update(), gr.update(), gr.update()
        return

    if not segments:
        emit("❌ No valid speech segments were produced")
        yield "\n".join(log), "", None, gr.update(), gr.update(), gr.update(), gr.update()
        return

    seg_total = sum(s['end'] - s['start'] for s in segments)
    vad_coverage_pct = seg_total / max(duration, 1) * 100
    emit(
        f"✓ {len(segments)} clips, {seg_total:.1f} sec of speech "
        f"({vad_coverage_pct:.0f}% of the input)"
    )

    # --- 3. Final validation (short speech clips are retained) ---
    progress(0.70, desc="Validating clips...")
    short_count = sum(
        1 for segment in segments
        if 0 < (segment["end"] - segment["start"]) < min_sec
    )
    if short_count:
        emit(
            f"  ℹ Retaining {short_count} speech clips shorter than {min_sec:.1f} sec "
            "so no recognized speech is silently discarded"
        )
    yield "\n".join(log), "", None, gr.update(), gr.update(), gr.update(), gr.update()

    # --- 4. Резка WAV + сохранение ---
    progress(0.80, desc="Saving clips...")
    emit("▶ Saving clips and transcripts...")
    wav, sr = sf.read(str(full_wav))
    if wav.ndim > 1:
        wav = wav[:, 0]

    transcripts_lines = []
    saved_paths = []
    saved_durations = []
    skipped = 0
    # Паддинг по краям — 100 мс с каждой стороны чтобы не обрезать хвост последней фонемы
    # и чуть-чуть «продышать» клип (VoxCPM лучше тренируется на клипах с естественными краями)
    PAD_SEC = 0.10
    for idx, seg in enumerate(segments):
        edge_padding = 0.0 if seg.get("_exact_bounds") else PAD_SEC
        s_sample = max(0, int((seg["start"] - edge_padding) * sr))
        e_sample = min(len(wav), int((seg["end"] + edge_padding) * sr))
        clip = wav[s_sample:e_sample]
        dur = len(clip) / sr if sr else 0
        if dur <= 0:
            skipped += 1
            continue
        fname = f"clip_{idx:04d}.wav"
        out_path = audio_dir / fname
        sf.write(str(out_path), clip, sr)
        transcripts_lines.append(f"{fname}|{seg['text']}")
        saved_paths.append(str(out_path))
        saved_durations.append(dur)

    if not saved_paths:
        emit("❌ No non-empty clips remained after slicing")
        yield "\n".join(log), "", None, gr.update(), gr.update(), gr.update(), gr.update()
        return

    transcripts_text = "\n".join(transcripts_lines)
    (ds_dir / "transcripts.txt").write_text(transcripts_text, encoding="utf-8")

    # Чистая речь = сумма длительностей сохранённых клипов (после сегментации и фильтрации)
    total_dur = sum(saved_durations)
    emit(
        f"✓ Saved {len(saved_paths)} clips, {total_dur:.1f} sec of speech "
        f"({total_dur/60:.1f} min)"
    )
    if skipped:
        emit(f"  Skipped {skipped} empty clips after trimming")

    # Удаляем промежуточный full wav
    try: full_wav.unlink()
    except: pass

    # Сохраняем source_info
    try:
        (ds_dir / "source_info.json").write_text(json.dumps({
            "source": str(src_path),
            "source_duration_sec": duration,
            "clips": len(saved_paths),
            "total_speech_sec": total_dur,
            "skipped_short": skipped,
            "params": {
                "min_sec": min_sec, "max_sec": max_sec,
                "target_min": target_min, "target_max": target_max,
            },
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

    progress(0.95, desc="Done")
    emit(f"\n✅ Dataset ready: train_data/{name}/")
    emit(f"   Transcripts: train_data/{name}/transcripts.txt")
    emit(f"   Audio: train_data/{name}/audio/*.wav")

    # --- Авто-тюнинг настроек LoRA на основе размера датасета ---
    if auto_tune:
        ar, aa, asteps, alr, reason = recommend_lora_settings(len(saved_paths), total_dur)
        emit("\n📊 Auto-tuned LoRA settings:")
        emit(f"   {reason}")
        emit(f"   → r={ar}, α={aa}, steps={asteps}, lr={alr}")
        emit("   Clear Auto-tune to use the slider values instead.")
        r, alpha, steps, lr = ar, aa, asteps, alr
    else:
        emit(
            f"\nℹ Auto-tune is off; using slider values: "
            f"r={r}, α={alpha}, steps={steps}, lr={lr}"
        )

    if start_training:
        emit("\n🎓 Automatic training is enabled and will start next; see the training log below.")
    else:
        emit("\nℹ The generated files and transcripts are ready for manual training below.")
    yield (
        "\n".join(log), transcripts_text, saved_paths,
        gr.update(value=r), gr.update(value=alpha),
        gr.update(value=steps), gr.update(value=lr),
    )


def maybe_auto_train(
    start_training: bool,
    name: str, files, transcripts: str,
    r: int, alpha: int, steps: int, lr: float,
    vram_profile: str = "8gb",
    progress=gr.Progress(),
):
    """Отдельный процесс тренировки. Запускается через .then() после auto_prepare_dataset.
    Если галочка «запустить после подготовки» не стояла — мгновенно выходит."""
    if not start_training:
        return
    for line in train_lora(
        name, files, transcripts, r, alpha, steps, lr,
        vram_profile=vram_profile, progress=progress,
    ):
        yield line


def prepare_train_data(
    name: str,
    files: list,
    transcripts_text: str,
    max_audio_seconds: float = 0.0,
) -> tuple[Path, int]:
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
        tx = tr_map.get(src.name) or tr_map.get(src.stem) or ""
        if not tx:
            continue  # без транскрипта пропускаем

        source_duration = _ffprobe_duration(str(src))
        # Auto-prepared standalone clips may include 100 ms context on each
        # edge. Treat that intentional padding as within the profile limit.
        if not max_audio_seconds or source_duration <= max_audio_seconds + 0.5:
            dst = audio_dir / src.name
            if not dst.exists():
                shutil.copy2(str(src), str(dst))
            item = {"audio": str(dst), "text": tx}
            if source_duration > 0:
                item["duration"] = source_duration
            manifest.append(item)
            continue

        # Oversized manual files are aligned once with the existing ASR, then
        # sliced on exact shared boundaries. The user's transcript remains
        # authoritative and every whitespace token is assigned exactly once.
        safe_stem = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in src.stem
        ) or "manual"
        converted = audio_dir / f"_{safe_stem}_source_16k.wav"
        if not extract_audio_16k_mono(str(src), converted):
            raise RuntimeError(f"Could not convert oversized training file: {src}")

        converted_info = sf.info(str(converted))
        exact_duration = converted_info.frames / float(converted_info.samplerate)
        timed_words = []
        try:
            for asr_segment in get_asr_model().recognize(str(converted)):
                try:
                    timed_words.extend(
                        extract_absolute_words_from_timestamped_segment(asr_segment)
                    )
                except Exception:
                    continue
        except Exception as exc:
            print(f"[manual split] ASR alignment fallback for {src.name}: {exc}")

        if not timed_words:
            source_tokens = [token for token in tx.split() if token]
            if not source_tokens:
                raise RuntimeError(f"Transcript is empty for oversized file: {src}")
            timed_words = []
            for index, token in enumerate(source_tokens):
                timed_words.append({
                    "text": token,
                    "start": exact_duration * index / len(source_tokens),
                    "end": exact_duration * (index + 1) / len(source_tokens),
                })

        plan = plan_no_drop_splits(
            timed_words,
            source_start=0.0,
            source_end=exact_duration,
            max_part_sec=max_audio_seconds,
            target_part_sec=max_audio_seconds,
        )
        validate_split_plan(plan, allow_textless_parts=True)
        weights = [float(len(part.fragments)) for part in plan.parts]
        child_transcripts = partition_transcript(tx, weights)
        slices = slice_audio_file(
            converted,
            audio_dir,
            plan,
            prefix=f"{safe_stem}_part",
        )
        for sliced, child_text in zip(slices, child_transcripts):
            if not child_text:
                continue  # accounted silence-only interval, not speech data
            manifest.append({
                "audio": str(sliced.path),
                "text": child_text,
                "duration": sliced.frames / float(sliced.sample_rate),
                "source_audio": str(src),
            })
        try:
            converted.unlink()
        except OSError:
            pass

    manifest_path = ds_dir / "train.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for it in manifest:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    return manifest_path, len(manifest)


def build_lora_training_config(
    *,
    pretrained,
    manifest,
    save_path,
    sample_count,
    r,
    alpha,
    steps,
    lr,
    vram_profile,
) -> dict:
    """Build the complete reproducible trainer config without side effects."""
    profile_name, profile = _resolve_vram_profile(vram_profile)
    rank, alpha_value, step_count, learning_rate = validate_lora_hyperparameters(
        r, alpha, steps, lr
    )
    try:
        sample_count = int(sample_count)
    except (TypeError, ValueError) as exc:
        raise ValueError("Training sample count must be a positive integer") from exc
    if sample_count < 1:
        raise ValueError("Training sample count must be a positive integer")

    # Preserve the recommender's effective batch when the fast profile doubles
    # the micro-batch, so profile selection changes speed rather than epochs.
    grad_accum = max(
        1,
        _pick_grad_accum(sample_count) // int(profile["batch_size"]),
    )
    return {
        "pretrained_path": str(pretrained),
        "train_manifest": str(manifest),
        "sample_rate": 16000,
        "out_sample_rate": 48000,
        "batch_size": profile["batch_size"],
        "grad_accum_steps": grad_accum,
        "num_workers": 0,
        "num_iters": step_count,
        "log_interval": 10,
        "valid_interval": max(100, step_count // 2),
        "save_interval": max(100, step_count // 2),
        "learning_rate": learning_rate,
        "weight_decay": 0.01,
        "warmup_steps": min(100, step_count // 10),
        "max_steps": step_count,
        "max_batch_tokens": profile["max_batch_tokens"],
        "vram_profile": profile_name,
        "gradient_checkpointing": profile["gradient_checkpointing"],
        "max_sample_tokens": profile["max_sample_tokens"],
        "split_oversized_samples": profile["split_oversized_samples"],
        "vram_safety_margin_mb": profile["vram_safety_margin_mb"],
        "save_path": str(save_path),
        "lambdas": {"loss/diff": 1.0, "loss/stop": 1.0},
        "lora": {
            "enable_lm": True,
            "enable_dit": True,
            "enable_proj": False,
            "r": rank,
            "alpha": alpha_value,
            "dropout": 0.0,
        },
    }


def train_lora(
    name, files, transcripts, r, alpha, steps, lr,
    vram_profile: str = "8gb", local_audio_path: str = "",
    progress=gr.Progress(),
):
    """Запустить обучение LoRA. Yield-генератор для live log."""
    import subprocess, yaml
    try:
        name = _normalize_lora_run_name(name)
        profile_name, profile = _resolve_vram_profile(vram_profile)
        r, alpha, steps, lr = validate_lora_hyperparameters(r, alpha, steps, lr)
    except ValueError as exc:
        yield f"❌ {exc}"
        return

    if not files and local_audio_path and str(local_audio_path).strip():
        try:
            files = [
                str(path)
                for path in resolve_audio_files(local_audio_path, base_dir=SCRIPT_DIR)
            ]
            if not str(transcripts or "").strip():
                transcripts = read_transcripts(
                    local_audio_path,
                    base_dir=SCRIPT_DIR,
                ) or ""
        except LocalMediaError as exc:
            yield f"❌ {exc}"
            return
        yield f"✓ Local path resolved to {len(files)} audio file(s)"

    if not files:
        yield "❌ Upload audio files or paste a local audio file/folder path"
        return

    train_script = get_training_script()
    if train_script is None:
        yield "❌ training/scripts/train_voxcpm_finetune.py was not found"
        return

    try:
        persist_vram_profile(profile_name)
    except OSError as exc:
        # A read-only cache must not prevent an otherwise valid local run.
        print(f"[VRAM] Could not persist profile selection: {exc}")

    progress(0.1, desc="Preparing the dataset...")
    try:
        manifest, n = prepare_train_data(
            name,
            files,
            transcripts,
            max_audio_seconds=profile["max_audio_seconds"],
        )
    except Exception as exc:
        yield f"❌ Could not prepare the dataset: {exc}"
        return
    if n == 0:
        yield (
            "❌ No valid samples. Check transcripts.txt or the transcript box "
            "(format: filename.wav|text)"
        )
        return
    yield f"✓ Dataset: {n} samples → {manifest}"

    save_path = LORA_DIR / name
    config_path = TRAIN_DATA_DIR / name / "train_config.yaml"

    # Путь к уже скачанному VoxCPM2 (в models/ через HF cache)
    # Ищем snapshot
    from huggingface_hub import snapshot_download
    try:
        try:
            pretrained = snapshot_download("openbmb/VoxCPM2", local_files_only=True)
        except Exception:
            pretrained = snapshot_download("openbmb/VoxCPM2")
    except Exception as exc:
        yield f"❌ Could not find or download the VoxCPM2 base model: {exc}"
        return

    cfg = build_lora_training_config(
        pretrained=pretrained,
        manifest=manifest,
        save_path=save_path,
        sample_count=n,
        r=r,
        alpha=alpha,
        steps=steps,
        lr=lr,
        vram_profile=profile_name,
    )

    # If a new rank/alpha was selected, resuming an older checkpoint would
    # produce a shape mismatch.  Abort on cleanup failure instead of silently
    # continuing with incompatible state.
    if save_path.exists():
        import shutil
        try:
            shutil.rmtree(save_path)
            yield f"🧹 Removed the previous checkpoint at {save_path}"
        except Exception as exc:
            yield f"❌ Could not remove the previous checkpoint at {save_path}: {exc}"
            return
    save_path.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)
    yield f"✓ Config: {config_path}"
    yield (
        f"✓ VRAM profile: {profile_name} | batch={profile['batch_size']}, "
        f"sample cap={profile['max_sample_tokens']} tokens, "
        f"checkpointing={'on' if profile['gradient_checkpointing'] else 'off'}, "
        f"safety margin={profile['vram_safety_margin_mb']} MB"
    )
    yield f"Starting training (steps={steps}, r={r}, α={alpha}, lr={lr})..."

    progress(0.15, desc="Starting training...")

    # The trainer is a separate CUDA process.  Release both PyTorch inference
    # state and ONNX Runtime ASR state first, otherwise their allocations remain
    # visible to the child and can consume most of an 8 GB card.
    memory_status = release_training_gpu_memory()
    yield f"🧹 {memory_status}"

    # PYTHONUNBUFFERED=1 + -u = принудительный flush после каждой строки,
    # иначе Python subprocess буферит stdout при пайпе и UI не видит прогресс
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    alloc_conf = env.get("PYTORCH_CUDA_ALLOC_CONF", "")
    if "expandable_segments:" not in alloc_conf:
        env["PYTORCH_CUDA_ALLOC_CONF"] = ",".join(
            part for part in (alloc_conf, "expandable_segments:True") if part
        )
    proc = subprocess.Popen(
        build_training_command(sys.executable, train_script, config_path),
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
        progress(1.0, desc="Done")
        log_lines.append(f"\n✅ Done! LoRA saved to {save_path}")
    else:
        log_lines.append(f"\n❌ Training exited with code {proc.returncode}")
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
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        return "cuda", f"{name} | VRAM: {vram:.1f} GB"
    return "cpu", "CPU (экспериментально / experimental — very slow)"

DEVICE, DEVICE_INFO = _detect_device()
print(f"[VoxCPM2] Device: {DEVICE_INFO}")

# === Ленивая загрузка модели ===
_model = None

def get_model(lora_weights_path: Optional[str] = None, force_reload: bool = False) -> VoxCPM:
    """Загружает модель. При первом lora_weights_path — модель перезагружается с LoRA-структурой
    (нужно для load_lora hot-swap в дальнейшем). Читает lora_config.json из папки чекпоинта."""
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

    # Resolve local VoxCPM2 path
    try:
        voxcpm2_local = _ensure_voxcpm2_local()
        model_path = voxcpm2_local
        print(f"[voxcpm2] Using local path: {voxcpm2_local}")
    except Exception as e:
        print(f"[voxcpm2] WARNING: Could not setup local VoxCPM2: {e}")
        print("[voxcpm2] Falling back to online HuggingFace ID (requires internet)...")
        model_path = MODEL_REF

    kwargs = dict(load_denoiser=True, optimize=False)
    # Use local/offline ZipEnhancer model
    try:
        zipenhancer_local = _ensure_zipenhancer_local()
        kwargs["zipenhancer_model_id"] = zipenhancer_local
        print(f"[zipenhancer] Using local path: {zipenhancer_local}")
    except Exception as e:
        print(f"[zipenhancer] WARNING: Could not setup local ZipEnhancer: {e}")
        print("[zipenhancer] Falling back to online mode (may require internet)...")
    if lora_weights_path:
        kwargs["lora_weights_path"] = lora_weights_path
        # Явно читаем lora_config.json и конструируем LoRAConfig — иначе voxcpm использует default (r=8)
        import json
        cfg_path = Path(lora_weights_path) / "lora_config.json"
        if cfg_path.exists():
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg_data = json.load(f).get("lora_config", {})
                from voxcpm.model.voxcpm2 import LoRAConfig as LoRAConfigV2
                kwargs["lora_config"] = LoRAConfigV2(**cfg_data)
                print(f"[lora] loaded lora_config.json: r={cfg_data.get('r')}, alpha={cfg_data.get('alpha')}")
            except Exception as exc:
                print(f"[lora] WARNING: failed to read lora_config.json: {exc}")
    _model = VoxCPM.from_pretrained(model_path, **kwargs)
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


def _generate_audio_stream(model, kwargs: dict, streaming: bool, progress):
    """Generator, yield'ит (sample_rate, ndarray) чанки для gr.Audio(streaming=True).
    streaming=False — один yield целого аудио в конце (для совместимости).
    streaming=True — yield каждого чанка по мере генерации (live-playback)."""
    sr = model.tts_model.sample_rate
    if not streaming:
        progress(0, desc="Генерация...")
        wav = _collect_audio(model.generate(**kwargs))
        yield sr, wav
        return
    # streaming с пре-буферизацией: первый yield — 2.5 сек (pre-buffer чтобы
    # плеер не запинался на старте), дальше — по 1 сек для progressive playback
    gen = model.generate_streaming(**kwargs)
    buf = []
    buf_samples = 0
    prebuffer_samples = int(sr * 8.0)   # 8 сек initial buffer (щедрый head-start)
    chunk_samples = int(sr * 2.0)        # 2 сек на каждый последующий yield
    any_yielded = False
    for i, chunk in enumerate(gen):
        arr = np.asarray(chunk)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        buf.append(arr)
        buf_samples += len(arr)
        try:
            progress(min(0.99, i / 120), desc=f"Chunk {i+1}")
        except Exception:
            pass
        threshold = prebuffer_samples if not any_yielded else chunk_samples
        if buf_samples >= threshold:
            yield sr, np.concatenate(buf)
            any_yielded = True
            buf = []
            buf_samples = 0
    # хвост — всё что осталось в буфере
    if buf:
        yield sr, np.concatenate(buf)
        any_yielded = True
    if not any_yielded:
        raise gr.Error("Модель не вернула аудио / Model returned no audio.")


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
        # Сброс audio-компонента перед новым стримом — иначе autoplay
        # не триггерится при повторной генерации
        yield None, used_seed
        chunks_accum = []
        sr = model.tts_model.sample_rate
        for sr_i, chunk in _generate_audio_stream(model, kwargs, bool(streaming), progress):
            sr = sr_i
            chunks_accum.append(chunk)
            yield (sr_i, chunk), used_seed
        # сохраняем архив на диск в output/
        if chunks_accum:
            _save_wav(np.concatenate(chunks_accum), sr, "tts", fmt)
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
        yield None, used_seed  # сброс audio-компонента для повторного autoplay
        chunks_accum = []
        sr = model.tts_model.sample_rate
        for sr_i, chunk in _generate_audio_stream(model, kwargs, bool(streaming), progress):
            sr = sr_i
            chunks_accum.append(chunk)
            yield (sr_i, chunk), used_seed
        if chunks_accum:
            _save_wav(np.concatenate(chunks_accum), sr, "design", fmt)
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
        yield None, used_seed  # сброс audio-компонента для повторного autoplay
        chunks_accum = []
        sr = model.tts_model.sample_rate
        for sr_i, chunk in _generate_audio_stream(model, kwargs, bool(streaming), progress):
            sr = sr_i
            chunks_accum.append(chunk)
            yield (sr_i, chunk), used_seed
        if chunks_accum:
            _save_wav(np.concatenate(chunks_accum), sr, "clone", fmt)
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
.brand-credits a {
  color: #fbbf24 !important;
  text-decoration: none !important;
  font-weight: 600 !important;
  padding: 0 !important;
  margin: 0 !important;
  background: none !important;
  display: inline !important;
}
.brand-credits a:hover { text-decoration: underline !important; }
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
  display: inline-flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 2px;
  line-height: 1;
  white-space: nowrap;
  min-width: 44px;
}
.lang-btn:hover { background: rgba(255,255,255,0.3); }
.lang-btn img { margin: 0 !important; vertical-align: middle !important; }
.brand-header { position: relative; }

button.primary {
  background: linear-gradient(135deg, #6d28d9 0%, #7e22ce 100%) !important;
  color: white !important;
  font-weight: 600 !important;
  border-radius: 10px !important;
}

.lora-refresh-btn {
  max-width: 44px !important;
  flex: 0 0 44px !important;
}
.lora-refresh-btn button {
  padding: 0 !important;
  min-width: 0 !important;
  width: 44px !important;
  height: 100% !important;
}
/* === Donate dropdown next to lang switcher === */
.donate-wrap {
  position: relative;
  display: inline-block;
}
.donate-wrap > summary.donate-btn {
  list-style: none;
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  gap: 4px;
  user-select: none;
}
.donate-wrap > summary.donate-btn::-webkit-details-marker { display: none; }
.donate-wrap:not([open]) .donate-popover { display: none !important; pointer-events: none; }
.donate-popover {
  position: absolute;
  top: calc(100% + 6px);
  right: 0;
  background: rgba(20, 20, 28, 0.98);
  backdrop-filter: blur(8px);
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 10px;
  padding: 10px 14px;
  min-width: 320px;
  box-shadow: 0 8px 24px rgba(0,0,0,0.4);
  z-index: 999;
  font-size: 13px;
  line-height: 1.5;
}
.donate-popover a {
  color: #c4a3ff !important;
  text-decoration: none !important;
  font-weight: 600 !important;
  padding: 0 !important;
  margin: 0 !important;
  background: none !important;
  display: inline !important;
}
.donate-popover a:hover { text-decoration: underline !important; }
.donate-intro a { color: #c4a3ff !important; }
.donate-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 10px;
  padding: 4px 0;
}
.donate-row > span {
  color: #9ca3af;
  font-weight: 600;
  font-size: 12px;
  flex: 0 0 80px;
  text-align: left;
  white-space: nowrap;
}
.donate-row > code {
  flex: 1;
  text-align: left;
}
.donate-row > code {
  background: rgba(255,255,255,0.06);
  padding: 2px 6px;
  border-radius: 4px;
  font-size: 11px;
  color: #e5e7eb;
  user-select: all;
}
.donate-intro {
  color: #cbd5e1;
  font-size: 12px;
  line-height: 1.5;
  margin: 0 0 4px 0;
}
.donate-sep {
  height: 1px;
  background: rgba(255,255,255,0.1);
  margin: 6px 0;
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

_DONATE_POPOVER_RU = """
<div class="donate-popover">
  <p class="donate-intro">Привет! Я Илья (<a href="https://t.me/nerual_dreming" target="_blank">Nerual Dreming</a>), я создаю AI-инструменты, которые работают локально — бесплатно, без облака, без подписок. Ваш донат позволяет фокусироваться на исследовании и создании новых открытых проектов, а не на выживании. Спасибо!</p>
  <div class="donate-sep"></div>
  <div class="donate-row"><a href="https://dalink.to/nerual_dreming" target="_blank">💳 Карта / PayPal (рубли, доллары, евро)</a></div>
  <div class="donate-row"><a href="https://boosty.to/neuro_art" target="_blank">🚀 Ежемесячная подписка на Boosty</a></div>
  <div class="donate-sep"></div>
  <div class="donate-row"><span>BTC</span><code>1E7dHL22RpyhJGVpcvKdbyZgksSYkYeEBC</code></div>
  <div class="donate-row"><span>ETH</span><code>0xb5db65adf478983186d4897ba92fe2c25c594a0c</code></div>
  <div class="donate-row"><span>USDT TRC20</span><code>TQST9Lp2TjK6FiVkn4fwfGUee7NmkxEE7C</code></div>
</div>
"""

_DONATE_POPOVER_EN = """
<div class="donate-popover">
  <p class="donate-intro">Hi! I'm Ilya (<a href="https://t.me/nerual_dreming" target="_blank">Nerual Dreming</a>), I build AI tools that anyone can run locally — for free, without cloud, without subscriptions. Your donation lets me focus on research and building new open-source projects instead of surviving. Thank you!</p>
  <div class="donate-sep"></div>
  <div class="donate-row"><a href="https://dalink.to/nerual_dreming" target="_blank">💳 Card / PayPal (USD, EUR, RUB)</a></div>
  <div class="donate-row"><a href="https://boosty.to/neuro_art" target="_blank">🚀 Monthly subscription on Boosty</a></div>
  <div class="donate-sep"></div>
  <div class="donate-row"><span>BTC</span><code>1E7dHL22RpyhJGVpcvKdbyZgksSYkYeEBC</code></div>
  <div class="donate-row"><span>ETH</span><code>0xb5db65adf478983186d4897ba92fe2c25c594a0c</code></div>
  <div class="donate-row"><span>USDT TRC20</span><code>TQST9Lp2TjK6FiVkn4fwfGUee7NmkxEE7C</code></div>
</div>
"""


_CREDITS_HTML_RU = (
    'Собрал <a href="https://t.me/nerual_dreming" target="_blank">Nerual Dreming</a> — '
    'основатель <a href="https://artgeneration.me" target="_blank">ArtGeneration.me</a>, '
    'техноблогер и нейро-евангелист. '
    'Канал <a href="https://t.me/neuroport" target="_blank">Нейро-Софт</a> — '
    'репаки и портативки полезных нейросетей.'
)
_CREDITS_HTML_EN = (
    'Built by <a href="https://t.me/nerual_dreming" target="_blank">Nerual Dreming</a> — '
    'founder of <a href="https://artgeneration.me" target="_blank">ArtGeneration.me</a>, '
    'tech-blogger and neuro-evangelist. '
    'Channel <a href="https://t.me/neuroport" target="_blank">Нейро-Софт</a> — '
    'portable builds of useful AI tools.'
)


def _brand_html(subtitle: str, credits_label: str, donate_title: str = "Поддержать проект", donate_popover: str = _DONATE_POPOVER_RU, donate_label: str = "Донат", credits_html: str = _CREDITS_HTML_RU) -> str:
    return f"""
<div class="brand-header">
  <div class="lang-switcher">
    <a href="?__lang=ru&__theme=dark" class="lang-btn">
      <img src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f1f7-1f1fa.svg" width="16" height="16" style="vertical-align:-3px;margin-right:4px"/>RU
    </a>
    <a href="?__lang=en&__theme=dark" class="lang-btn">
      <img src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f1ec-1f1e7.svg" width="16" height="16" style="vertical-align:-3px;margin-right:4px"/>EN
    </a>
    <details class="donate-wrap">
      <summary class="lang-btn donate-btn" title="{donate_title}"><img src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1fa99.svg" width="16" height="16"/>{donate_label}</summary>
      {donate_popover}
    </details>
  </div>
  <div class="brand-title">🎙️ VoxCPM2 — Multilingual TTS</div>
  <div class="brand-subtitle">{subtitle}</div>
  <div class="brand-credits">
    {credits_html}
  </div>
  <div class="device-badge">💻 {DEVICE_INFO}</div>
</div>
"""

_BRAND_HTML_RU = _brand_html(
    "2B параметров · 30 языков · 48 kHz · Voice Design &amp; Cloning",
    "Портативная сборка",
    donate_title="Поддержать проект",
    donate_popover=_DONATE_POPOVER_RU,
    donate_label="Донат",
    credits_html=_CREDITS_HTML_RU,
)
_BRAND_HTML_EN = _brand_html(
    "2B parameters · 30 languages · 48 kHz · Voice Design &amp; Cloning",
    "Portable build",
    donate_title="Support the project",
    donate_popover=_DONATE_POPOVER_EN,
    donate_label="Donate",
    credits_html=_CREDITS_HTML_EN,
)


# === UI Builder ===
def _seed_row(prefix: str):
    # Аналогично LoRA-ряду: label выносится в Markdown над Row,
    # а Number внутри ряда идёт без собственного label (show_label=False, container=False)
    # — тогда Number и Checkbox становятся одинаковой высоты и чекбокс встаёт на уровне input'а
    gr.Markdown(f"**{I18N('label_seed')}**")
    with gr.Row(equal_height=True):
        seed = gr.Number(
            value=-1, show_label=False, precision=0,
            scale=20, min_width=200, container=False,
            elem_id=f"{prefix}_seed",
        )
        locked = gr.Checkbox(
            value=False, label=I18N("label_lock"),
            scale=0, min_width=220,
            elem_id=f"{prefix}_locked",
        )
    return seed, locked


def _advanced_block(prefix: str, show_denoise: bool = False):
    """Accordion с расширенными параметрами."""
    with gr.Accordion(label=I18N("label_advanced"), open=False, elem_id=f"{prefix}_advanced"):
        _NONE = "-- Без LoRA --"
        _choices = [_NONE] + scan_local_loras()
        gr.Markdown("**🧬 LoRA**")
        with gr.Row(equal_height=True):
            lora_sel = gr.Dropdown(
                show_label=False, choices=_choices,
                value=(_ACTIVE_LORA if _ACTIVE_LORA in _choices else _NONE),
                interactive=True, scale=20, min_width=200,
                container=False, elem_id=f"{prefix}_lora",
            )
            lora_refresh = gr.Button("🔄", size="sm", scale=0, min_width=44, elem_classes=["lora-refresh-btn"], elem_id=f"{prefix}_lora_refresh")

        lora_sel.change(
            fn=lambda n: lora_detach() if n == _NONE else lora_attach(n),
            inputs=[lora_sel], outputs=[], show_progress="hidden",
        )
        lora_refresh.click(
            fn=lambda: gr.update(choices=[_NONE] + scan_local_loras()),
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
        "lora_train_title": "🎓 Train new LoRA (manual)",
        "lora_auto_title": "🎬 Auto-prepare dataset from video/audio",
        "lora_auto_desc": "Upload a long video/audio file — it will be automatically split into meaningful clips (whole sentences/paragraphs), transcribed via Parakeet TDT 0.6B v3 INT8 (~670 MB, downloads on first use), and (optionally) used to train a LoRA.",
        "lora_auto_file": "Video or audio file",
        "lora_auto_local_path": "Or paste a local media file path",
        "info_lora_auto_local_path": "Equivalent to dropping one file above; uploads take priority when both are set.",
        "lora_auto_name": "Dataset / LoRA name",
        "lora_auto_min_sec": "Min clip (sec)",
        "lora_auto_max_sec": "Max clip (sec)",
        "lora_auto_target_min": "Target min (sec)",
        "lora_auto_target_max": "Target max (sec)",
        "lora_auto_params": "Segmentation parameters",
        "lora_auto_start_training": "Start training after dataset is ready",
        "lora_auto_tune": "🤖 Auto-tune LoRA settings (r, α, steps, lr) by dataset size",
        "lora_auto_btn": "🎬 Auto-prepare",
        "lora_auto_log": "Auto-prep log",
        "lora_auto_how_title": "How does this work?",
        "lora_auto_how_desc": (
            "**Pipeline:**\n\n"
            "1. **ffmpeg** extracts the audio track and converts it to 16 kHz mono WAV\n"
            "2. **Parakeet TDT 0.6B v3 INT8** (NVIDIA NeMo, 25 European languages incl. Russian) "
            "transcribes everything in one pass with word-level timestamps. The model downloads once (~670 MB) "
            "into the HuggingFace cache and runs on GPU if CUDA is available\n"
            "3. Words are grouped into **whole sentences** by punctuation (`.` `!` `?` `…`)\n"
            "4. Short neighbouring sentences (< target min) are glued together into meaningful paragraphs "
            "if the pause between them is under 0.5 s\n"
            "5. Overly long sentences (> max sec) are split at commas / semicolons / long pauses — never mid-word\n"
            "6. Each clip is cut straight by word-level timestamps from Parakeet (+100 ms padding at the edges) — no silence-trim hacks\n"
            "7. If **Auto-tune** is ON — `r / α / steps / lr` are chosen automatically from the clip count "
            "(based on OpenBMB's official table, e.g. 10-20 clips → `r=16, steps=500`; 100+ clips → `r=32, steps=2000`)\n\n"
            "**What you get:** `train_data/<name>/audio/clip_NNNN.wav` + `transcripts.txt` — ready for LoRA training. "
            "If the checkbox is on, training starts immediately after preparation."
        ),
        "lora_manual_desc": "Prepare the dataset manually: upload 5-50 audio files (wav/mp3/flac, 3-15 sec each). Below, provide transcripts line by line in the format `filename.wav|exact text`.\n\n**Minimum**: 5-10 minutes of audio. Optimal — clean recording of a single speaker.",
        # --- UI labels (was hardcoded) ---
        "label_examples": "Examples",
        "label_audio_files": "Audio files (wav/mp3/flac/m4a)",
        "lora_manual_local_path": "Or paste a local audio file/folder path",
        "info_lora_manual_local_path": "Equivalent to selecting files above. Accepts audio files directly or an Auto-prepare dataset folder containing audio/ and transcripts.txt.",
        "label_transcripts": "Transcripts (format: filename|text)",
        "label_lora_r": "LoRA rank (r)",
        "label_lora_alpha": "LoRA alpha",
        "label_lora_steps": "Training steps",
        "label_lora_lr": "Learning rate",
        "label_vram_profile": "GPU VRAM profile",
        "vram_profile_8gb": "8 GB — safe",
        "vram_profile_12gb": "12 GB — balanced",
        "vram_profile_16gb": "16 GB — balanced",
        "vram_profile_over_16gb": ">16 GB — max speed (unrestricted)",
        "info_vram_profile": "Manual choice. Lower-VRAM profiles cap sample length and enable memory savings; larger cards are never forced into the 8 GB limits.",
        "vram_recommended_8gb": "Recommended starting profile for the detected GPU: **8 GB — safe**.",
        "vram_recommended_12gb": "Recommended starting profile for the detected GPU: **12 GB — balanced**.",
        "vram_recommended_16gb": "Recommended starting profile for the detected GPU: **16 GB — balanced**.",
        "vram_recommended_over_16gb": "Recommended starting profile for the detected GPU: **>16 GB — max speed**.",
        "label_train_log": "Training log",
        "btn_train_start": "🎓 Start training",
        "lora_name_info": "Used for both dataset folder (train_data/<name>/) and LoRA folder (lora/<name>/)",
        # --- Placeholders ---
        "ph_tts_text": "Enter text in any of 30 supported languages...",
        "ph_design_desc": "e.g. A young woman with a soft gentle voice",
        "ph_design_text": "Hello, welcome to VoxCPM2!",
        "ph_clone_transcript": "Exact text of what is spoken in the reference audio",
        "ph_clone_style": "e.g. slightly faster, cheerful tone",
        "ph_transcripts": "clip_001.wav|Hello, my name is Ivan.\nclip_002.wav|Today is a beautiful day.",
        # --- Info hints ---
        "info_auto_min_sec": "Shorter clips are dropped",
        "info_auto_max_sec": "VoxCPM dislikes clips longer than 15 sec",
        "info_auto_target_min": "Short sentences get merged",
        "info_auto_target_max": "Long ones split by commas/pauses",
        "info_auto_tune": "Only selects r / alpha / steps / lr. Transcription and clip creation always run, even when Auto-tune is off.",
        "info_lora_r": "OpenBMB default is 32 (for 2-60 min speech). Use 64 for 60+ min / new language",
        "info_lora_alpha": "Usually = r (OpenBMB YAML: 32)",
        "info_lora_steps": "OpenBMB default: 1000 (for 5-10 min). 2000 for full convergence (FAQ)",
        "info_lora_lr": "0.0001 standard (lower to 5e-5 for 60+ min)",
        "md_step2_note": "In **🎬 Auto** mode with **Auto-tune** on, these sliders are **ignored** — settings are picked from minutes of speech (OpenBMB table). Otherwise the values below are used.",
        "md_step1_title": "📁 Step 1 · Dataset preparation",
        "md_step2_title": "🎓 Step 2 · LoRA training settings",
        "label_status": "Status",
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
        "lora_train_title": "🎓 Обучить новую LoRA (ручной режим)",
        "lora_auto_title": "🎬 Авто-подготовка датасета из видео/аудио",
        "lora_auto_desc": "Загрузи длинное видео или аудио — программа автоматически нарежет его на осмысленные куски (целые предложения / абзацы), распознает через Parakeet TDT 0.6B v3 INT8 (~670 MB, скачается при первом использовании) и (опционально) сразу запустит обучение LoRA.",
        "lora_auto_file": "Видео или аудио",
        "lora_auto_local_path": "Или вставьте локальный путь к медиафайлу",
        "info_lora_auto_local_path": "Работает как перетаскивание одного файла; загруженный файл имеет приоритет.",
        "lora_auto_name": "Имя датасета / LoRA",
        "lora_auto_min_sec": "Мин клип (сек)",
        "lora_auto_max_sec": "Макс клип (сек)",
        "lora_auto_target_min": "Целевой min (сек)",
        "lora_auto_target_max": "Целевой max (сек)",
        "lora_auto_params": "Параметры нарезки",
        "lora_auto_start_training": "Запустить обучение после подготовки датасета",
        "lora_auto_tune": "🤖 Авто-тюнинг настроек LoRA (r, α, steps, lr) по размеру датасета",
        "lora_auto_btn": "🎬 Авто-подготовка",
        "lora_auto_log": "Лог авто-подготовки",
        "lora_auto_how_title": "Как это работает?",
        "lora_auto_how_desc": (
            "**Пайплайн:**\n\n"
            "1. **ffmpeg** извлекает аудио-дорожку и конвертирует в 16 kHz mono WAV\n"
            "2. **Parakeet TDT 0.6B v3 INT8** (NVIDIA NeMo, 25 европейских языков включая русский) "
            "транскрибирует всё за один проход, выдавая таймстампы каждого слова. Модель скачается при "
            "первом запуске (~670 MB) в HF cache и использует GPU если CUDA доступна\n"
            "3. Слова группируются в **целые предложения** по пунктуации (`.` `!` `?` `…`)\n"
            "4. Короткие соседние предложения (< целевой min) склеиваются в осмысленные абзацы, "
            "если пауза между ними меньше 0.5 сек\n"
            "5. Слишком длинные предложения (> max sec) режутся по запятым / точкам с запятой / "
            "длинным паузам — никогда не посреди слова\n"
            "6. Клипы режутся ровно по word-timestamps от Parakeet (+100 мс паддинг по краям, чтобы не обрезать хвост фонемы) — без костылей с VAD/librosa\n"
            "7. Если **Авто-тюнинг** включён — `r / α / steps / lr` подбираются автоматически по "
            "количеству клипов (по официальной таблице OpenBMB, например 10-20 клипов → `r=16, steps=500`; "
            "100+ клипов → `r=32, steps=2000`)\n\n"
            "**Что получаешь:** `train_data/<name>/audio/clip_NNNN.wav` + `transcripts.txt` — готово к "
            "обучению LoRA. Если галочка стоит — обучение стартует сразу после подготовки."
        ),
        "lora_manual_desc": "Подготовьте датасет вручную: загрузите 5-50 аудио (wav/mp3/flac, 3-15 сек каждое). Ниже постройте транскрипты в формате: `имя_файла.wav|точный текст`.\n\n**Минимум**: 5-10 минут аудио. Оптимально — чистая запись одного голоса.",
        # --- UI labels ---
        "label_examples": "Примеры",
        "label_audio_files": "Аудиофайлы (wav/mp3/flac/m4a)",
        "lora_manual_local_path": "Или вставьте локальный путь к аудиофайлу/папке",
        "info_lora_manual_local_path": "Работает как выбор файлов выше. Поддерживает готовую папку Auto-prepare с audio/ и transcripts.txt.",
        "label_transcripts": "Транскрипты (формат: имя_файла|текст)",
        "label_lora_r": "LoRA ранг (r)",
        "label_lora_alpha": "LoRA alpha",
        "label_lora_steps": "Шагов обучения",
        "label_lora_lr": "Learning rate",
        "label_vram_profile": "Профиль видеопамяти GPU",
        "vram_profile_8gb": "8 ГБ — безопасный",
        "vram_profile_12gb": "12 ГБ — сбалансированный",
        "vram_profile_16gb": "16 ГБ — сбалансированный",
        "vram_profile_over_16gb": ">16 ГБ — максимальная скорость (без ограничений)",
        "info_vram_profile": "Выбирается вручную. Профили с меньшей VRAM ограничивают длину и экономят память; мощные GPU не получают ограничения профиля 8 ГБ.",
        "vram_recommended_8gb": "Рекомендуемый стартовый профиль для обнаруженного GPU: **8 ГБ — безопасный**.",
        "vram_recommended_12gb": "Рекомендуемый стартовый профиль для обнаруженного GPU: **12 ГБ — сбалансированный**.",
        "vram_recommended_16gb": "Рекомендуемый стартовый профиль для обнаруженного GPU: **16 ГБ — сбалансированный**.",
        "vram_recommended_over_16gb": "Рекомендуемый стартовый профиль для обнаруженного GPU: **>16 ГБ — максимальная скорость**.",
        "label_train_log": "Лог тренировки",
        "btn_train_start": "🎓 Начать обучение",
        "lora_name_info": "Используется и для папки датасета (train_data/<имя>/), и для папки LoRA (lora/<имя>/)",
        # --- Placeholders ---
        "ph_tts_text": "Введите текст на любом из 30 поддерживаемых языков...",
        "ph_design_desc": "например: Молодая женщина, нежный и мягкий голос",
        "ph_design_text": "Привет, добро пожаловать в VoxCPM2!",
        "ph_clone_transcript": "Точный текст того что говорится в референс-аудио",
        "ph_clone_style": "например: чуть быстрее, бодрым тоном",
        "ph_transcripts": "clip_001.wav|Здравствуйте, меня зовут Иван.\nclip_002.wav|Сегодня отличная погода.",
        # --- Info hints ---
        "info_auto_min_sec": "Клипы короче отбрасываются",
        "info_auto_max_sec": "VoxCPM не любит клипы длиннее 15 сек",
        "info_auto_target_min": "Короткие предложения склеиваются",
        "info_auto_target_max": "Длинные режутся по запятым/паузам",
        "info_auto_tune": "Меняет только r / alpha / steps / lr. Транскрипция и создание клипов выполняются всегда, даже если Auto-tune выключен.",
        "info_lora_r": "Официальный дефолт OpenBMB — 32 (для 2-60 мин речи). 64 для 60+ мин / смены языка",
        "info_lora_alpha": "Обычно = r (по YAML OpenBMB: 32)",
        "info_lora_steps": "Официальный дефолт: 1000 (для 5-10 мин). 2000 для полной сходимости (FAQ OpenBMB)",
        "info_lora_lr": "0.0001 стандарт (снизь до 5e-5 для 60+ мин)",
        "md_step2_note": "В режиме **🎬 Авто** с включённой галочкой «Авто-тюнинг» эти слайдеры **игнорируются** — параметры подбираются автоматически по количеству минут речи (таблица OpenBMB). В остальных случаях используются значения ниже.",
        "md_step1_title": "📁 Шаг 1 · Подготовка датасета",
        "md_step2_title": "🎓 Шаг 2 · Настройки обучения LoRA",
        "label_status": "Статус",
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
      document.documentElement.lang = lang;
    }
  } catch(e) {}

  // После загрузки Gradio — находим i18n модуль и принудительно выставляем локаль
  // (override navigator.language через Object.defineProperty ненадёжен в Gradio 6.10,
  // поэтому вызываем changeLocale напрямую — несколько раз с задержкой,
  // чтобы все табы/компоненты успели перемонтироваться)
  function applyLang() {
    try {
      var u = new URL(window.location);
      var lang = u.searchParams.get('__lang');
      if (!lang) return;
      var linkEl = document.querySelector('link[href*="i18n-"][rel*="preload"]');
      if (!linkEl) { setTimeout(applyLang, 200); return; }
      import(linkEl.href).then(function(mod){
        if (!mod || typeof mod.J !== 'function') return;
        var trigger = function() {
          try {
            mod.J(lang);
            window.dispatchEvent(new Event('languagechange'));
          } catch(e) {}
        };
        // Мульти-проход: немедленно, через 300мс, 1сек, 2сек — чтобы поймать все lazy-mount компоненты (табы)
        trigger();
        setTimeout(trigger, 300);
        setTimeout(trigger, 1000);
        setTimeout(trigger, 2000);
      }).catch(function(){});
    } catch(e) {}
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function(){ setTimeout(applyLang, 100); });
  } else {
    setTimeout(applyLang, 100);
  }
})();
</script>
"""


def build_ui():
    with gr.Blocks(
        title="VoxCPM2 — Multilingual TTS",
        delete_cache=(300, 3600),
    ) as demo:
        gr.HTML(I18N("brand_header_html"))

        # === Таб 1: TTS ===
        with gr.Tab(label=I18N("tab_tts"), render_children=False):
            gr.Markdown(I18N("tts_instructions"))
            with gr.Row(equal_height=False):
                with gr.Column(scale=1):
                    tts_text = gr.Textbox(label=I18N("label_text"), placeholder=I18N("ph_tts_text"), lines=4)
                    tts_cfg, tts_steps, tts_fmt, tts_norm, tts_retry, tts_retry_max, tts_retry_ratio, tts_min_len, tts_max_len, tts_stream, _ = _advanced_block("tts", show_denoise=False)
                    tts_seed, tts_locked = _seed_row("tts")
                    tts_btn = gr.Button(I18N("btn_tts"), variant="primary", size="lg")
                with gr.Column(scale=1):
                    tts_out = gr.Audio(label=I18N("label_output"), type="numpy", streaming=True, autoplay=True)
            gr.Examples(examples=TTS_EXAMPLES, inputs=[tts_text], label=I18N("label_examples"), examples_per_page=10)
            # Pre-click сброс audio-компонента → .then() генератор.
            # Без этого Gradio 6.10 держит старое streaming-состояние (autoplay не срабатывает,
            # новые чанки «накладываются» на старые).
            tts_btn.click(
                lambda: gr.update(value=None),
                outputs=[tts_out],
                show_progress="hidden",
            ).then(
                tts_generate,
                inputs=[tts_text, tts_cfg, tts_steps, tts_fmt, tts_retry_max, tts_retry_ratio, tts_min_len, tts_max_len, tts_stream, tts_seed, tts_locked, tts_norm, tts_retry],
                outputs=[tts_out, tts_seed],
            )

        # === Таб 2: Voice Design ===
        with gr.Tab(label=I18N("tab_design"), render_children=False):
            gr.Markdown(I18N("design_instructions"))
            with gr.Row(equal_height=False):
                with gr.Column(scale=1):
                    vd_desc = gr.Textbox(label=I18N("label_description"), placeholder=I18N("ph_design_desc"), lines=2)
                    vd_text = gr.Textbox(label=I18N("label_content"), placeholder=I18N("ph_design_text"), lines=3)
                    vd_cfg, vd_steps, vd_fmt, vd_norm, vd_retry, vd_retry_max, vd_retry_ratio, vd_min_len, vd_max_len, vd_stream, _ = _advanced_block("vd", show_denoise=False)
                    vd_seed, vd_locked = _seed_row("vd")
                    vd_btn = gr.Button(I18N("btn_design"), variant="primary", size="lg")
                with gr.Column(scale=1):
                    vd_out = gr.Audio(label=I18N("label_output"), type="numpy", streaming=True, autoplay=True)
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
                lambda: gr.update(value=None),
                outputs=[vd_out],
                show_progress="hidden",
            ).then(
                voice_design,
                inputs=[vd_desc, vd_text, vd_cfg, vd_steps, vd_fmt, vd_retry_max, vd_retry_ratio, vd_min_len, vd_max_len, vd_stream, vd_seed, vd_locked, vd_norm, vd_retry],
                outputs=[vd_out, vd_seed],
            )

        _initial_voices = scan_local_voices()
        # Limit to first 50 — Gradio 6 Dropdown has reactive issues with very large lists
        _voice_choices = ["-- Свой файл --"] + _initial_voices[:50]

        # === Таб 3: Voice Cloning ===
        with gr.Tab(label=I18N("tab_clone"), render_children=False):
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
                        placeholder=I18N("ph_clone_transcript"),
                        lines=2,
                        elem_id="vc_transcript",
                    )
                    vc_text = gr.Textbox(label=I18N("label_content"), placeholder=I18N("ph_design_text"), lines=3, elem_id="vc_text")
                    vc_style = gr.Textbox(
                        label=I18N("label_style"),
                        placeholder=I18N("ph_clone_style"),
                        lines=1,
                        elem_id="vc_style",
                    )
                    vc_cfg, vc_steps, vc_fmt, vc_norm, vc_retry, vc_retry_max, vc_retry_ratio, vc_min_len, vc_max_len, vc_stream, vc_denoise = _advanced_block("vc", show_denoise=True)
                    vc_seed, vc_locked = _seed_row("vc")
                    vc_btn = gr.Button(I18N("btn_clone"), variant="primary", size="lg", elem_id="vc_btn")
                with gr.Column(scale=1):
                    vc_out = gr.Audio(label=I18N("label_output"), type="numpy", streaming=True, autoplay=True, elem_id="vc_out")
                    with gr.Accordion(I18N("accordion_cloud"), open=False, elem_id="vc_download_accordion"):
                        gr.Markdown(f"*Репозиторий: `{CLOUD_VOICES_REPO}`*")
                        vc_cloud_status = gr.Textbox(
                            label=I18N("label_status"),
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
                lambda: gr.update(value=None),
                outputs=[vc_out],
                show_progress="hidden",
            ).then(
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
        with gr.Tab(label=I18N("tab_lora"), render_children=False):
            lora_name = gr.Textbox(
                label=I18N("lora_auto_name"),
                placeholder="my_voice_v1",
                value="",
                info=I18N("lora_name_info"),
            )

            gr.Markdown(I18N("md_step1_title"))

            # Два режима — взаимоисключающие суб-табы
            with gr.Accordion(I18N("lora_auto_title"), open=True):
                gr.Markdown(I18N("lora_auto_desc"))
                with gr.Accordion(I18N("lora_auto_how_title"), open=False):
                    gr.Markdown(I18N("lora_auto_how_desc"))
                auto_file = gr.File(
                    label=I18N("lora_auto_file"),
                    file_types=[
                        ".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v",
                        ".mp3", ".wav", ".flac", ".m4a", ".ogg", ".opus",
                    ],
                )
                auto_local_media_path = gr.Textbox(
                    label=I18N("lora_auto_local_path"),
                    info=I18N("info_lora_auto_local_path"),
                    placeholder=r"C:\path\to\recording.mp3",
                    lines=1,
                )
                with gr.Accordion(I18N("lora_auto_params"), open=False):
                    with gr.Row():
                        auto_min_sec = gr.Slider(1.0, 5.0, value=2.0, step=0.5, label=I18N("lora_auto_min_sec"), info=I18N("info_auto_min_sec"))
                        auto_max_sec = gr.Slider(8.0, 20.0, value=15.0, step=1.0, label=I18N("lora_auto_max_sec"), info=I18N("info_auto_max_sec"))
                    with gr.Row():
                        auto_target_min = gr.Slider(2.0, 8.0, value=4.0, step=0.5, label=I18N("lora_auto_target_min"), info=I18N("info_auto_target_min"))
                        auto_target_max = gr.Slider(6.0, 15.0, value=10.0, step=0.5, label=I18N("lora_auto_target_max"), info=I18N("info_auto_target_max"))
                with gr.Row():
                    auto_tune_chk = gr.Checkbox(
                        label=I18N("lora_auto_tune"),
                        value=True,
                        info=I18N("info_auto_tune"),
                    )
                    auto_start_train = gr.Checkbox(
                        label=I18N("lora_auto_start_training"),
                        value=False,
                    )
                auto_btn = gr.Button(I18N("lora_auto_btn"), variant="primary", size="lg")
                auto_log = gr.Textbox(label=I18N("lora_auto_log"), interactive=False, lines=18)

                # === РУЧНОЙ режим ===
            with gr.Accordion(I18N("lora_train_title"), open=False):
                gr.Markdown(I18N("lora_manual_desc"))
                lora_files = gr.Files(
                    label=I18N("label_audio_files"),
                    file_types=[".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus"],
                    file_count="multiple",
                )
                lora_local_audio_path = gr.Textbox(
                    label=I18N("lora_manual_local_path"),
                    info=I18N("info_lora_manual_local_path"),
                    placeholder=r"C:\path\to\audio_folder",
                    lines=1,
                )
                lora_transcripts = gr.Textbox(
                    label=I18N("label_transcripts"),
                    placeholder=I18N("ph_transcripts"),
                    lines=8,
                )

            gr.Markdown("---")
            gr.Markdown(I18N("md_step2_title"))
            gr.Markdown(I18N("md_step2_note"))
            recommended_profile = load_selected_vram_profile()
            gr.Markdown(f"**GPU:** {DEVICE_INFO}")
            gr.Markdown(I18N(f"vram_recommended_{recommended_profile}"))
            lora_vram_profile = gr.Dropdown(
                choices=VRAM_PROFILE_DROPDOWN_CHOICES,
                value=recommended_profile,
                label=I18N("label_vram_profile"),
                info=I18N("info_vram_profile"),
                interactive=True,
            )
            with gr.Row():
                lora_r = gr.Slider(8, 128, value=32, step=8, label=I18N("label_lora_r"), info=I18N("info_lora_r"))
                lora_alpha = gr.Slider(8, 128, value=32, step=8, label=I18N("label_lora_alpha"), info=I18N("info_lora_alpha"))
            with gr.Row():
                lora_steps = gr.Slider(100, 5000, value=1000, step=50, label=I18N("label_lora_steps"), info=I18N("info_lora_steps"))
                lora_lr = gr.Slider(0.00001, 0.001, value=0.0001, step=0.00001, label=I18N("label_lora_lr"), info=I18N("info_lora_lr"))

            lora_train_btn = gr.Button(I18N("btn_train_start"), variant="primary", size="lg")
            lora_train_log = gr.Textbox(label=I18N("label_train_log"), interactive=False, lines=15)

            # --- wiring ---
            auto_btn.click(
                auto_prepare_dataset,
                inputs=[
                    lora_name, auto_file,
                    auto_min_sec, auto_max_sec, auto_target_min, auto_target_max,
                    auto_start_train, auto_tune_chk,
                    lora_r, lora_alpha, lora_steps, lora_lr,
                    auto_local_media_path,
                ],
                outputs=[auto_log, lora_transcripts, lora_files, lora_r, lora_alpha, lora_steps, lora_lr],
                api_name="auto_prepare_lora_dataset",
            ).then(
                maybe_auto_train,
                inputs=[
                    auto_start_train, lora_name, lora_files, lora_transcripts,
                    lora_r, lora_alpha, lora_steps, lora_lr, lora_vram_profile,
                ],
                outputs=[lora_train_log],
                api_name="auto_train_lora",
            )

            lora_train_btn.click(
                train_lora,
                inputs=[
                    lora_name, lora_files, lora_transcripts,
                    lora_r, lora_alpha, lora_steps, lora_lr, lora_vram_profile,
                    lora_local_audio_path,
                ],
                outputs=[lora_train_log],
                api_name="train_lora",
            )
            lora_vram_profile.change(
                _persist_vram_profile_from_ui,
                inputs=[lora_vram_profile],
                outputs=[],
                show_progress="hidden",
                api_visibility="private",
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
        head=_HEAD_JS,
        show_error=True,
    )
