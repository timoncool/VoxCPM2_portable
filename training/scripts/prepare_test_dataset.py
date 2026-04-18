"""Скачивает маленький single-speaker русский датасет для теста тренировки LoRA.

Берёт первые N записей (по умолчанию 50, ~5-10 мин) из
https://huggingface.co/datasets/niobures/russian-single-speaker-speech-dataset
и раскладывает в формат готовый для VoxCPM2 LoRA:
  train_data/test_ru/audio/clip_NNN.wav
  train_data/test_ru/transcripts.txt   (формат: имя|текст — скопируй в UI)

Запуск: python training/scripts/prepare_test_dataset.py [--n 50] [--name test_ru]
"""
import argparse
import sys
from pathlib import Path

import soundfile as sf
from datasets import load_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="Сколько семплов скачать")
    ap.add_argument("--name", default="test_ru", help="Имя папки внутри train_data/")
    ap.add_argument("--repo", default="akuzdeuov/ruslan-tts",
                    help="HF dataset id (default: RUSLAN — single male Russian voice)")
    ap.add_argument("--min-sec", type=float, default=3.0, help="Мин. длина клипа в сек")
    ap.add_argument("--max-sec", type=float, default=15.0, help="Макс. длина клипа в сек")
    args = ap.parse_args()

    script_dir = Path(__file__).parent.parent.parent.absolute()
    out_dir = script_dir / "train_data" / args.name
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    print(f"[prepare] Стримлю {args.repo} (downloading 50 samples = ~100-200 MB)...")
    ds = load_dataset(args.repo, split="train", streaming=True)

    lines = []
    n_ok = 0
    for i, item in enumerate(ds):
        if n_ok >= args.n:
            break
        audio = item.get("audio") or {}
        array = audio.get("array")
        sr = audio.get("sampling_rate")
        text = item.get("text") or item.get("transcription") or item.get("sentence") or ""
        if array is None or sr is None or not text.strip():
            continue
        dur = len(array) / sr
        if dur < args.min_sec or dur > args.max_sec:
            continue
        name = f"clip_{n_ok:03d}.wav"
        sf.write(str(audio_dir / name), array, sr)
        lines.append(f"{name}|{text.strip()}")
        n_ok += 1
        if n_ok % 10 == 0:
            print(f"  [{n_ok}/{args.n}] {name} ({dur:.1f}s) → {text[:60]}")

    # Транскрипты в формате для UI
    tr_path = out_dir / "transcripts.txt"
    tr_path.write_text("\n".join(lines), encoding="utf-8")

    print()
    print(f"✓ {n_ok} сэмплов в {audio_dir}")
    print(f"✓ Транскрипты в {tr_path}")
    print()
    print("Как использовать в UI:")
    print(f"  1. Таб 'LoRA' → Имя: {args.name}")
    print(f"  2. 'Аудиофайлы' → загрузи все .wav из {audio_dir}")
    print(f"  3. 'Транскрипты' → скопируй содержимое {tr_path.name}")
    print(f"  4. Rank=32, α=32, steps=500-1000, lr=0.0001 → 'Начать обучение'")


if __name__ == "__main__":
    main()
