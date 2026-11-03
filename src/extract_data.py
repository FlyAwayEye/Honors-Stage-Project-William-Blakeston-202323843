import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import csv
import io
import shutil
from pathlib import Path
import librosa
import soundfile as sf
from datasets import load_dataset, concatenate_datasets, Audio

# 1. Configuration
DATASET_ID = "fixie-ai/common_voice_17_0"
TARGET_ACCENTS = ["scotland", "scottish", "ireland", "irish", "wales", "welsh"]
EXCLUDED_ACCENTS = [
    "england", "southern", "spanish", "united states", "american",
    "canadian", "australian", "indian", "german", "french", "italian"
]
SAMPLE_RATE = 16000

# Full Time Targets (12 hours total)
SPLIT_TARGETS_HOURS = {
    "test": 1.0,        # Isolated Challenge Dataset (1 hour)
    "validation": 1.0,  # Validation subset (1 hour)
    "train": 10.0       # Training subset (10 hours)
}

BASE_DIR = Path("data")


def matches_target_accent(row: dict) -> str | None:
    """Return the matched accent label if strictly Scottish, Irish, or Welsh."""
    raw_accent = str(row.get("accents") or row.get("accent") or "").lower().strip()
    if not raw_accent:
        return None
    for excluded in EXCLUDED_ACCENTS:
        if excluded in raw_accent:
            return None
    for target in TARGET_ACCENTS:
        if target in raw_accent:
            return raw_accent
    return None


def main():
    if BASE_DIR.exists():
        shutil.rmtree(BASE_DIR)
    BASE_DIR.mkdir(exist_ok=True)

    print("Connecting to Common Voice 17.0 stream...")
    ds_test = load_dataset(DATASET_ID, "en", split="test", streaming=True).cast_column("audio", Audio(decode=False))
    ds_val = load_dataset(DATASET_ID, "en", split="validation", streaming=True).cast_column("audio", Audio(decode=False))
    ds_train = load_dataset(DATASET_ID, "en", split="train", streaming=True).cast_column("audio", Audio(decode=False))

    full_stream = concatenate_datasets([ds_test, ds_val, ds_train])

    splits = ["test", "validation", "train"]
    target_seconds = {s: SPLIT_TARGETS_HOURS[s] * 3600.0 for s in splits}
    accumulated_seconds = {s: 0.0 for s in splits}
    saved_counts = {s: 0 for s in splits}

    csv_files = {}
    csv_writers = {}
    out_dirs = {}

    for s in splits:
        out_dir = BASE_DIR / s / "clips"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_dirs[s] = out_dir

        f = open(BASE_DIR / s / "metadata.csv", mode="w", newline="", encoding="utf-8")
        csv_files[s] = f
        writer = csv.writer(f)
        writer.writerow(["file_name", "sentence", "accent", "client_id", "duration_s"])
        csv_writers[s] = writer

    # Permanent mapping of speaker_id -> split_name to guarantee zero data leakage
    speaker_to_split: dict[str, str] = {}
    total_scanned = 0

    print("\n--- Streaming & Routing Splits Simultaneously (test: 1h, val: 1h, train: 10h) ---")

    try:
        for row in full_stream:
            total_scanned += 1
            if total_scanned % 5000 == 0:
                status = " | ".join(
                    f"{s}: {accumulated_seconds[s]/3600:.2f}/{SPLIT_TARGETS_HOURS[s]}h ({saved_counts[s]})"
                    for s in splits
                )
                print(f"Scanned {total_scanned:,} | {status}")

            accent_match = matches_target_accent(row)
            if not accent_match:
                continue

            sentence = str(row.get("sentence", "")).strip()
            if not sentence:
                continue

            raw_id = str(row.get("client_id") or "").strip()
            speaker_id = raw_id if raw_id else f"unknown_{total_scanned}"

            # Assign new speaker permanently to the unfinished split with the lowest % progress
            if speaker_id not in speaker_to_split:
                unfinished = [s for s in splits if accumulated_seconds[s] < target_seconds[s]]
                if not unfinished:
                    break
                assigned_split = min(
                    unfinished,
                    key=lambda s: accumulated_seconds[s] / target_seconds[s]
                )
                speaker_to_split[speaker_id] = assigned_split
            else:
                assigned_split = speaker_to_split[speaker_id]

            # If this speaker's assigned split is already full, skip (never cross-contaminate splits)
            if accumulated_seconds[assigned_split] >= target_seconds[assigned_split]:
                continue

            audio_bytes = row["audio"].get("bytes")
            if not audio_bytes:
                continue

            try:
                audio_array, _ = librosa.load(io.BytesIO(audio_bytes), sr=SAMPLE_RATE, mono=True)
            except Exception:
                continue

            duration = len(audio_array) / SAMPLE_RATE
            if duration < 1.0 or duration > 30.0:
                continue

            idx = saved_counts[assigned_split]
            file_name = f"{assigned_split}_{idx:06d}.wav"
            wav_path = out_dirs[assigned_split] / file_name

            sf.write(str(wav_path), audio_array, SAMPLE_RATE, subtype="PCM_16")

            csv_writers[assigned_split].writerow([
                f"clips/{file_name}", sentence, accent_match, speaker_id, round(duration, 2)
            ])
            csv_files[assigned_split].flush()

            accumulated_seconds[assigned_split] += duration
            saved_counts[assigned_split] += 1

            # Stop as soon as all 3 splits reach their target hours
            if all(accumulated_seconds[s] >= target_seconds[s] for s in splits):
                print("\nAll splits reached target hours!")
                break
    finally:
        for f in csv_files.values():
            f.close()

    print("\n--- Final Dataset Summary ---")
    for s in splits:
        print(f"Split '{s}': {saved_counts[s]} clips ({accumulated_seconds[s]/3600:.2f} hours)")
    print("Data Extraction & Preprocessing Complete!")
    os._exit(0)


if __name__ == "__main__":
    main()