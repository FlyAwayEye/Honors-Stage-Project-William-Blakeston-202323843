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
# Exclude composite tags where the speaker also lists a non-target accent
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
    # Clean out the 0.02hr test files before starting the full extraction
    if BASE_DIR.exists():
        shutil.rmtree(BASE_DIR)
    BASE_DIR.mkdir(exist_ok=True)

    print("Connecting to Common Voice 17.0 stream...")
    ds_test = load_dataset(DATASET_ID, "en", split="test", streaming=True).cast_column("audio", Audio(decode=False))
    ds_val = load_dataset(DATASET_ID, "en", split="validation", streaming=True).cast_column("audio", Audio(decode=False))
    ds_train = load_dataset(DATASET_ID, "en", split="train", streaming=True).cast_column("audio", Audio(decode=False))

    # Combine streams so test/validation never run out of files before hitting 1.0 hour
    full_stream = concatenate_datasets([ds_test, ds_val, ds_train])
    stream_iter = iter(full_stream)

    used_speakers = set()
    total_scanned = 0

    for split_name in ["test", "validation", "train"]:
        target_hours = SPLIT_TARGETS_HOURS[split_name]
        target_seconds = target_hours * 3600
        accumulated_seconds = 0.0
        saved_count = 0
        current_split_speakers = set()

        out_dir = BASE_DIR / split_name / "clips"
        out_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = BASE_DIR / split_name / "metadata.csv"

        print(f"\n--- Building '{split_name}' split (Target: {target_hours} hours) ---")

        with open(metadata_path, mode="w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["file_name", "sentence", "accent", "client_id", "duration_s"])

            for row in stream_iter:
                total_scanned += 1
                accent_match = matches_target_accent(row)
                if not accent_match:
                    if total_scanned % 2500 == 0:
                        print(f"Scanned {total_scanned:,} total clips | [{split_name}] Saved: {saved_count} ({accumulated_seconds/3600:.2f} / {target_hours} hrs)")
                    continue

                speaker_id = row.get("client_id", f"unknown_{total_scanned}")
                # Strict speaker isolation: never allow a speaker from a previous split
                if speaker_id in used_speakers:
                    continue

                sentence = str(row.get("sentence", "")).strip()
                if not sentence:
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

                file_name = f"{split_name}_{saved_count:06d}.wav"
                wav_path = out_dir / file_name

                sf.write(str(wav_path), audio_array, SAMPLE_RATE, subtype="PCM_16")

                writer.writerow([f"clips/{file_name}", sentence, accent_match, speaker_id, round(duration, 2)])
                accumulated_seconds += duration
                saved_count += 1
                current_split_speakers.add(speaker_id)

                if saved_count % 25 == 0 or saved_count == 1:
                    print(f"[{split_name}] Saved {saved_count} clips ({accent_match}) | {accumulated_seconds/3600:.2f} / {target_hours} hrs")

                if accumulated_seconds >= target_seconds:
                    print(f"Reached target for '{split_name}': {saved_count} clips ({accumulated_seconds/3600:.2f} hours).")
                    break

        # Lock all speakers used in this split so they cannot appear in subsequent splits
        used_speakers.update(current_split_speakers)

    print("\nData Extraction & Preprocessing Complete!")
    os._exit(0)  # Cleanly terminate background HTTP stream without retry warnings


if __name__ == "__main__":
    main()