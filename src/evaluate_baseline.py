import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import csv
import json
import re
import time
from pathlib import Path
import torch
import soundfile as sf
import jiwer
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# 1. Configuration
MODEL_ID = "openai/whisper-small"
TEST_DIR = Path("data/test")
METADATA_PATH = TEST_DIR / "metadata.csv"
RESULTS_DIR = Path("results")
BATCH_SIZE = 16


def normalize_text(text: str) -> str:
    """Standardize text by lowercasing and removing punctuation for fair WER evaluation."""
    text = text.lower()
    # Keep only alphanumeric characters and spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # Collapse multiple spaces
    return re.sub(r"\s+", " ", text).strip()


def categorize_accent(raw_accent: str) -> str:
    """Group free-form Common Voice accent tags into Scottish, Irish, or Welsh."""
    raw = raw_accent.lower()
    if "scot" in raw:
        return "Scottish"
    if "ir" in raw:
        return "Irish"
    if "wal" in raw or "welsh" in raw:
        return "Welsh"
    return "Other"


def compute_wer_breakdown(references: list[str], hypotheses: list[str]) -> dict:
    """Compute WER = (S + D + I) / N along with individual error counts."""
    output = jiwer.process_words(references, hypotheses)
    total_ref_words = output.substitutions + output.deletions + output.hits
    return {
        "wer_percent": round(output.wer * 100, 2),
        "substitutions_S": output.substitutions,
        "deletions_D": output.deletions,
        "insertions_I": output.insertions,
        "total_words_N": total_ref_words,
        "hits": output.hits
    }


def main():
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Cannot find {METADATA_PATH}. Run extract_data.py first.")

    RESULTS_DIR.mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Loading baseline model '{MODEL_ID}' on {device.upper()} ({torch_dtype})...")

    processor = WhisperProcessor.from_pretrained(MODEL_ID)
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch_dtype
    ).to(device)
    model.eval()

    # Load test metadata
    samples = []
    with open(METADATA_PATH, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append(row)

    print(f"Evaluating {len(samples)} clips from Challenge Dataset ('data/test')...\n")
    start_time = time.time()

    all_refs_norm = []
    all_hyps_norm = []
    all_refs_raw = []
    all_hyps_raw = []
    accent_buckets = {"Scottish": ([], []), "Irish": ([], []), "Welsh": ([], [])}
    detailed_rows = []

    for i in range(0, len(samples), BATCH_SIZE):
        batch = samples[i : i + BATCH_SIZE]
        audio_arrays = []

        for item in batch:
            wav_path = TEST_DIR / item["file_name"]
            audio_data, _ = sf.read(str(wav_path), dtype="float32")
            audio_arrays.append(audio_data)

        inputs = processor(
            audio_arrays,
            sampling_rate=16000,
            return_tensors="pt"
        )
        input_features = inputs.input_features.to(device=device, dtype=torch_dtype)

        with torch.no_grad():
            generated_ids = model.generate(
                input_features,
                language="en",
                task="transcribe"
            )

        transcriptions = processor.batch_decode(generated_ids, skip_special_tokens=True)

        for item, hyp_raw in zip(batch, transcriptions):
            ref_raw = item["sentence"].strip()
            hyp_raw = hyp_raw.strip()

            ref_norm = normalize_text(ref_raw)
            hyp_norm = normalize_text(hyp_raw)

            if not ref_norm:
                continue

            all_refs_raw.append(ref_raw)
            all_hyps_raw.append(hyp_raw)
            all_refs_norm.append(ref_norm)
            all_hyps_norm.append(hyp_norm)

            group = categorize_accent(item["accent"])
            if group in accent_buckets:
                accent_buckets[group][0].append(ref_norm)
                accent_buckets[group][1].append(hyp_norm)

            clip_wer = jiwer.wer(ref_norm, hyp_norm) * 100
            detailed_rows.append({
                "file_name": item["file_name"],
                "accent_group": group,
                "raw_accent": item["accent"],
                "reference_raw": ref_raw,
                "prediction_raw": hyp_raw,
                "reference_norm": ref_norm,
                "prediction_norm": hyp_norm,
                "clip_wer_percent": round(clip_wer, 2)
            })

        done = min(i + BATCH_SIZE, len(samples))
        if done % 64 == 0 or done == len(samples):
            elapsed = time.time() - start_time
            print(f"Processed {done}/{len(samples)} clips ({elapsed:.1f}s elapsed)")

    total_time = round(time.time() - start_time, 2)

    # Calculate overall and per-accent metrics