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

# Word Error Rate formula definition
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
    overall_norm_metrics = compute_wer_breakdown(all_refs_norm, all_hyps_norm)
    overall_raw_wer = round(jiwer.wer(all_refs_raw, all_hyps_raw) * 100, 2)

    accent_metrics = {}
    for group, (refs, hyps) in accent_buckets.items():
        if refs:
            accent_metrics[group] = {
                "clip_count": len(refs),
                **compute_wer_breakdown(refs, hyps)
            }

    summary = {
        "model_id": MODEL_ID,
        "dataset_split": "test (Challenge Dataset)",
        "total_clips_evaluated": len(all_refs_norm),
        "evaluation_time_seconds": total_time,
        "overall_normalized": overall_norm_metrics,
        "overall_raw_wer_percent": overall_raw_wer,
        "by_accent_normalized": accent_metrics
    }

    # Save JSON summary and CSV predictions
    summary_path = RESULTS_DIR / "baseline_summary.json"
    with open(summary_path, mode="w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    predictions_path = RESULTS_DIR / "baseline_predictions.csv"
    with open(predictions_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=detailed_rows[0].keys())
        writer.writeheader()
        writer.writerows(detailed_rows)

    # Print formatted report
    print("\n" + "=" * 60)
    print(f"BASELINE EVALUATION REPORT: {MODEL_ID}")
    print("=" * 60)
    print(f"Total Clips Evaluated : {len(all_refs_norm)} (~1.00 hour)")
    print(f"Total Execution Time  : {total_time}s")
    print(f"Overall Raw WER       : {overall_raw_wer}%")
    print(f"Overall Normalized WER: {overall_norm_metrics['wer_percent']}%")
    print(f"  - Substitutions (S) : {overall_norm_metrics['substitutions_S']}")
    print(f"  - Deletions (D)     : {overall_norm_metrics['deletions_D']}")
    print(f"  - Insertions (I)    : {overall_norm_metrics['insertions_I']}")
    print(f"  - Total Words (N)   : {overall_norm_metrics['total_words_N']}")
    print("-" * 60)
    print("Normalized WER by Regional Accent:")
    for group, stats in accent_metrics.items():
        print(f"  * {group:<10} ({stats['clip_count']:>3} clips): {stats['wer_percent']}% WER "
              f"[S:{stats['substitutions_S']}, D:{stats['deletions_D']}, I:{stats['insertions_I']}, N:{stats['total_words_N']}]")
    print("=" * 60)
    print(f"Saved summary to     : {summary_path}")
    print(f"Saved predictions to : {predictions_path}")


if __name__ == "__main__":
    main()