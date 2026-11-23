import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import csv
import json
import math
import shutil
import time
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    get_cosine_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model

# 1. Configuration & Hyperparameters
MODEL_ID = "openai/whisper-small"
TRAIN_DIR = Path("data/train")
VAL_DIR = Path("data/validation")
OUTPUT_DIR = Path("models/whisper-small-lora")
RESULTS_DIR = Path("results")

BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 2          # Effective batch size = 16
EPOCHS = 3
LEARNING_RATE = 5e-5
WARMUP_RATIO = 0.1
EVAL_EVERY_UPDATE_STEPS = 200  # Validate roughly every 0.5 epoch


class CommonVoiceAccentDataset(Dataset):
    """Loads 16kHz mono .wav clips and reference transcripts on the fly."""
    def __init__(self, split_dir: Path):
        self.split_dir = split_dir
        self.samples = []
        metadata_path = split_dir / "metadata.csv"
        with open(metadata_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["sentence"].strip():
                    self.samples.append(row)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        wav_path = self.split_dir / item["file_name"]
        audio_array, _ = sf.read(str(wav_path), dtype="float32")
        return {
            "audio": audio_array,
            "sentence": item["sentence"].strip()
        }


class WhisperDataCollator:
    """Converts audio arrays to Mel spectrograms and strips decoder_start_token_id from labels."""
    def __init__(self, processor: WhisperProcessor, decoder_start_token_id: int):
        self.processor = processor
        self.decoder_start_token_id = decoder_start_token_id

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        audios = [f["audio"] for f in features]
        sentences = [f["sentence"] for f in features]

        inputs = self.processor.feature_extractor(
            audios, sampling_rate=16000, return_tensors="pt"
        )
        input_features = inputs.input_features

        labels_batch = self.processor.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=448,
            return_tensors="pt"
        )
        # Mask padding tokens with -100 so they are ignored in cross-entropy loss
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Strip <|startoftranscript|> (50258) because Whisper's forward() prepends it internally
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        return {
            "input_features": input_features,
            "labels": labels
        }


def evaluate_loss(model, val_loader, device, amp_dtype) -> float:
    """Compute average validation loss across the validation split."""
    model.eval()
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            input_features = batch["input_features"].to(device)
            labels = batch["labels"].to(device)
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                outputs = model(input_features=input_features, labels=labels)
            total_loss += outputs.loss.item()
            total_batches += 1

    model.train()
    return total_loss / max(1, total_batches)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for LoRA training.")

    device = "cuda"
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # Clear previous checkpoint so stale weights can never be evaluated by mistake
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    print(f"Loading '{MODEL_ID}' on {torch.cuda.get_device_name(0)} (mixed precision: {amp_dtype})...")
    processor = WhisperProcessor.from_pretrained(MODEL_ID, language="English", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID).to(device)
    model.config.use_cache = False

    decoder_start_id = model.config.decoder_start_token_id
    print(f"Configured decoder_start_token_id: {decoder_start_id}")

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        lora_dropout=0.1,
        bias="none"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = CommonVoiceAccentDataset(TRAIN_DIR)
    val_dataset = CommonVoiceAccentDataset(VAL_DIR)
    collator = WhisperDataCollator(processor, decoder_start_token_id=decoder_start_id)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    updates_per_epoch = math.ceil(len(train_loader) / GRAD_ACCUM_STEPS)
    total_update_steps = updates_per_epoch * EPOCHS
    warmup_steps = int(total_update_steps * WARMUP_RATIO)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_update_steps
    )

    print(
        f"\n--- Starting LoRA Fine-Tuning ({EPOCHS} Epochs | {total_update_steps} Update Steps | "
        f"{len(train_dataset)} Train Clips | {len(val_dataset)} Val Clips) ---"
    )

    history = []
    best_val_loss = float("inf")
    global_update_step = 0
    overall_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        optimizer.zero_grad()
        evaluated_this_step = False

        for step, batch in enumerate(train_loader, start=1):
            input_features = batch["input_features"].to(device)
            labels = batch["labels"].to(device)

            # One-time verification that <|startoftranscript|> (50258) was stripped cleanly
            if epoch == 1 and step == 1:
                sample_ids = [t for t in labels[0].tolist() if t != -100]
                print(f"[Sanity Check] First 6 label IDs : {sample_ids[:6]}")
                print(f"[Sanity Check] Decoded label 0   : {processor.tokenizer.decode(sample_ids, skip_special_tokens=False)}\n")

            with torch.amp.autocast("cuda", dtype=amp_dtype):
                outputs = model(input_features=input_features, labels=labels)
                loss = outputs.loss / GRAD_ACCUM_STEPS

            loss.backward()
            running_loss += outputs.loss.item()
            evaluated_this_step = False

            if step % GRAD_ACCUM_STEPS == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_update_step += 1

                if global_update_step % EVAL_EVERY_UPDATE_STEPS == 0:
                    val_loss = evaluate_loss(model, val_loader, device, amp_dtype)
                    vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
                    avg_train_loss = running_loss / step
                    print(
                        f"--> [Mid-Epoch Check | Step {global_update_step}/{total_update_steps}] "
                        f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Peak VRAM: {vram_gb:.2f} GB"
                    )
                    history.append({
                        "epoch": round(epoch - 1 + (step / len(train_loader)), 2),
                        "update_step": global_update_step,
                        "train_loss": round(avg_train_loss, 4),
                        "val_loss": round(val_loss, 4),
                        "peak_vram_gb": round(vram_gb, 2)
                    })
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        model.save_pretrained(str(OUTPUT_DIR))
                        processor.save_pretrained(str(OUTPUT_DIR))
                        print(f"    [Saved new best LoRA checkpoint (Val Loss: {best_val_loss:.4f})]")
                    evaluated_this_step = True

            if step % 100 == 0 or step == len(train_loader):
                avg_train_loss = running_loss / step
                vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
                elapsed = time.time() - epoch_start
                print(
                    f"Epoch {epoch}/{EPOCHS} | Batch {step}/{len(train_loader)} | "
                    f"Train Loss: {avg_train_loss:.4f} | Peak VRAM: {vram_gb:.2f} GB | {elapsed:.1f}s"
                )

        if not evaluated_this_step:
            avg_epoch_train_loss = running_loss / len(train_loader)
            val_loss = evaluate_loss(model, val_loader, device, amp_dtype)
            epoch_time = round(time.time() - epoch_start, 2)
            vram_gb = round(torch.cuda.max_memory_allocated() / (1024 ** 3), 2)

            print(
                f"--> End of Epoch {epoch}: Train Loss = {avg_epoch_train_loss:.4f} | "
                f"Val Loss = {val_loss:.4f} ({epoch_time}s)"
            )
            history.append({
                "epoch": epoch,
                "update_step": global_update_step,
                "train_loss": round(avg_epoch_train_loss, 4),
                "val_loss": round(val_loss, 4),
                "peak_vram_gb": vram_gb
            })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model.save_pretrained(str(OUTPUT_DIR))
                processor.save_pretrained(str(OUTPUT_DIR))
                print(f"    [Saved new best LoRA checkpoint (Val Loss: {best_val_loss:.4f})]")

    total_training_time = round(time.time() - overall_start, 2)
    history_summary = {
        "base_model": MODEL_ID,
        "precision": str(amp_dtype),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRAD_ACCUM_STEPS,
        "learning_rate": LEARNING_RATE,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "target_modules": ["q_proj", "k_proj", "v_proj", "out_proj"],
        "best_validation_loss": round(best_val_loss, 4),
        "total_training_time_seconds": total_training_time,
        "eval_history": history
    }

    history_path = RESULTS_DIR / "training_history.json"
    with open(history_path, mode="w", encoding="utf-8") as f:
        json.dump(history_summary, f, indent=4)

    print("\n" + "=" * 60)
    print("LoRA FINE-TUNING COMPLETE!")
    print(f"Total Training Time : {total_training_time / 60:.2f} minutes")
    print(f"Best Val Loss       : {best_val_loss:.4f}")
    print(f"LoRA Weights Saved  : {OUTPUT_DIR}")
    print(f"Training Log Saved  : {history_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()