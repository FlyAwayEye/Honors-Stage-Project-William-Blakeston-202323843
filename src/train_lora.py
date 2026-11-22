import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import warnings
warnings.filterwarnings("ignore", message="MatMul8bitLt: inputs will be cast")

import csv
import json
import time
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 1. Configuration & Hyperparameters
MODEL_ID = "openai/whisper-small"
TRAIN_DIR = Path("data/train")
VAL_DIR = Path("data/validation")
OUTPUT_DIR = Path("models/whisper-small-lora")
RESULTS_DIR = Path("results")

BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 2          # Effective batch size = 16
EPOCHS = 3
LEARNING_RATE = 1e-4
WARMUP_RATIO = 0.1


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
    """Converts audio arrays to Mel spectrograms and pads token labels with -100."""
    def __init__(self, processor: WhisperProcessor):
        self.processor = processor

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

        # Strip beginning-of-transcript token if appended automatically
        bos_id = self.processor.tokenizer.bos_token_id
        if (labels[:, 0] == bos_id).all().cpu().item():
            labels = labels[:, 1:]

        return {
            "input_features": input_features,
            "labels": labels
        }


def evaluate_loss(model, val_loader, device) -> float:
    """Compute average validation loss across the validation split."""
    model.eval()
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            input_features = batch["input_features"].to(device, dtype=torch.float16)
            labels = batch["labels"].to(device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = model(input_features=input_features, labels=labels)
            total_loss += outputs.loss.item()
            total_batches += 1

    return total_loss / max(1, total_batches)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for 8-bit quantization and LoRA training.")

    device = "cuda"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    print(f"Loading processor and '{MODEL_ID}' in 8-bit quantization on {torch.cuda.get_device_name(0)}...")
    processor = WhisperProcessor.from_pretrained(MODEL_ID, language="English", task="transcribe")

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto"
    )
    model.config.use_cache = False

    # Prepare 8-bit model and attach LoRA adapters to attention projections
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Build Datasets and DataLoaders
    train_dataset = CommonVoiceAccentDataset(TRAIN_DIR)
    val_dataset = CommonVoiceAccentDataset(VAL_DIR)
    collator = WhisperDataCollator(processor)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_update_steps = (len(train_loader) // GRAD_ACCUM_STEPS) * EPOCHS
    warmup_steps = int(total_update_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_update_steps
    )

    print(f"\n--- Starting LoRA Fine-Tuning ({EPOCHS} Epochs | {len(train_dataset)} Train Clips | {len(val_dataset)} Val Clips) ---")
    history = []
    best_val_loss = float("inf")
    overall_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader, start=1):
            input_features = batch["input_features"].to(device, dtype=torch.float16)
            labels = batch["labels"].to(device)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = model(input_features=input_features, labels=labels)
                loss = outputs.loss / GRAD_ACCUM_STEPS

            loss.backward()
            running_loss += outputs.loss.item()

            if step % GRAD_ACCUM_STEPS == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if step % 100 == 0 or step == len(train_loader):
                avg_train_loss = running_loss / step
                vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
                elapsed = time.time() - epoch_start
                print(
                    f"Epoch {epoch}/{EPOCHS} | Step {step}/{len(train_loader)} | "
                    f"Train Loss: {avg_train_loss:.4f} | Peak VRAM: {vram_gb:.2f} GB | {elapsed:.1f}s"
                )

        avg_epoch_train_loss = running_loss / len(train_loader)
        val_loss = evaluate_loss(model, val_loader, device)
        epoch_time = round(time.time() - epoch_start, 2)

        print(
            f"--> End of Epoch {epoch}: Train Loss = {avg_epoch_train_loss:.4f} | "
            f"Val Loss = {val_loss:.4f} ({epoch_time}s)"
        )

        history.append({
            "epoch": epoch,
            "train_loss": round(avg_epoch_train_loss, 4),
            "val_loss": round(val_loss, 4),
            "epoch_time_seconds": epoch_time,
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / (1024 ** 3), 2)
        })

        # Save best checkpoint whenever validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(str(OUTPUT_DIR))
            processor.save_pretrained(str(OUTPUT_DIR))
            print(f"    [Saved new best LoRA adapter weights to '{OUTPUT_DIR}']")

    total_training_time = round(time.time() - overall_start, 2)
    history_summary = {
        "base_model": MODEL_ID,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRAD_ACCUM_STEPS,
        "learning_rate": LEARNING_RATE,
        "best_validation_loss": round(best_val_loss, 4),
        "total_training_time_seconds": total_training_time,
        "epoch_history": history
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