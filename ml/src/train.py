import argparse
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import Trainer, TrainingArguments, Wav2Vec2Processor

from .config import (
    BATCH_SIZE,
    CHECKPOINTS_DIR,
    EPOCHS,
    GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE,
    MODEL_NAME,
    NUM_WORKERS,
    RESULTS_DIR,
    SAMPLE_RATE,
    SPLITS_DIR,
    WARMUP_STEPS,
    WEIGHT_DECAY,
)
from .dataset import AudioDataset, create_splits, load_dataset_metadata
from .model import VoiceAntiSpoofModel
from .utils import calculate_binary_metrics, ensure_directory, get_device, save_json, set_seed


class AudioDataCollator:
    """Pad variable-length waveforms and create model inputs for a batch."""

    def __init__(self, processor: Wav2Vec2Processor, sampling_rate: int = SAMPLE_RATE):
        self.processor = processor
        self.sampling_rate = sampling_rate

    def __call__(self, features: list[tuple[torch.Tensor, int]]) -> dict[str, torch.Tensor]:
        waveforms = [waveform.numpy() for waveform, _ in features]
        labels = torch.tensor([label for _, label in features], dtype=torch.long)
        batch = self.processor(
            waveforms,
            sampling_rate=self.sampling_rate,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        batch["labels"] = labels
        return batch


def compute_metrics(eval_prediction: Any) -> dict[str, float | None]:
    """Calculate binary metrics from Hugging Face Trainer predictions."""
    predictions = eval_prediction.predictions
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    logits = np.asarray(predictions)
    labels = np.asarray(eval_prediction.label_ids)
    probabilities = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
    predicted_labels = np.argmax(logits, axis=-1)
    return calculate_binary_metrics(labels, predicted_labels, probabilities)


def _class_distribution(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Return class counts for a metadata split."""
    return {str(label): sum(row["label"] == label for row in rows) for label in (0, 1)}


def _build_parser() -> argparse.ArgumentParser:
    """Build command-line arguments without creating or requiring a dataset."""
    parser = argparse.ArgumentParser(description="Train the voice anti-spoofing model.")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("ml/data/raw/metadata.csv"),
        help="Path to the dataset metadata CSV.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting.")
    return parser


def main() -> None:
    """Load metadata, train on train/validation data, and evaluate the test set once."""
    args = _build_parser().parse_args()
    set_seed(args.seed)

    try:
        rows = load_dataset_metadata(args.metadata)
        split_data = create_splits(rows, output_dir=SPLITS_DIR, seed=args.seed)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"Dataset setup failed: {error}") from error

    train_rows = split_data["train"]
    validation_rows = split_data["validation"]
    test_rows = split_data["test"]
    if not train_rows or not validation_rows or not test_rows:
        raise SystemExit("Dataset setup failed: train, validation, and test splits are required")

    device = get_device()
    if device == "cpu":
        warnings.warn("CUDA is unavailable; training will fall back to CPU.", UserWarning)

    print(f"Training samples: {len(train_rows)}")
    print(f"Validation samples: {len(validation_rows)}")
    print(f"Test samples: {len(test_rows)}")
    print(f"Training class distribution: {_class_distribution(train_rows)}")
    print(f"Validation class distribution: {_class_distribution(validation_rows)}")
    print(f"Test class distribution: {_class_distribution(test_rows)}")
    print(f"Device: {device}")
    print(f"Model: {MODEL_NAME}")

    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    train_dataset = AudioDataset(train_rows, metadata_path=args.metadata)
    validation_dataset = AudioDataset(validation_rows, metadata_path=args.metadata)
    test_dataset = AudioDataset(test_rows, metadata_path=args.metadata)
    model = VoiceAntiSpoofModel.from_pretrained(MODEL_NAME)

    ensure_directory(CHECKPOINTS_DIR)
    ensure_directory(RESULTS_DIR)
    final_model_dir = Path("ml/models/final")
    ensure_directory(final_model_dir)

    training_args = TrainingArguments(
        output_dir=str(CHECKPOINTS_DIR),
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=EPOCHS,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="roc_auc",
        greater_is_better=True,
        fp16=torch.cuda.is_available() and torch.cuda.is_fp16_supported(),
        dataloader_num_workers=NUM_WORKERS,
        report_to="none",
        remove_unused_columns=False,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=AudioDataCollator(processor),
        compute_metrics=compute_metrics,
    )
    trainer.train()

    validation_metrics = trainer.evaluate(eval_dataset=validation_dataset)
    test_metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
    trainer.save_model(str(final_model_dir))
    processor.save_pretrained(str(final_model_dir))

    save_json(
        {
            "model_name": MODEL_NAME,
            "device": device,
            "training_samples": len(train_rows),
            "validation_samples": len(validation_rows),
            "test_samples": len(test_rows),
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
        },
        Path(RESULTS_DIR) / "training_metrics.json",
    )


if __name__ == "__main__":
    main()