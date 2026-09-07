import csv
from pathlib import Path
from typing import Any, Sequence

import torch
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from torch.utils.data import Dataset

from .preprocessing import load_audio, preprocess_audio


MetadataRow = dict[str, Any]


def load_dataset_metadata(metadata_path: str | Path) -> list[MetadataRow]:
    """Load and validate dataset metadata from a CSV file."""
    csv_path = Path(metadata_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Metadata file not found: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        fieldnames = set(reader.fieldnames or [])
        missing_columns = {"path", "label"} - fieldnames
        if missing_columns:
            raise ValueError(
                f"Metadata is missing required columns: {sorted(missing_columns)}"
            )

        rows: list[MetadataRow] = []
        invalid_rows: list[str] = []
        for row_number, row in enumerate(reader, start=2):
            relative_path = (row.get("path") or "").strip()
            label_text = (row.get("label") or "").strip()
            speaker_id = (row.get("speaker_id") or "").strip()
            errors: list[str] = []

            audio_path = Path(relative_path)
            if not relative_path:
                errors.append("missing path")
            elif audio_path.suffix.lower() not in {".wav", ".flac"}:
                errors.append("audio must be a WAV file; FLAC is optionally supported")
            elif not audio_path.is_absolute():
                audio_path = csv_path.parent / audio_path
            if relative_path and not audio_path.is_file():
                errors.append(f"audio file not found: {audio_path}")

            try:
                label = int(label_text)
                if label not in {0, 1}:
                    errors.append("label must be 0 or 1")
            except ValueError:
                label = -1
                errors.append("label must be an integer 0 or 1")

            if errors:
                invalid_rows.append(f"row {row_number}: {'; '.join(errors)}")
                continue

            metadata_row: MetadataRow = {"path": relative_path, "label": label}
            if speaker_id:
                metadata_row["speaker_id"] = speaker_id
            rows.append(metadata_row)

    if invalid_rows:
        details = "\n".join(invalid_rows)
        raise ValueError(f"Invalid metadata rows:\n{details}")
    if not rows:
        raise ValueError("Dataset metadata is empty")

    return rows


def _validate_split_sizes(rows: Sequence[MetadataRow]) -> None:
    """Check that a split contains both binary classes."""
    labels = {row["label"] for row in rows}
    if labels != {0, 1}:
        raise ValueError("Impossible split: every split must contain labels 0 and 1")


def _choose_grouped_fold(
    rows: Sequence[MetadataRow],
    n_splits: int,
    target_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Choose the grouped-stratified fold closest to the requested size."""
    labels = [row["label"] for row in rows]
    groups = [row["speaker_id"] for row in rows]
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    target_size = len(rows) * target_fraction
    candidates = list(splitter.split(rows, labels, groups))
    if not candidates:
        raise ValueError("Impossible split: grouped stratification produced no folds")

    _, selected_indices = min(
        candidates,
        key=lambda split: abs(len(split[1]) - target_size),
    )
    selected = set(selected_indices.tolist())
    remaining_indices = [index for index in range(len(rows)) if index not in selected]
    return remaining_indices, sorted(selected)


def _write_split(rows: Sequence[MetadataRow], path: Path) -> None:
    """Write split metadata to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["path", "label"]
    if any("speaker_id" in row for row in rows):
        fieldnames.append("speaker_id")

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def create_splits(
    rows: Sequence[MetadataRow],
    output_dir: str | Path = "ml/data/splits",
    seed: int = 42,
    validation_size: float = 0.2,
    test_size: float = 0.2,
) -> dict[str, list[MetadataRow]]:
    """Create reproducible stratified train, validation, and test splits."""
    if not rows:
        raise ValueError("Impossible split: dataset is empty")
    if validation_size <= 0 or test_size <= 0 or validation_size + test_size >= 1:
        raise ValueError("Impossible split: split sizes must be positive and leave training data")

    labels = [row["label"] for row in rows]
    if set(labels) != {0, 1}:
        raise ValueError("Impossible split: dataset must contain labels 0 and 1")

    has_speaker_ids = all(row.get("speaker_id") for row in rows)
    if has_speaker_ids:
        remaining_indices, test_indices = _choose_grouped_fold(
            rows, 5, test_size, seed
        )
        remaining_rows = [rows[index] for index in remaining_indices]
        validation_fraction = validation_size / (1 - test_size)
        train_indices, validation_indices = _choose_grouped_fold(
            remaining_rows, 4, validation_fraction, seed + 1
        )
        train_rows = [remaining_rows[index] for index in train_indices]
        validation_rows = [remaining_rows[index] for index in validation_indices]
        test_rows = [rows[index] for index in test_indices]
    else:
        train_validation_rows, test_rows = train_test_split(
            list(rows),
            test_size=test_size,
            random_state=seed,
            stratify=labels,
        )
        train_rows, validation_rows = train_test_split(
            train_validation_rows,
            test_size=validation_size / (1 - test_size),
            random_state=seed,
            stratify=[row["label"] for row in train_validation_rows],
        )

    for split_rows in (train_rows, validation_rows, test_rows):
        _validate_split_sizes(split_rows)

    split_data = {
        "train": list(train_rows),
        "validation": list(validation_rows),
        "test": list(test_rows),
    }
    split_path = Path(output_dir)
    for split_name, split_rows in split_data.items():
        _write_split(split_rows, split_path / f"{split_name}.csv")

    return split_data


class AudioDataset(Dataset[tuple[torch.Tensor, int]]):
    """PyTorch dataset that loads and preprocesses one audio file at a time."""

    def __init__(self, rows: Sequence[MetadataRow], metadata_path: str | Path | None = None):
        if not rows:
            raise ValueError("Dataset is empty")
        self.rows = list(rows)
        self.metadata_dir = Path(metadata_path).parent if metadata_path else Path.cwd()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        audio_path = Path(row["path"])
        if not audio_path.is_absolute():
            audio_path = self.metadata_dir / audio_path

        waveform, sample_rate = load_audio(audio_path)
        waveform, _ = preprocess_audio(waveform, sample_rate)
        return torch.from_numpy(waveform), int(row["label"])