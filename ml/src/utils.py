import json
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def set_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    """Return the available PyTorch device name."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def ensure_directory(path: str | Path) -> None:
    """Create a directory and its parents if they do not exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(data: Any, path: str | Path) -> None:
    """Save data as readable JSON."""
    with Path(path).open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def load_json(path: str | Path) -> Any:
    """Load and return data from a JSON file."""
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def calculate_binary_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_prob: Sequence[float] | Sequence[Sequence[float]],
) -> dict[str, float | None]:
    """Calculate binary classification metrics, including ROC-AUC when possible."""
    true_values = np.asarray(y_true)
    predicted_values = np.asarray(y_pred)
    probability_values = np.asarray(y_prob)

    metrics: dict[str, float | None] = {
        "accuracy": float(accuracy_score(true_values, predicted_values)),
        "precision": float(precision_score(true_values, predicted_values, zero_division=0)),
        "recall": float(recall_score(true_values, predicted_values, zero_division=0)),
        "f1": float(f1_score(true_values, predicted_values, zero_division=0)),
        "roc_auc": None,
    }

    if probability_values.ndim == 2 and probability_values.shape[1] >= 2:
        positive_probabilities = probability_values[:, 1]
    elif probability_values.ndim == 1:
        positive_probabilities = probability_values
    else:
        positive_probabilities = None

    if positive_probabilities is not None and np.unique(true_values).size == 2:
        try:
            metrics["roc_auc"] = float(roc_auc_score(true_values, positive_probabilities))
        except ValueError:
            pass

    return metrics


def print_metrics(metrics: dict[str, float | None]) -> None:
    """Print metric names and values in a readable format."""
    for name, value in metrics.items():
        formatted_value = "unavailable" if value is None else f"{value:.4f}"
        print(f"{name.capitalize()}: {formatted_value}")