"""Shared helpers for Task 4 model training and evaluation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
FIG_DIR = ROOT / "figures"


def unify_member(value: Any) -> str:
    """Normalize member labels, including the historical cross-modality alias."""
    member = str(value).strip().lower()
    return "junior" if member == "nkuba" else member


def evaluate(y_true: Iterable[Any], proba: np.ndarray, classes: Iterable[Any]) -> dict[str, float]:
    """Return accuracy, macro-F1, and multiclass log-loss from probabilities."""
    y_true = np.asarray(list(y_true))
    classes = np.asarray(list(classes))
    proba = np.asarray(proba, dtype=float)
    row_sums = proba.sum(axis=1, keepdims=True)
    proba = np.divide(proba, row_sums, out=np.zeros_like(proba), where=row_sums > 0)
    if np.any(row_sums <= 0):
        raise ValueError("Every probability row must have a positive sum.")
    y_pred = classes[np.argmax(proba, axis=1)]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y_true, proba, labels=classes)),
    }


def pick_unknown_threshold(
    known_scores: Iterable[float],
    unknown_scores: Iterable[float],
    higher_is_known: bool = True,
) -> float:
    """Choose the threshold with highest balanced accuracy on known/unknown scores."""
    known = np.asarray(list(known_scores), dtype=float)
    unknown = np.asarray(list(unknown_scores), dtype=float)
    known = known[np.isfinite(known)]
    unknown = unknown[np.isfinite(unknown)]
    if not len(known) or not len(unknown):
        raise ValueError("Both known_scores and unknown_scores must contain finite values.")

    values = np.unique(np.concatenate([known, unknown]))
    midpoints = (values[:-1] + values[1:]) / 2
    span = max(float(values[-1] - values[0]), 1.0)
    thresholds = np.concatenate(([values[0] - span], midpoints, [values[-1] + span]))
    best_threshold = float(thresholds[0])
    best_score = -1.0
    for threshold in thresholds:
        if higher_is_known:
            true_known = known >= threshold
            false_accept = unknown >= threshold
        else:
            true_known = known <= threshold
            false_accept = unknown <= threshold
        balanced_accuracy = (float(true_known.mean()) + float((~false_accept).mean())) / 2
        if balanced_accuracy > best_score:
            best_score = balanced_accuracy
            best_threshold = float(threshold)
    return best_threshold


def save_confusion_matrix(y_true, y_pred, labels, title: str, filename: str) -> Path:
    """Save a labelled confusion matrix PNG under figures/."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / filename
    matrix = confusion_matrix(y_true, y_pred, labels=list(labels))
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=list(labels),
        yticklabels=list(labels),
        xlabel="Predicted",
        ylabel="True",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    cutoff = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > cutoff else "black",
            )
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def save_json(obj: Any, filename: str) -> Path:
    """Write JSON-serializable metrics or metadata under models/."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / filename
    with path.open("w", encoding="utf-8") as file:
        json.dump(obj, file, indent=2, default=_json_default)
    return path


def _json_default(value: Any):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")