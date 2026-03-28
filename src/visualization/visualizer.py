from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, confusion_matrix
from sklearn.model_selection import learning_curve

sns.set_theme(style="whitegrid")


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, outputs_dir: Path, logger: logging.Logger) -> Path:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    path = outputs_dir / "confusion_matrix.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    logger.info("Confusion matrix saved to %s", path)
    return path


def plot_roc_curve(model, X_test: np.ndarray, y_test: np.ndarray, outputs_dir: Path, logger: logging.Logger) -> Path:
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
    ax.set_title("ROC Curve")
    path = outputs_dir / "roc_curve.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    logger.info("ROC curve saved to %s", path)
    return path


def plot_feature_importance(model, feature_names: List[str], outputs_dir: Path, logger: logging.Logger) -> Path:
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=sorted_importances, y=sorted_features, ax=ax, palette="viridis")
    ax.set_title("Feature Importance")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    path = outputs_dir / "feature_importance.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    logger.info("Feature importance saved to %s", path)
    return path


def plot_learning_curve(
    model,
    X: np.ndarray,
    y: np.ndarray,
    outputs_dir: Path,
    logger: logging.Logger,
    cv: int = 3,
) -> Path:
    train_sizes, train_scores, val_scores = learning_curve(
        model,
        X,
        y,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        train_sizes=np.linspace(0.2, 1.0, 5),
    )
    train_mean = np.mean(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(train_sizes, train_mean, label="Training F1", marker="o")
    ax.plot(train_sizes, val_mean, label="Validation F1", marker="s")
    ax.set_title("Learning Curve (Training vs Validation)")
    ax.set_xlabel("Training examples")
    ax.set_ylabel("F1 Score")
    ax.legend()

    path = outputs_dir / "learning_curve.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    logger.info("Training vs validation curve saved to %s", path)
    return path


def generate_all_plots(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    feature_names: List[str],
    outputs_dir: Path,
    logger: logging.Logger,
) -> Tuple[Path, Path, Path, Path]:
    outputs_dir.mkdir(parents=True, exist_ok=True)
    cm_path = plot_confusion_matrix(y_test, y_pred, outputs_dir, logger)
    roc_path = plot_roc_curve(model, X_test, y_test, outputs_dir, logger)
    fi_path = plot_feature_importance(model, feature_names, outputs_dir, logger)
    curve_path = plot_learning_curve(model, X_train, y_train, outputs_dir, logger)
    return cm_path, roc_path, fi_path, curve_path
