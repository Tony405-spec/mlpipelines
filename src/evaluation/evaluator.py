from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: Dict,
    logger: logging.Logger,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate the trained model and persist metrics to disk.
    """
    outputs_dir = Path(config["paths"]["outputs"])
    outputs_dir.mkdir(parents=True, exist_ok=True)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    metrics_path = outputs_dir / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)
    logger.info("Metrics saved to %s", metrics_path)
    return metrics, y_pred, y_proba, y_test
