from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import pandas as pd
from sklearn.datasets import make_classification


def load_data(config: Dict, logger: logging.Logger) -> pd.DataFrame:
    """
    Load or generate dataset.
    If the configured raw_data path exists, load it; otherwise, generate a synthetic classification dataset.
    """
    paths = config["paths"]
    data_cfg = config["data"]
    raw_path = Path(paths["raw_data"])
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    if raw_path.exists():
        logger.info("Loading existing raw data from %s", raw_path)
        df = pd.read_csv(raw_path)
    else:
        logger.info("Generating synthetic dataset with %s samples and %s features", data_cfg["n_samples"], data_cfg["n_features"])
        X, y = make_classification(
            n_samples=data_cfg["n_samples"],
            n_features=data_cfg["n_features"],
            n_informative=data_cfg["n_informative"],
            n_redundant=data_cfg["n_redundant"],
            class_sep=data_cfg.get("class_sep", 1.0),
            random_state=data_cfg["random_state"],
        )
        feature_cols = [f"feature_{i}" for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=feature_cols)
        df["target"] = y
        df.to_csv(raw_path, index=False)
        logger.info("Synthetic dataset saved to %s", raw_path)
    return df
