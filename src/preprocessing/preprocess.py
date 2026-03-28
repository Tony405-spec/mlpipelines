from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess_data(df: pd.DataFrame, config: Dict, logger: logging.Logger) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], StandardScaler]:
    """
    Clean the dataset, perform train/test split with scaling, and persist the processed training slice to CSV.
    Returns both train and test arrays (scaled) along with feature names and the fitted scaler.
    """
    paths = config["paths"]
    prep_cfg = config["preprocessing"]
    target_col = "target"

    logger.info("Starting preprocessing with %d rows", len(df))
    df = df.dropna().drop_duplicates().reset_index(drop=True)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe")

    feature_cols = [col for col in df.columns if col != target_col]
    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=prep_cfg["test_size"],
        random_state=prep_cfg["random_state"],
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    processed_path = Path(paths["processed_data"])
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    processed_df = pd.DataFrame(X_train_scaled, columns=feature_cols)
    processed_df[target_col] = y_train.values
    processed_df.to_csv(processed_path, index=False)
    logger.info("Processed training data saved to %s", processed_path)

    return X_train_scaled, X_test_scaled, y_train.to_numpy(), y_test.to_numpy(), feature_cols, scaler
