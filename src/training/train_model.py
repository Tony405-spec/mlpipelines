from __future__ import annotations

import itertools
import logging
from typing import Dict, List, Tuple

import numpy as np
import ray
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


@ray.remote
def _score_params(params: Dict, X_train: np.ndarray, y_train: np.ndarray, random_state: int) -> Tuple[Dict, float]:
    model = RandomForestClassifier(
        random_state=random_state,
        n_jobs=-1,
        **params,
    )
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring="f1")
    return params, float(scores.mean())


def train_model(X_train: np.ndarray, y_train: np.ndarray, config: Dict, logger: logging.Logger) -> Tuple[RandomForestClassifier, Dict, List[Tuple[Dict, float]]]:
    train_cfg = config["training"]
    hyper_cfg = train_cfg["hyperparameters"]
    random_state = train_cfg["random_state"]

    keys = list(hyper_cfg.keys())
    values = [hyper_cfg[k] for k in keys]
    param_grid = [dict(zip(keys, combination)) for combination in itertools.product(*values)]

    logger.info("Starting hyperparameter search over %d combinations", len(param_grid))
    X_ref = ray.put(X_train)
    y_ref = ray.put(y_train)
    result_ids = [_score_params.remote(params, X_ref, y_ref, random_state) for params in param_grid]
    results: List[Tuple[Dict, float]] = ray.get(result_ids)

    best_params, best_score = max(results, key=lambda item: item[1])
    logger.info("Best params: %s with F1 %.4f", best_params, best_score)

    best_model = RandomForestClassifier(
        random_state=random_state,
        n_jobs=train_cfg.get("n_jobs", -1),
        **best_params,
    )
    best_model.fit(X_train, y_train)
    return best_model, best_params, results
