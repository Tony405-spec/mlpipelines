from __future__ import annotations

import json
import logging
import sys
from logging.config import dictConfig
from pathlib import Path
from typing import Dict, Optional

import joblib
import ray
import yaml

SRC_ROOT = Path(__file__).resolve().parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from ingestion.data_loader import load_data
from preprocessing.preprocess import preprocess_data
from training.train_model import train_model
from evaluation.evaluator import evaluate_model
from visualization.visualizer import generate_all_plots


def load_config(config_path: str) -> Dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_logging(log_config_path: Path) -> None:
    log_config = yaml.safe_load(log_config_path.read_text())
    Path("logs").mkdir(exist_ok=True)
    dictConfig(log_config)


def persist_search_results(outputs_dir: Path, best_params: Dict, search_results) -> None:
    outputs_dir.mkdir(parents=True, exist_ok=True)
    (outputs_dir / "best_params.json").write_text(json.dumps(best_params, indent=2))
    search_results_payload = [{"params": params, "f1": score} for params, score in search_results]
    (outputs_dir / "search_results.json").write_text(json.dumps(search_results_payload, indent=2))


def save_model(outputs_dir: Path, model) -> Path:
    outputs_dir.mkdir(parents=True, exist_ok=True)
    model_path = outputs_dir / "model.joblib"
    joblib.dump(model, model_path)
    return model_path


def run_pipeline(config_path: str = "config.yaml", ray_address: Optional[str] = None) -> Dict:
    config = load_config(config_path)
    setup_logging(Path("configs/logging.yaml"))
    logger = logging.getLogger("mlpipeline")

    ray_cfg = config.get("ray", {})
    ray.init(
        num_cpus=ray_cfg.get("num_cpus", None),
        local_mode=ray_cfg.get("local_mode", False),
        ignore_reinit_error=ray_cfg.get("ignore_reinit_error", True),
        address=ray_address,
    )
    logger.info("Ray initialized with config: %s", ray_cfg)

    df = load_data(config, logger)
    X_train, X_test, y_train, y_test, feature_names, _ = preprocess_data(df, config, logger)

    model, best_params, search_results = train_model(X_train, y_train, config, logger)
    outputs_dir = Path(config["paths"]["outputs"])
    persist_search_results(outputs_dir, best_params, search_results)
    model_path = save_model(outputs_dir, model)
    logger.info("Best model saved to %s", model_path)

    metrics, y_pred, y_proba, y_true = evaluate_model(model, X_test, y_test, config, logger)
    logger.info("Evaluation metrics: %s", metrics)

    if config["visualization"]["enabled"]:
        generate_all_plots(
            model,
            X_train,
            y_train,
            X_test,
            y_true,
            y_pred,
            feature_names,
            outputs_dir=outputs_dir,
            logger=logger,
        )

    ray.shutdown()
    return metrics
