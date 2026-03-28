from __future__ import annotations

from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

from orchestration.pipeline import run_pipeline


def _build_test_config(tmp_path: Path) -> Path:
    cfg = {
        "paths": {
            "raw_data": str(tmp_path / "data" / "raw.csv"),
            "processed_data": str(tmp_path / "data" / "processed.csv"),
            "outputs": str(tmp_path / "outputs"),
            "reports": str(tmp_path / "reports"),
            "logs": str(tmp_path / "logs" / "pipeline.log"),
        },
        "logging_config": str(ROOT / "configs" / "logging.yaml"),
        "data": {
            "source": "synthetic",
            "n_samples": 300,
            "n_features": 12,
            "n_informative": 8,
            "n_redundant": 2,
            "class_sep": 1.2,
            "random_state": 0,
        },
        "preprocessing": {
            "test_size": 0.25,
            "random_state": 0,
            "scaler": "standard",
        },
        "training": {
            "model": "random_forest",
            "hyperparameters": {
                "n_estimators": [20],
                "max_depth": [5],
                "min_samples_split": [2],
            },
            "n_jobs": -1,
            "random_state": 0,
        },
        "evaluation": {"metrics": ["accuracy", "precision", "recall", "f1"]},
        "visualization": {"enabled": True},
        "ray": {"num_cpus": 2, "local_mode": True, "ignore_reinit_error": True},
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(cfg))
    return cfg_path


def test_run_pipeline_end_to_end(tmp_path: Path):
    cfg_path = _build_test_config(tmp_path)
    metrics = run_pipeline(config_path=str(cfg_path))
    for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        assert key in metrics
        assert 0.0 <= metrics[key] <= 1.0
