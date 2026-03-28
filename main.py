from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

from orchestration.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distributed ML pipeline orchestrator")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to pipeline config")
    parser.add_argument("--ray-address", type=str, default=None, help="Existing Ray cluster address")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = run_pipeline(config_path=args.config, ray_address=args.ray_address)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
