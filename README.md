# Distributed ML Pipelines

Portfolio-ready, production-style, multi-node machine learning workflow built with **Ray**, **scikit-learn**, and fully scripted visualization. The repository demonstrates a modular, Ray-parallel pipeline from ingestion through visualization with fine-grained nodes and clear orchestration.

## Project Architecture

Pipeline nodes and data flow:

- **Ingestion** (`src/ingestion/data_loader.py`): Load existing CSV or generate synthetic classification data; persists raw dataset.
- **Preprocessing** (`src/preprocessing/preprocess.py`): Clean, de-duplicate, scale features, and split train/test; saves processed CSV.
- **Training** (`src/training/train_model.py`): Ray-parallel grid search over RandomForest hyperparameters using cross-validation; trains best model.
- **Evaluation** (`src/evaluation/evaluator.py`): Compute accuracy, precision, recall, F1, and ROC-AUC; persist metrics.
- **Visualization** (`src/visualization/visualizer.py`): Save confusion matrix, ROC curve, feature importance, and learning (training vs validation) curves.
- **Orchestration** (`src/orchestration/pipeline.py`): Load config, initialize logging, start Ray, wire all nodes, persist artifacts.

Mermaid diagram of the flow:

```mermaid
flowchart LR
    A[Ingestion] --> B[Preprocessing]
    B --> C[Training (Ray Grid Search)]
    C --> D[Evaluation]
    D --> E[Visualization]
    C -->|Best Model| D
    D -->|Metrics + Plots| F[Outputs/Reports]
```

## Repository Tree

```
.
├── config.yaml
├── configs
│   └── logging.yaml
├── data
│   ├── .gitkeep
│   ├── processed
│   │   └── .gitkeep
│   └── raw
│       └── .gitkeep
├── main.py
├── notebooks
│   └── .gitkeep
├── outputs
│   └── .gitkeep
├── reports
│   └── .gitkeep
├── requirements.txt
├── src
│   ├── __init__.py
│   ├── evaluation
│   │   ├── __init__.py
│   │   └── evaluator.py
│   ├── ingestion
│   │   ├── __init__.py
│   │   └── data_loader.py
│   ├── orchestration
│   │   ├── __init__.py
│   │   └── pipeline.py
│   ├── preprocessing
│   │   ├── __init__.py
│   │   └── preprocess.py
│   ├── training
│   │   ├── __init__.py
│   │   └── train_model.py
│   └── visualization
│       ├── __init__.py
│       └── visualizer.py
└── tests
    └── test_pipeline.py
```

## Configuration

All pipeline settings live in `config.yaml`:

- **paths**: raw/processed data, outputs, reports, logs
- **data**: synthetic data parameters (samples, features, random seeds)
- **preprocessing**: test split, scaler type
- **training**: model type, hyperparameter grid, seeds, CPU usage
- **evaluation**: metrics list
- **visualization**: toggle plotting
- **ray**: CPU count, local/cluster mode flags

Logging is defined in `configs/logging.yaml` with console + rotating file handlers.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Use the src layout by exporting `PYTHONPATH` or relying on `main.py` which injects it automatically:

```bash
export PYTHONPATH=$(pwd)/src
```

## Running the Pipeline

```bash
python main.py --config config.yaml
```

Key artifacts after a run:

- `data/raw/raw_data.csv` – generated or loaded dataset
- `data/processed/processed.csv` – cleaned training subset
- `outputs/model.joblib` – trained RandomForest model
- `outputs/metrics.json` – accuracy/precision/recall/F1/ROC-AUC
- `outputs/confusion_matrix.png`
- `outputs/roc_curve.png`
- `outputs/feature_importance.png`
- `outputs/learning_curve.png`
- `outputs/best_params.json` & `outputs/search_results.json`

### Connect to an existing Ray cluster

```bash
python main.py --config config.yaml --ray-address ray://<head-node-ip>:10001
```

### Run tests

```bash
PYTHONPATH=$(pwd)/src pytest -q
```

## Implementation Notes

- **Distributed training**: Hyperparameter candidates are fanned out as Ray remote tasks; cross-validation runs in parallel across CPUs.
- **Fine-grained nodes**: Each pipeline step is a discrete module with clear interfaces and logging.
- **Determinism**: Seeds set across ingestion and training for reproducibility.
- **Visualization**: Matplotlib/Seaborn use headless backend; plots saved directly to disk.

## Scaling to Production

- **Clustered Ray**: Swap `local_mode` to `false` and point `ray_address` at a Ray head node to distribute across machines.
- **Autoscaling**: Integrate Ray’s cluster launcher (K8s/EC2) with this code unchanged.
- **Data backends**: Replace synthetic ingestion with real sources (object storage, Kafka) and use Ray Datasets for distributed ETL.
- **Model registry**: Persist `model.joblib` to an artifact store (S3/MLflow) and attach CI for drift/quality gates.
- **Observability**: Extend logging.yaml to ship logs to ELK/Datadog; add Ray metrics exporters for cluster health.
