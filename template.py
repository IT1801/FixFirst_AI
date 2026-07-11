import os
from pathlib import Path
import logging

# Configure elegant logger matching your style
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

project_name = 'fixfirst'

list_of_files = [
    '.github/workflows/.gitkeep',

    # Shared Core Layout
    f'src/{project_name}/__init__.py',
    f'src/{project_name}/core/__init__.py',
    f'src/{project_name}/core/config.py',
    f'src/{project_name}/core/db.py',

    # Data Pipeline Logic
    f'src/{project_name}/data_pipeline/__init__.py',
    f'src/{project_name}/data_pipeline/ingestion.py',
    f'src/{project_name}/data_pipeline/preprocessing.py',

    # Machine Learning Core (Phase 3, 4, 5)
    f'src/{project_name}/ml/__init__.py',
    f'src/{project_name}/ml/labeling.py',      # SetFit / Zero-Shot NLI alternative
    f'src/{project_name}/ml/training.py',      # DistilBERT classifier pipelines
    f'src/{project_name}/ml/evaluation.py',    # Baseline metrics vs AWARE gold
    f'src/{project_name}/ml/inference.py',     # Hybrid routing (Confidence Gated)

    # Business Metrics Engine
    f'src/{project_name}/business/__init__.py',
    f'src/{project_name}/business/scoring.py',  # Criticality score formula

    # Orchestration Layer
    f'src/{project_name}/orchestration/__init__.py',
    f'src/{project_name}/orchestration/flows.py', # Prefect DAG mappings

    # API / Serving Layer
    f'src/{project_name}/api/__init__.py',
    f'src/{project_name}/api/main.py',
    f'src/{project_name}/api/routes.py',
    f'src/{project_name}/api/schemas.py',

    # App Dashboard Layer
    f'src/{project_name}/dashboard/__init__.py',
    f'src/{project_name}/dashboard/app.py',

    # Persistent Storage Scaffolding
    'data/raw/.gitkeep',
    'data/processed/.gitkeep',
    'data/silver_labels/.gitkeep',
    'data/gold_eval/.gitkeep',

    # Saved Weights and Artifacts
    'models/checkpoints/.gitkeep',
    'models/mlflow_store/.gitkeep',

    # Administrative and Research Logs
    'docs/.gitkeep',
    'logs/.gitkeep',

    # Notebook Playgrounds
    'research/01_data_exploration.ipynb',
    'research/02_zeroshot_labeling_qa.ipynb',
    'research/03_model_finetuning.ipynb',
    'research/04_criticality_simulation.ipynb',

    # Unit & Integration Tests
    'tests/__init__.py',
    'tests/test_pipeline.py',
    'tests/test_ml.py',
    'tests/test_scoring.py',
    'tests/test_api.py',

    # Build Configuration & Dependency Files
    'requirements/base.txt',
    'requirements/training.txt',
    'config/settings.yaml',
    'pyproject.toml',
    'Makefile',
    '.env.example',
    '.gitignore',
    'README.md',

    # Containerization Ecosystem
    'Dockerfile',
    '.dockerignore',
    'docker-compose.yml',
]

for file in list_of_files:
    file_path = Path(file)
    file_dir, file_name = os.path.split(file_path)

    # Ensure targeted parent directories exist cleanly
    if file_dir:
        os.makedirs(file_dir, exist_ok=True)
        logging.info(f"Directory verified/created: '{file_dir}'")

    # Only create a fresh file if it doesn't exist or is currently blank
    if not file_path.exists() or file_path.stat().st_size == 0:
        file_path.touch()
        logging.info(f"File created: '{file_name}' at '{file_dir or '.'}'")
    else:
        logging.info(f"Skipped existing non-empty file: '{file_name}' at '{file_dir or '.'}'")