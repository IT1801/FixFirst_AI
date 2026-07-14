# FixFirst AI — Makefile
# Wraps the setup/pipeline commands from the README so you don't have to
# remember PYTHONPATH=src and exact script paths every time.
#
# Usage: make <target> [VAR=value]
#   e.g. make ingest CSV=data/raw/aware_reviews.csv
#        make label LIMIT=10
#        make infer SPLIT=val

PYTHON        := python3
PYTHONPATH    := src
SRC_ENV       := PYTHONPATH=$(PYTHONPATH)

CSV           ?= data/raw/aware_reviews.csv
LIMIT         ?=
SPLIT         ?= test
HALF_LIFE     ?= 90

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# --- Environment -----------------------------------------------------------

.PHONY: env
env: ## Copy .env.example to .env (won't overwrite an existing .env)
	@test -f .env && echo ".env already exists, skipping" || cp .env.example .env

.PHONY: install
install: ## Install serving/pipeline dependencies (no torch)
	$(PYTHON) -m pip install -r requirements.txt --break-system-packages

.PHONY: install-training
install-training: ## Install training-only dependencies (torch, transformers, mlflow)
	$(PYTHON) -m pip install -r requirements-training.txt --break-system-packages

# --- Docker infrastructure --------------------------------------------------

.PHONY: up
up: ## Start Postgres + MLflow only (for running scripts locally against them)
	docker compose up -d db mlflow

.PHONY: up-all
up-all: ## Build and start the full stack (db, mlflow, api, dashboard)
	docker compose up --build

.PHONY: up-all-detached
up-all-detached: ## Build and start the full stack in the background
	docker compose up --build -d

.PHONY: down
down: ## Stop all containers
	docker compose down

.PHONY: down-clean
down-clean: ## Stop all containers AND remove volumes (wipes Postgres/MLflow data)
	docker compose down -v

.PHONY: logs
logs: ## Tail logs from all running containers
	docker compose logs -f

.PHONY: psql
psql: ## Open a psql shell inside the running db container
	docker compose exec db psql -U fixfirst -d fixfirst

# --- Database bootstrap ------------------------------------------------------

.PHONY: init-db
init-db: ## Create the fixfirst schema and all tables
	$(SRC_ENV) $(PYTHON) scripts/init_db.py

.PHONY: seed-features
seed-features: ## Seed features_master taxonomy (idempotent)
	$(SRC_ENV) $(PYTHON) scripts/seed_features.py

.PHONY: bootstrap
bootstrap: up init-db seed-features ## Start infra + init DB + seed features in one step

# --- Data pipeline -----------------------------------------------------------

.PHONY: ingest
ingest: ## Ingest AWARE CSV into raw_reviews (CSV=path/to/file.csv)
	$(SRC_ENV) $(PYTHON) scripts/ingest_aware.py --csv $(CSV)

.PHONY: preprocess
preprocess: ## Clean, dedupe, filter, split into train/val/test Parquet
	$(SRC_ENV) $(PYTHON) scripts/run_preprocessing.py

# --- Model training (requires: make install-training) ------------------------

.PHONY: train-category
train-category: ## Train the aspect category (multi-label) classifier
	$(SRC_ENV) $(PYTHON) scripts/train_aspect_category.py $(if $(LIMIT),--limit $(LIMIT),)

.PHONY: train-sentiment
train-sentiment: ## Train the aspect sentiment classifier
	$(SRC_ENV) $(PYTHON) scripts/train_aspect_sentiment.py $(if $(LIMIT),--limit $(LIMIT),)

.PHONY: train
train: train-category train-sentiment ## Train both fine-tuned models

.PHONY: eval
eval: ## Evaluate both models against AWARE gold labels
	$(SRC_ENV) $(PYTHON) scripts/run_gold_eval.py

# --- Inference & scoring ------------------------------------------------------

.PHONY: infer
infer: ## Run hybrid ABSA inference (SPLIT=train|val|test, default test)
	$(SRC_ENV) $(PYTHON) scripts/run_hybrid_inference.py --split $(SPLIT) $(if $(LIMIT),--limit $(LIMIT),)

.PHONY: score
score: ## Compute and persist criticality scores (HALF_LIFE=days, default 90)
	$(SRC_ENV) $(PYTHON) scripts/run_scoring.py --half-life-days $(HALF_LIFE)

.PHONY: pipeline
pipeline: ## Run the full Prefect flow (preprocess -> infer -> score)
	$(SRC_ENV) $(PYTHON) scripts/run_pipeline_flow.py --inference-split $(SPLIT)

.PHONY: pipeline-bootstrap
pipeline-bootstrap: ## Run the full Prefect flow INCLUDING AWARE ingestion (first run only)
	$(SRC_ENV) $(PYTHON) scripts/run_pipeline_flow.py --run-ingestion --aware-csv $(CSV) --inference-split $(SPLIT)

# --- Local (non-Docker) serving ------------------------------------------------

.PHONY: api
api: ## Run the FastAPI server locally with hot reload
	$(SRC_ENV) uvicorn fixfirst.api.main:app --reload --port 8000

.PHONY: dashboard
dashboard: ## Run the Streamlit dashboard locally
	$(SRC_ENV) flask --app src/fixfirst/dashboard/app.py run --port 8501 --debug

# --- Housekeeping ---------------------------------------------------------

.PHONY: clean
clean: ## Remove __pycache__, .pyc files, and log files
	find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -rf logs

.PHONY: reset-db
reset-db: down-clean up init-db seed-features ## Wipe and rebuild the database from scratch