"""
Training script for the aspect SENTIMENT classifier (single-label, 3-class).

Fine-tunes a pretrained transformer as a sentence-PAIR classifier: given
(review_text, feature_display_name), predict sentiment toward that specific
feature. This is the second half of the hybrid ABSA pipeline — the category
classifier (Phase 4, File 8) decides WHICH features are discussed; this
model decides HOW the user feels about each one.

Usage:
    PYTHONPATH=src python scripts/train_aspect_sentiment.py
"""

import json
import os
import sys
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from fixfirst.config.configuration import settings
from fixfirst.exceptions.exception import FixFirstException
from fixfirst.logging.logger import logging
from fixfirst.ml._training.aspect_sentiment.dataset import (
    AspectSentimentDataset,
    SENTIMENT_LABELS,
    build_sentiment_examples,
)
from fixfirst.ml._training.aspect_sentiment.metrics import compute_sentiment_metrics

BASE_MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
BATCH_SIZE = 16
NUM_EPOCHS = 4
LEARNING_RATE = 2e-5
VAL_SIZE = 0.1
RANDOM_STATE = 42


def _load_inputs() -> pd.DataFrame:
    silver_dir = settings.resolve_path(settings.data_silver_labels_dir)
    labels_df = pd.read_parquet(silver_dir / "silver_labels.parquet")
    return labels_df


def train_aspect_sentiment_model(limit: int = None) -> Dict:
    """
    Full training run: build sentence-pair dataset -> tokenize -> fine-tune
    -> evaluate -> log to MLflow -> save model + metadata -> register
    ModelRun in Postgres. Returns the final validation metrics dict.
    """
    import torch
    from torch import nn
    import mlflow
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        Trainer,
        TrainingArguments,
        EarlyStoppingCallback,
    )

    from fixfirst.ml._labeling.taxonomy import load_active_taxonomy

    # --- CUSTOM TRAINER FOR WEIGHTED CROSS ENTROPY LOSS ---
    class WeightedSentimentTrainer(Trainer):
        def __init__(self, *args, class_weights_tensor=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.class_weights_tensor = class_weights_tensor

        def compute_loss(self, model, inputs, return_outputs=False):
            # Extract the integer targets
            labels = inputs.pop("labels")
            # Get model predictions
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Apply our custom inverse class weights to the loss function
            if self.class_weights_tensor is not None:
                self.class_weights_tensor = self.class_weights_tensor.to(logits.device)
                loss_fct = nn.CrossEntropyLoss(weight=self.class_weights_tensor)
            else:
                loss_fct = nn.CrossEntropyLoss()
                
            loss = loss_fct(
                logits.view(-1, self.model.config.num_labels), 
                labels.view(-1)
            )
            
            return (loss, outputs) if return_outputs else loss
    # ------------------------------------------------------

    try:
        taxonomy = load_active_taxonomy()
        feature_display_names = {t["feature_key"]: t["display_name"] for t in taxonomy}

        labels_df = _load_inputs()
        if limit:
            labels_df = labels_df.head(limit)

        # UNPACK THE TUPLE HERE
        examples_df, class_weights = build_sentiment_examples(labels_df, feature_display_names)

        # Stratify by label so all 3 sentiment classes are represented in
        # both splits, guarding against a rare class landing entirely in
        # one split by chance on smaller datasets.
        label_counts = examples_df["label"].value_counts()
        can_stratify = (label_counts >= 2).all() and examples_df["label"].nunique() > 1

        train_df, val_df = train_test_split(
            examples_df,
            test_size=VAL_SIZE,
            random_state=RANDOM_STATE,
            stratify=examples_df["label"] if can_stratify else None,
        )
        if not can_stratify:
            logging.info(
                "train_aspect_sentiment_model: skipping stratified split "
                "(a sentiment class has <2 examples), using plain random split"
            )
        logging.info(f"train_aspect_sentiment_model: train={len(train_df)} val={len(val_df)}")

        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL_NAME,
            num_labels=len(SENTIMENT_LABELS),
        )

        train_dataset = AspectSentimentDataset(
            train_df["text_a"].tolist(),
            train_df["text_b"].tolist(),
            train_df["label"].tolist(),
            tokenizer,
            max_length=MAX_LENGTH,
        )
        val_dataset = AspectSentimentDataset(
            val_df["text_a"].tolist(),
            val_df["text_b"].tolist(),
            val_df["label"].tolist(),
            tokenizer,
            max_length=MAX_LENGTH,
        )

        def compute_metrics(eval_pred):
            logits, true_labels = eval_pred
            return compute_sentiment_metrics(logits, true_labels)

        output_dir = str(settings.resolve_path(settings.model_artifact_dir) / "aspect_sentiment" / "checkpoints")

        if torch.backends.mps.is_available():
            os.environ["ACCELERATE_TORCH_DEVICE"] = "mps"
            
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            logging_steps=50,
            report_to=[],
        )
        if torch.backends.mps.is_available():
            training_args.__dict__["__cached__setup_devices"] = torch.device("mps")
        logging.info(f"train_aspect_sentiment_model: training device={training_args.device}")

        # Convert the dynamic weights list to a tensor
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

        # USE THE CUSTOM WEIGHTED TRAINER
        trainer = WeightedSentimentTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
            class_weights_tensor=class_weights_tensor, # Pass the weights here
        )

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment_name)

        with mlflow.start_run(run_name="aspect_sentiment_classifier") as run:
            mlflow.log_params(
                {
                    "task": "aspect_sentiment",
                    "base_model": BASE_MODEL_NAME,
                    "num_labels": len(SENTIMENT_LABELS),
                    "sentiment_labels": SENTIMENT_LABELS,
                    "max_length": MAX_LENGTH,
                    "batch_size": BATCH_SIZE,
                    "num_epochs": NUM_EPOCHS,
                    "learning_rate": LEARNING_RATE,
                    "train_size": len(train_df),
                    "val_size": len(val_df),
                    "stratified_split": can_stratify,
                }
            )

            trainer.train()
            final_metrics = trainer.evaluate()
            mlflow.log_metrics(
                {k: v for k, v in final_metrics.items() if isinstance(v, (int, float))}
            )

            model_save_dir = str(settings.resolve_path(settings.model_artifact_dir) / "aspect_sentiment" / "final")
            trainer.save_model(model_save_dir)
            tokenizer.save_pretrained(model_save_dir)

            meta = {
                "base_model": BASE_MODEL_NAME,
                "sentiment_labels": SENTIMENT_LABELS,  
                "max_length": MAX_LENGTH,
                "mlflow_run_id": run.info.run_id,
            }
            meta_path = f"{model_save_dir}/aspect_sentiment_meta.json"
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

            try:
                mlflow.log_artifacts(model_save_dir, artifact_path="model")
            except Exception as e:
                logging.error(
                    "train_aspect_sentiment_model: model was saved locally but "
                    f"MLflow artifact upload failed (non-fatal): {e}"
                )

            _register_model_run(run.info.run_id, final_metrics)

            logging.info(f"train_aspect_sentiment_model: complete. Run ID: {run.info.run_id}")
            logging.info(f"Final metrics: {final_metrics}")

        return final_metrics
    except FixFirstException:
        raise
    except Exception as e:
        raise FixFirstException(e, sys)


def _register_model_run(mlflow_run_id: str, metrics: Dict) -> None:
    from fixfirst.core._db.base import get_db
    from fixfirst.core._db.models import ModelRun, ModelTask

    try:
        with get_db() as db:
            db.add(
                ModelRun(
                    mlflow_run_id=mlflow_run_id,
                    model_name=BASE_MODEL_NAME,
                    model_version="v1",
                    task=ModelTask.aspect_sentiment,
                    metrics={k: v for k, v in metrics.items() if isinstance(v, (int, float))},
                )
            )
        logging.info(f"_register_model_run: logged ModelRun row for mlflow_run_id={mlflow_run_id}")
    except Exception as e:
        logging.error(f"_register_model_run: failed to write ModelRun row (non-fatal): {e}")