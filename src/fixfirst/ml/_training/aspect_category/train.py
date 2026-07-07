"""Fine-tune the multi-label aspect category classifier."""

import json
import os
import sys
from typing import Dict

import pandas as pd
from sklearn.model_selection import train_test_split

from fixfirst.config.configuration import settings
from fixfirst.exceptions.exception import FixFirstException
from fixfirst.logging.logger import logging
from fixfirst.ml._training.aspect_category.dataset import (
    AspectCategoryDataset,
    build_category_examples,
)
from fixfirst.ml._training.aspect_category.metrics import compute_metrics_from_logits
from fixfirst.ml._training.common import build_label_index

BASE_MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
BATCH_SIZE = 8
NUM_EPOCHS = 4
LEARNING_RATE = 2e-5
VAL_SIZE = 0.1
RANDOM_STATE = 42
PREDICTION_THRESHOLD = 0.5


def _load_inputs():
    silver_dir = settings.resolve_path(settings.data_silver_labels_dir)
    labels_path = silver_dir / "silver_labels.parquet"
    progress_path = silver_dir / "silver_labeling_progress.parquet"
    if not labels_path.exists() or not progress_path.exists():
        raise FixFirstException(
            "Silver-label files are missing — run `make label` before `make train`.", sys
        )
    return pd.read_parquet(labels_path), pd.read_parquet(progress_path)


def train_aspect_category_model(limit: int = None) -> Dict:
    """Train, evaluate, persist, and register the category classifier."""
    try:
        import mlflow
        import torch
        from torch import nn
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
        )

        from fixfirst.ml._labeling.taxonomy import load_active_taxonomy

        # --- CUSTOM TRAINER FOR WEIGHTED LOSS ---
        class WeightedTrainer(Trainer):
            def __init__(self, *args, pos_weights_tensor=None, **kwargs):
                super().__init__(*args, **kwargs)
                self.pos_weights_tensor = pos_weights_tensor

            def compute_loss(self, model, inputs, return_outputs=False):
                # Extract the multi-hot targets
                labels = inputs.pop("labels")
                # Get model predictions
                outputs = model(**inputs)
                logits = outputs.logits
                
                # Apply our custom capped weights to the loss function
                if self.pos_weights_tensor is not None:
                    self.pos_weights_tensor = self.pos_weights_tensor.to(logits.device)
                    loss_fct = nn.BCEWithLogitsLoss(pos_weight=self.pos_weights_tensor)
                else:
                    loss_fct = nn.BCEWithLogitsLoss()
                    
                loss = loss_fct(
                    logits.view(-1, self.model.config.num_labels), 
                    labels.view(-1, self.model.config.num_labels)
                )
                
                return (loss, outputs) if return_outputs else loss
        # ----------------------------------------

        taxonomy = load_active_taxonomy()
        feature_keys = [item["feature_key"] for item in taxonomy]
        label_index = build_label_index(feature_keys)
        label_names = [name for name, _ in sorted(label_index.items(), key=lambda item: item[1])]

        labels_df, progress_df = _load_inputs()
        if limit is not None:
            selected_ids = progress_df["review_id"].drop_duplicates().head(limit)
            progress_df = progress_df[progress_df["review_id"].isin(selected_ids)]
            labels_df = labels_df[labels_df["review_id"].isin(selected_ids)]

        # UNPACK THE TUPLE HERE
        examples_df, pos_weights = build_category_examples(labels_df, progress_df, feature_keys)
        
        if len(examples_df) < 2:
            raise FixFirstException("At least two labeled reviews are required for training.", sys)

        train_df, val_df = train_test_split(
            examples_df,
            test_size=max(1, int(round(len(examples_df) * VAL_SIZE))),
            random_state=RANDOM_STATE,
        )
        logging.info(
            f"train_aspect_category_model: train={len(train_df)} val={len(val_df)} "
            f"labels={len(label_names)}"
        )

        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL_NAME,
            num_labels=len(label_names),
            problem_type="multi_label_classification",
            id2label={index: name for name, index in label_index.items()},
            label2id=label_index,
        )
        
        train_dataset = AspectCategoryDataset(
            train_df["review_text"].tolist(),
            train_df["labels"].tolist(),
            tokenizer,
            MAX_LENGTH,
        )
        val_dataset = AspectCategoryDataset(
            val_df["review_text"].tolist(),
            val_df["labels"].tolist(),
            tokenizer,
            MAX_LENGTH,
        )

        def compute_metrics(eval_prediction):
            logits, targets = eval_prediction
            return compute_metrics_from_logits(
                logits, targets, label_names, threshold=PREDICTION_THRESHOLD
            )

        artifact_root = settings.resolve_path(settings.model_artifact_dir) / "aspect_category"
        if torch.backends.mps.is_available():
            os.environ["ACCELERATE_TORCH_DEVICE"] = "mps"
            
        training_args = TrainingArguments(
            output_dir=str(artifact_root / "checkpoints"),
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
        logging.info(f"train_aspect_category_model: training device={training_args.device}")

        # Convert the pos_weights dictionary to a tensor sorted by label index
        weights_list = [pos_weights[name] for name, _ in sorted(label_index.items(), key=lambda item: item[1])]
        pos_weights_tensor = torch.tensor(weights_list, dtype=torch.float32)

        # USE THE CUSTOM WEIGHTED TRAINER
        trainer = WeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
            pos_weights_tensor=pos_weights_tensor, # Pass the weights here
        )

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment_name)
        with mlflow.start_run(run_name="aspect_category_classifier") as run:
            mlflow.log_params(
                {
                    "task": "aspect_category",
                    "base_model": BASE_MODEL_NAME,
                    "num_labels": len(label_names),
                    "max_length": MAX_LENGTH,
                    "batch_size": BATCH_SIZE,
                    "num_epochs": NUM_EPOCHS,
                    "learning_rate": LEARNING_RATE,
                    "train_size": len(train_df),
                    "val_size": len(val_df),
                    "prediction_threshold": PREDICTION_THRESHOLD,
                }
            )
            trainer.train()
            final_metrics = trainer.evaluate()
            mlflow.log_metrics(
                {key: value for key, value in final_metrics.items() if isinstance(value, (int, float))}
            )

            model_save_dir = artifact_root / "final"
            trainer.save_model(str(model_save_dir))
            tokenizer.save_pretrained(str(model_save_dir))
            metadata = {
                "base_model": BASE_MODEL_NAME,
                "label_index": label_index,
                "max_length": MAX_LENGTH,
                "threshold": PREDICTION_THRESHOLD,
                "mlflow_run_id": run.info.run_id,
            }
            with (model_save_dir / "aspect_category_meta.json").open("w") as file:
                json.dump(metadata, file, indent=2)
            try:
                mlflow.log_artifacts(str(model_save_dir), artifact_path="model")
            except Exception as exc:
                logging.error(
                    "train_aspect_category_model: model was saved locally but "
                    f"MLflow artifact upload failed (non-fatal): {exc}"
                )
            _register_model_run(run.info.run_id, final_metrics)

        logging.info(f"train_aspect_category_model: complete. Run ID: {run.info.run_id}")
        return final_metrics
    except FixFirstException:
        raise
    except Exception as exc:
        raise FixFirstException(exc, sys) from exc


def _register_model_run(mlflow_run_id: str, metrics: Dict) -> None:
    from fixfirst.core._db.base import get_db
    from fixfirst.core._db.models import ModelRun, ModelTask

    try:
        with get_db() as database:
            database.add(
                ModelRun(
                    mlflow_run_id=mlflow_run_id,
                    model_name=BASE_MODEL_NAME,
                    model_version="v1",
                    task=ModelTask.aspect_category,
                    metrics={
                        key: value for key, value in metrics.items() if isinstance(value, (int, float))
                    },
                )
            )
        logging.info(f"_register_model_run: logged ModelRun row for {mlflow_run_id}")
    except Exception as exc:
        logging.error(f"_register_model_run: failed to write ModelRun row (non-fatal): {exc}")