"""Training script for the aspect SENTIMENT classifier (single-label, 3-class)."""

import json
import os
import sys
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from fixfirst.constants import EXTRACTED_LABELS_FILENAME
from fixfirst.exceptions.exception import FixFirstException
from fixfirst.logging.logger import logging
from fixfirst.ml._training.aspect_sentiment.dataset import (
    AspectSentimentDataset,
    SENTIMENT_LABELS,
    build_sentiment_examples,
)
from fixfirst.ml._training.aspect_sentiment.metrics import compute_sentiment_metrics
from fixfirst.ml._training.common import BaseModelTrainer


class AspectSentimentTrainer(BaseModelTrainer):
    """Trainer class for the aspect sentiment model."""

    def _load_inputs(self) -> pd.DataFrame:
        extracted_dir = self.settings.resolve_path(self.settings.data_extracted_labels_dir)
        labels_df = pd.read_parquet(extracted_dir / EXTRACTED_LABELS_FILENAME)
        return labels_df

    def train(self) -> Dict[str, Any]:
        """Full training run."""
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
        from peft import get_peft_model, LoraConfig, TaskType

        from fixfirst.core._db.base import get_db
        from fixfirst.core._db.models import FeatureMaster, ModelTask

        # --- CUSTOM TRAINER FOR WEIGHTED CROSS ENTROPY LOSS ---
        class WeightedSentimentTrainer(Trainer):
            def __init__(self, *args, class_weights_tensor=None, **kwargs):
                super().__init__(*args, **kwargs)
                self.class_weights_tensor = class_weights_tensor

            def compute_loss(self, model, inputs, return_outputs=False):
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                logits = outputs.logits
                
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
            with get_db() as db:
                taxonomy = db.query(FeatureMaster.feature_key, FeatureMaster.display_name).filter(FeatureMaster.is_active.is_(True)).all()
                feature_display_names = {t.feature_key: t.display_name for t in taxonomy}

            labels_df = self._load_inputs()
            if self.limit:
                labels_df = labels_df.head(self.limit)

            examples_df, class_weights = build_sentiment_examples(labels_df, feature_display_names)

            label_counts = examples_df["label"].value_counts()
            can_stratify = (label_counts >= 2).all() and examples_df["label"].nunique() > 1

            train_df, val_df = train_test_split(
                examples_df,
                test_size=self.split_config.val_size,
                random_state=self.split_config.random_state,
                stratify=examples_df["label"] if can_stratify else None,
            )
            if not can_stratify:
                logging.info(
                    "AspectSentimentTrainer: skipping stratified split "
                    "(a sentiment class has <2 examples), using plain random split"
                )
            logging.info(f"AspectSentimentTrainer: train={len(train_df)} val={len(val_df)}")

            tokenizer = AutoTokenizer.from_pretrained(self.ml_config.base_model_name)
            base_model = AutoModelForSequenceClassification.from_pretrained(
                self.ml_config.base_model_name,
                num_labels=len(SENTIMENT_LABELS),
            )
            peft_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=8,
                lora_alpha=16,
                lora_dropout=0.1
            )
            model = get_peft_model(base_model, peft_config)
            model.print_trainable_parameters()

            train_dataset = AspectSentimentDataset(
                train_df["text_a"].tolist(),
                train_df["text_b"].tolist(),
                train_df["label"].tolist(),
                tokenizer,
                max_length=self.ml_config.max_length,
            )
            val_dataset = AspectSentimentDataset(
                val_df["text_a"].tolist(),
                val_df["text_b"].tolist(),
                val_df["label"].tolist(),
                tokenizer,
                max_length=self.ml_config.max_length,
            )

            def compute_metrics(eval_pred):
                logits, true_labels = eval_pred
                return compute_sentiment_metrics(logits, true_labels)

            output_dir = str(self.settings.resolve_path(self.settings.model_artifact_dir) / "aspect_sentiment" / "checkpoints")

            if torch.backends.mps.is_available():
                os.environ["ACCELERATE_TORCH_DEVICE"] = "mps"
                
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=self.ml_config.num_epochs,
                per_device_train_batch_size=self.ml_config.batch_size,
                per_device_eval_batch_size=self.ml_config.batch_size,
                learning_rate=self.ml_config.learning_rate,
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
            logging.info(f"AspectSentimentTrainer: training device={training_args.device}")

            class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

            trainer = WeightedSentimentTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
                class_weights_tensor=class_weights_tensor,
            )

            mlflow.set_tracking_uri(self.settings.mlflow_tracking_uri)
            mlflow.set_experiment(self.settings.mlflow_experiment_name)

            with mlflow.start_run(run_name="aspect_sentiment_classifier") as run:
                mlflow.log_params(
                    {
                        "task": "aspect_sentiment",
                        "base_model": self.ml_config.base_model_name,
                        "num_labels": len(SENTIMENT_LABELS),
                        "sentiment_labels": SENTIMENT_LABELS,
                        "max_length": self.ml_config.max_length,
                        "batch_size": self.ml_config.batch_size,
                        "num_epochs": self.ml_config.num_epochs,
                        "learning_rate": self.ml_config.learning_rate,
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

                model_save_dir = str(self.settings.resolve_path(self.settings.model_artifact_dir) / "aspect_sentiment" / "final")
                trainer.save_model(model_save_dir)
                tokenizer.save_pretrained(model_save_dir)

                meta = {
                    "base_model": self.ml_config.base_model_name,
                    "sentiment_labels": SENTIMENT_LABELS,  
                    "max_length": self.ml_config.max_length,
                    "mlflow_run_id": run.info.run_id,
                }
                meta_path = f"{model_save_dir}/aspect_sentiment_meta.json"
                with open(meta_path, "w") as f:
                    json.dump(meta, f, indent=2)

                try:
                    mlflow.log_artifacts(model_save_dir, artifact_path="model")
                except Exception as e:
                    logging.error(
                        "AspectSentimentTrainer: model was saved locally but "
                        f"MLflow artifact upload failed (non-fatal): {e}"
                    )

                self._register_model_run(run.info.run_id, ModelTask.aspect_sentiment, final_metrics)

                logging.info(f"AspectSentimentTrainer: complete. Run ID: {run.info.run_id}")
                logging.info(f"Final metrics: {final_metrics}")

            return final_metrics
        except FixFirstException:
            raise
        except Exception as e:
            raise FixFirstException(e, sys) from e


def train_aspect_sentiment_model(limit: int = None) -> Dict:
    """Backward compatible function."""
    return AspectSentimentTrainer(limit=limit).train()