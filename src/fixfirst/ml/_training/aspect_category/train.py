"""Fine-tune the multi-label aspect category classifier."""

import json
import os
import sys
from typing import Dict, Any

import pandas as pd
from sklearn.model_selection import train_test_split

from fixfirst.constants import EXTRACTED_LABELS_FILENAME, EXTRACTED_PROGRESS_FILENAME
from fixfirst.exceptions.exception import FixFirstException
from fixfirst.logging.logger import logging
from fixfirst.ml._training.aspect_category.dataset import (
    AspectCategoryDataset,
    build_category_examples,
)
from fixfirst.ml._training.aspect_category.metrics import compute_metrics_from_logits
from fixfirst.ml._training.common import build_label_index, BaseModelTrainer


class AspectCategoryTrainer(BaseModelTrainer):
    """Trainer class for the aspect category model."""

    def _load_inputs(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        extracted_dir = self.settings.resolve_path(self.settings.data_extracted_labels_dir)
        labels_path = extracted_dir / EXTRACTED_LABELS_FILENAME
        progress_path = extracted_dir / EXTRACTED_PROGRESS_FILENAME
        if not labels_path.exists() or not progress_path.exists():
            raise FixFirstException(
                "Extracted-label files are missing — run `make label` before `make train`.", sys
            )
        return pd.read_parquet(labels_path), pd.read_parquet(progress_path)

    def train(self) -> Dict[str, Any]:
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

            from fixfirst.core._db.base import get_db
            from fixfirst.core._db.models import FeatureMaster, ModelTask

            # --- CUSTOM TRAINER WITH FOCAL LOSS ---
            class FocalLoss(nn.Module):
                def __init__(self, pos_weight=None, gamma=2.0):
                    super().__init__()
                    self.pos_weight = pos_weight
                    self.gamma = gamma

                def forward(self, logits, targets):
                    p = torch.sigmoid(logits)
                    ce_loss = nn.functional.binary_cross_entropy_with_logits(
                        logits, targets, reduction="none", pos_weight=self.pos_weight
                    )
                    p_t = p * targets + (1 - p) * (1 - targets)
                    loss = ce_loss * ((1 - p_t) ** self.gamma)
                    return loss.mean()

            class WeightedTrainer(Trainer):
                def __init__(self, *args, pos_weights_tensor=None, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.pos_weights_tensor = pos_weights_tensor

                def compute_loss(self, model, inputs, return_outputs=False):
                    labels = inputs.pop("labels")
                    outputs = model(**inputs)
                    logits = outputs.logits
                    
                    if self.pos_weights_tensor is not None:
                        self.pos_weights_tensor = self.pos_weights_tensor.to(logits.device)
                        
                    loss_fct = FocalLoss(pos_weight=self.pos_weights_tensor)
                    loss = loss_fct(
                        logits.view(-1, self.model.config.num_labels), 
                        labels.view(-1, self.model.config.num_labels)
                    )
                    return (loss, outputs) if return_outputs else loss
            # ----------------------------------------

            with get_db() as db:
                taxonomy = db.query(FeatureMaster.feature_key).filter(FeatureMaster.is_active.is_(True)).all()
                feature_keys = [item.feature_key for item in taxonomy]
            label_index = build_label_index(feature_keys)
            label_names = [name for name, _ in sorted(label_index.items(), key=lambda item: item[1])]

            labels_df, progress_df = self._load_inputs()
            if self.limit is not None:
                selected_ids = progress_df["review_id"].drop_duplicates().head(self.limit)
                progress_df = progress_df[progress_df["review_id"].isin(selected_ids)]
                labels_df = labels_df[labels_df["review_id"].isin(selected_ids)]

            examples_df, pos_weights = build_category_examples(labels_df, progress_df, feature_keys)
            
            if len(examples_df) < 2:
                raise FixFirstException("At least two labeled reviews are required for training.", sys)

            train_df, val_df = train_test_split(
                examples_df,
                test_size=max(1, int(round(len(examples_df) * self.split_config.val_size))),
                random_state=self.split_config.random_state,
            )
            logging.info(
                f"AspectCategoryTrainer: train={len(train_df)} val={len(val_df)} "
                f"labels={len(label_names)}"
            )

            tokenizer = AutoTokenizer.from_pretrained(self.ml_config.base_model_name)
            model = AutoModelForSequenceClassification.from_pretrained(
                self.ml_config.base_model_name,
                num_labels=len(label_names),
                problem_type="multi_label_classification",
                id2label={index: name for name, index in label_index.items()},
                label2id=label_index,
            )
            
            train_dataset = AspectCategoryDataset(
                train_df["review_text"].tolist(),
                train_df["labels"].tolist(),
                tokenizer,
                self.ml_config.max_length,
            )
            val_dataset = AspectCategoryDataset(
                val_df["review_text"].tolist(),
                val_df["labels"].tolist(),
                tokenizer,
                self.ml_config.max_length,
            )

            def compute_metrics(eval_prediction):
                logits, targets = eval_prediction
                return compute_metrics_from_logits(
                    logits, targets, label_names, threshold=None
                )

            artifact_root = self.settings.resolve_path(self.settings.model_artifact_dir) / "aspect_category"
            if torch.backends.mps.is_available():
                os.environ["ACCELERATE_TORCH_DEVICE"] = "mps"
                
            training_args = TrainingArguments(
                output_dir=str(artifact_root / "checkpoints"),
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
            logging.info(f"AspectCategoryTrainer: training device={training_args.device}")

            weights_list = [pos_weights[name] for name, _ in sorted(label_index.items(), key=lambda item: item[1])]
            pos_weights_tensor = torch.tensor(weights_list, dtype=torch.float32)

            trainer = WeightedTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
                pos_weights_tensor=pos_weights_tensor,
            )

            mlflow.set_tracking_uri(self.settings.mlflow_tracking_uri)
            mlflow.set_experiment(self.settings.mlflow_experiment_name)
            with mlflow.start_run(run_name="aspect_category_classifier") as run:
                mlflow.log_params(
                    {
                        "task": "aspect_category",
                        "base_model": self.ml_config.base_model_name,
                        "num_labels": len(label_names),
                        "max_length": self.ml_config.max_length,
                        "batch_size": self.ml_config.batch_size,
                        "num_epochs": self.ml_config.num_epochs,
                        "learning_rate": self.ml_config.learning_rate,
                        "train_size": len(train_df),
                        "val_size": len(val_df),
                        "prediction_threshold": self.ml_config.prediction_threshold,
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
                
                optimal_threshold = final_metrics.get("eval_optimal_threshold", self.ml_config.prediction_threshold)
                metadata = {
                    "base_model": self.ml_config.base_model_name,
                    "label_index": label_index,
                    "max_length": self.ml_config.max_length,
                    "threshold": optimal_threshold,
                    "mlflow_run_id": run.info.run_id,
                }
                with (model_save_dir / "aspect_category_meta.json").open("w") as file:
                    json.dump(metadata, file, indent=2)
                try:
                    mlflow.log_artifacts(str(model_save_dir), artifact_path="model")
                except Exception as exc:
                    logging.error(
                        "AspectCategoryTrainer: model was saved locally but "
                        f"MLflow artifact upload failed (non-fatal): {exc}"
                    )
                self._register_model_run(run.info.run_id, ModelTask.aspect_category, final_metrics)

            logging.info(f"AspectCategoryTrainer: complete. Run ID: {run.info.run_id}")
            return final_metrics
        except FixFirstException:
            raise
        except Exception as exc:
            raise FixFirstException(exc, sys) from exc


def train_aspect_category_model(limit: int = None) -> Dict:
    """Backward compatible function."""
    return AspectCategoryTrainer(limit=limit).train()