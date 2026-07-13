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
        """Load train and val aspect_category JSONL files."""
        fmt_dir = self.settings.resolve_path(self.settings.data_training_format_dir)
        train_path = fmt_dir / "train" / "aspect_category.jsonl"
        val_path   = fmt_dir / "val"   / "aspect_category.jsonl"
        for p in (train_path, val_path):
            if not p.exists():
                raise FixFirstException(
                    f"Training-format file missing: {p} — run `make preprocess` first.", sys
                )
        def _read_jsonl(path) -> pd.DataFrame:
            import json
            records = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
            return pd.DataFrame(records)
        return _read_jsonl(train_path), _read_jsonl(val_path)

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
            from peft import get_peft_model, LoraConfig, TaskType

            # --- CUSTOM TRAINER WITH STANDARD BCE LOSS ---
            class WeightedTrainer(Trainer):
                def __init__(self, *args, pos_weights_tensor=None, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.pos_weights_tensor = pos_weights_tensor

                def compute_loss(self, model, inputs, return_outputs=False):
                    labels = inputs.pop("labels")
                    outputs = model(**inputs)
                    logits = outputs.logits
                    
                    weights = None
                    if self.pos_weights_tensor is not None:
                        weights = self.pos_weights_tensor.to(logits.device)
                        
                    loss_fct = nn.BCEWithLogitsLoss(pos_weight=weights)
                    loss = loss_fct(
                        logits.view(-1, logits.shape[-1]), 
                        labels.view(-1, logits.shape[-1])
                    )
                    return (loss, outputs) if return_outputs else loss
            # ----------------------------------------

            train_df, val_df = self._load_inputs()

            import json as _json
            import numpy as np

            # Load the canonical vocab produced by the preprocessing pipeline
            fmt_dir = self.settings.resolve_path(self.settings.data_training_format_dir)
            vocab_path = fmt_dir / "label_vocab.json"
            if vocab_path.exists():
                with vocab_path.open() as _f:
                    vocab_data = _json.load(_f)
                LABEL_VOCAB   = vocab_data["label_vocab"]
                label_index   = vocab_data["label_to_idx"]
            else:
                # Fall back to sorted unique labels across the training set
                all_labels = sorted({l for row in train_df["labels"] for l in (row if isinstance(row, list) else [])})
                label_index = build_label_index(all_labels)
                LABEL_VOCAB = [k for k, _ in sorted(label_index.items(), key=lambda x: x[1])]

            label_names = [k for k, _ in sorted(label_index.items(), key=lambda x: x[1])]
            n_labels = len(label_names)

            def _rows_to_examples(df: pd.DataFrame) -> pd.DataFrame:
                rows = []
                pos_counts = np.zeros(n_labels, dtype=np.float32)
                for _, row in df.iterrows():
                    lbls = row["labels"] if isinstance(row["labels"], list) else []
                    vec = np.zeros(n_labels, dtype=np.float32)
                    for lbl in lbls:
                        idx = label_index.get(lbl)
                        if idx is not None:
                            vec[idx] = 1.0
                            pos_counts[idx] += 1.0
                    rows.append({"review_text": row["text"], "labels": vec})
                return pd.DataFrame(rows), pos_counts

            train_examples, pos_counts = _rows_to_examples(train_df)
            val_examples, _            = _rows_to_examples(val_df)

            if self.limit is not None:
                train_examples = train_examples.head(self.limit)

            total = len(train_examples)
            MAX_WEIGHT_CAP = 15.0
            pos_weights = {}
            for feat, idx in label_index.items():
                neg = total - pos_counts[idx]
                pos_weights[feat] = min(float(neg / (pos_counts[idx] + 1e-5)), MAX_WEIGHT_CAP)

            if len(train_examples) < 2:
                raise FixFirstException("At least two labeled reviews are required for training.", sys)

            logging.info(
                f"AspectCategoryTrainer: train={len(train_examples)} val={len(val_examples)} "
                f"labels={len(label_names)}"
            )

            tokenizer = AutoTokenizer.from_pretrained(self.ml_config.base_model_name)
            base_model = AutoModelForSequenceClassification.from_pretrained(
                self.ml_config.base_model_name,
                num_labels=len(label_names),
                problem_type="multi_label_classification",
                id2label={index: name for name, index in label_index.items()},
                label2id=label_index,
            )
            peft_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=16,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=[
                    "query_proj",
                    "key_proj",
                    "value_proj",
                    "dense",
                ],
                modules_to_save=["pooler", "classifier"]
            )
            model = get_peft_model(base_model, peft_config)
            model.print_trainable_parameters()
            
            train_dataset = AspectCategoryDataset(
                train_examples["review_text"].tolist(),
                train_examples["labels"].tolist(),
                tokenizer,
                self.ml_config.max_length,
            )
            val_dataset = AspectCategoryDataset(
                val_examples["review_text"].tolist(),
                val_examples["labels"].tolist(),
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
                learning_rate=2e-4, # Boosted learning rate for PEFT multi-label training
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

            try:
                mlflow.set_tracking_uri(self.settings.mlflow_tracking_uri)
                mlflow.set_experiment(self.settings.mlflow_experiment_name)
            except Exception as exc:
                logging.warning(f"Failed to connect to MLflow server: {exc}. Falling back to local file tracking.")
                mlflow.set_tracking_uri("file:./mlruns")
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
                        "train_size": len(train_examples),
                        "val_size": len(val_examples),
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
                logging.info("Skipping model registration as per file-based training requirement.")

            logging.info(f"AspectCategoryTrainer: complete. Run ID: {run.info.run_id}")
            return final_metrics
        except FixFirstException:
            raise
        except Exception as exc:
            raise FixFirstException(exc, sys) from exc


def train_aspect_category_model(limit: int = None) -> Dict:
    """Backward compatible function."""
    return AspectCategoryTrainer(limit=limit).train()