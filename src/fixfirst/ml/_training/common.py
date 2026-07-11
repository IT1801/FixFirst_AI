"""Shared utilities and base classes for classifier training."""

import sys
from abc import ABC, abstractmethod
from typing import Dict, Iterable, Any

from fixfirst.config.configuration import ConfigurationManager
from fixfirst.exceptions.exception import FixFirstException
from fixfirst.logging.logger import logging


def build_label_index(labels: Iterable[str]) -> Dict[str, int]:
    """Return a deterministic alphabetical label-to-index mapping."""
    return {label: index for index, label in enumerate(sorted(set(labels)))}


class BaseModelTrainer(ABC):
    """Abstract base class for model trainers."""

    def __init__(self, limit: int = None):
        self.config_manager = ConfigurationManager()
        self.settings = self.config_manager.get_settings()
        self.ml_config = self.config_manager.get_ml_training_config()
        self.split_config = self.config_manager.get_split_config()
        self.limit = limit

    @abstractmethod
    def train(self) -> Dict[str, Any]:
        """Train the model, save it, log it, and return metrics."""
        pass

    def _register_model_run(self, mlflow_run_id: str, task_enum, metrics: Dict) -> None:
        """Register the model run in the database."""
        from fixfirst.core._db.base import get_db
        from fixfirst.core._db.models import ModelRun

        try:
            with get_db() as database:
                database.add(
                    ModelRun(
                        mlflow_run_id=mlflow_run_id,
                        model_name=self.ml_config.base_model_name,
                        model_version="v1",
                        task=task_enum,
                        metrics={
                            key: value for key, value in metrics.items() if isinstance(value, (int, float))
                        },
                    )
                )
            logging.info(f"_register_model_run: logged ModelRun row for {mlflow_run_id}")
        except Exception as exc:
            logging.error(f"_register_model_run: failed to write ModelRun row (non-fatal): {exc}")

