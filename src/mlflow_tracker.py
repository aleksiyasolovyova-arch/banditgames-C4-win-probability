"""
MLflow Integration for Connect4 Win Probability Model

Handles experiment tracking, metrics logging, and model versioning.
"""

import os
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.tracking import MlflowClient

import logging
from typing import Dict, Any, Optional
from datetime import datetime
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLflowTracker:
    """
    Manages MLflow experiment tracking for Connect4 win probability models.
    """

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_name: Optional[str] = None,
        artifact_location: Optional[str] = None
    ):
        # ------------------------------------------------------------
        # ENV-DRIVEN CONFIG (env → args → defaults)
        # ------------------------------------------------------------
        self.tracking_uri = (
            os.getenv("MLFLOW_TRACKING_URI")
            or tracking_uri
            or "http://localhost:5000"
        )

        self.experiment_name = (
            os.getenv("MLFLOW_EXPERIMENT_NAME")
            or experiment_name
            or "connect4-win-probability"
        )

        self.artifact_location = (
            os.getenv("MLFLOW_ARTIFACT_LOCATION")
            or artifact_location
            or "./mlflow-artifacts"
        )

        self.experiment = None
        self.client = None
        self.enabled = False

        # ------------------------------------------------------------
        # HARD STOP: disable unless explicitly enabled
        # ------------------------------------------------------------
        if os.getenv("ENABLE_MLFLOW", "false").lower() != "true":
            logger.info("MLflow disabled (ENABLE_MLFLOW=false)")
            return

        # ------------------------------------------------------------
        # Safe initialization
        # ------------------------------------------------------------
        try:
            mlflow.set_tracking_uri(self.tracking_uri)

            self.experiment = self._get_or_create_experiment()
            self.client = MlflowClient(self.tracking_uri)

            self.enabled = True

            logger.info(f"MLflow tracker initialized at {self.tracking_uri}")
            logger.info(
                f"Using experiment '{self.experiment_name}' "
                f"(ID: {self.experiment.experiment_id})"
            )

        except Exception as e:
            logger.warning(
                "MLflow tracking disabled (server unreachable). "
                f"Reason: {e}"
            )

    # ------------------------------------------------------------------
    # Experiment management
    # ------------------------------------------------------------------
    def _get_or_create_experiment(self):
        experiment = mlflow.get_experiment_by_name(self.experiment_name)

        if experiment is None:
            experiment_id = mlflow.create_experiment(
                name=self.experiment_name,
                artifact_location=self.artifact_location
            )
            experiment = mlflow.get_experiment(experiment_id)
            logger.info(f"Created new experiment: {self.experiment_name}")
        else:
            logger.info(f"Using existing experiment: {self.experiment_name}")

        return experiment

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------
    def start_run(
        self,
        run_name: str,
        tags: Optional[Dict[str, str]] = None
    ):
        if not self.enabled:
            return None

        default_tags = {
            "model_type": "win-probability",
            "timestamp": datetime.utcnow().isoformat(),
            "model_version": os.getenv("MODEL_VERSION", "dev")
        }

        if tags:
            default_tags.update(tags)

        run = mlflow.start_run(
            experiment_id=self.experiment.experiment_id,
            run_name=run_name,
            tags=default_tags
        )

        logger.info(f"Started run '{run_name}' (ID: {run.info.run_id})")
        return run

    def end_run(self):
        if self.enabled:
            mlflow.end_run()

    # ------------------------------------------------------------------
    # Dataset & split logging
    # ------------------------------------------------------------------
    def log_dataset_info(self, dataset_path: str, df: pd.DataFrame):
        if not self.enabled:
            return

        mlflow.log_param(
            "dataset_path",
            os.getenv("DATASET_PATH", dataset_path)
        )
        mlflow.log_param("num_rows", len(df))
        mlflow.log_param("num_features", len(df.columns))

        if "gameId" in df.columns:
            mlflow.log_param("num_games", df["gameId"].nunique())

        if "win" in df.columns:
            mlflow.log_metric("dataset_win_rate", df["win"].mean())

    def log_split_info(self, train_size: int, val_size: int, test_size: int):
        if not self.enabled:
            return

        total = train_size + val_size + test_size

        mlflow.log_param("train_size", train_size)
        mlflow.log_param("val_size", val_size)
        mlflow.log_param("test_size", test_size)

        mlflow.log_param("train_ratio", train_size / total)
        mlflow.log_param("val_ratio", val_size / total)
        mlflow.log_param("test_ratio", test_size / total)

    # ------------------------------------------------------------------
    # Hyperparameters & metrics
    # ------------------------------------------------------------------
    def log_hyperparameters(self, params: Dict[str, Any]):
        if not self.enabled:
            return

        for key, value in params.items():
            mlflow.log_param(key, value)

    def log_test_metrics(self, metrics: Dict[str, float]):
        if not self.enabled:
            return

        for name, value in metrics.items():
            mlflow.log_metric(f"test_{name}", value)

        mlflow.log_dict(metrics, "test_metrics.json")

    # ------------------------------------------------------------------
    # Model logging
    # ------------------------------------------------------------------
    def log_model(
        self,
        model,
        model_type: str = "sklearn",
        registered_model_name: Optional[str] = None
    ):
        if not self.enabled:
            return

        if model_type == "xgboost":
            mlflow.xgboost.log_model(
                model,
                artifact_path="model",
                registered_model_name=registered_model_name
            )
        else:
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                registered_model_name=registered_model_name
            )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def get_best_run(self, metric: str = "test_log_loss"):
        if not self.enabled or not self.client:
            return None

        runs = self.client.search_runs(
            experiment_ids=[self.experiment.experiment_id],
            order_by=[f"metrics.{metric} ASC"],
            max_results=1
        )

        if not runs:
            return None

        best = runs[0]
        return {
            "run_id": best.info.run_id,
            "run_name": best.data.tags.get("mlflow.runName"),
            "metrics": best.data.metrics,
            "params": best.data.params
        }
