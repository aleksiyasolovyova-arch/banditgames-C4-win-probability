"""
Train Connect4 Win Probability Model

Predicts:
P(LOSS), P(DRAW), P(WIN) for the side to move.

Designed for:
- Self-play data
- Future live AI vs human games

Includes:
- XGBoost / LightGBM / RandomForest
- Proper probabilistic metrics
- MLflow experiment tracking
"""
import os

import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    f1_score,
    classification_report,
    confusion_matrix,
    brier_score_loss
)
import joblib
import logging
from pathlib import Path
import json
from datetime import datetime
import sys

from tensorboard_logger import WinProbTensorBoardLogger

sys.path.insert(0, str(Path(__file__).parent))
from preprocessing import Connect4WinProbPreprocessor

# MLflow integration
try:
    from mlflow_tracker import MLflowTracker
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logging.warning("MLflow not available - experiment tracking disabled")

logger = logging.getLogger(__name__)


class WinProbabilityTrainer:
    """Trainer for Connect4 win probability models"""

    def __init__(self, model_type: str = "xgboost"):
        self.model_type = model_type
        self.model = None
        self.metrics = {}

    # ------------------------------------------------------------------
    # Model creation
    # ------------------------------------------------------------------
    def create_model(self, **kwargs):
        if self.model_type == "xgboost":
            return xgb.XGBClassifier(
                n_estimators=kwargs.get("n_estimators", 400),
                max_depth=kwargs.get("max_depth", 8),
                learning_rate=kwargs.get("learning_rate", 0.05),
                subsample=kwargs.get("subsample", 0.8),
                colsample_bytree=kwargs.get("colsample_bytree", 0.8),
                objective="multi:softprob",
                num_class=3,
                eval_metric="mlogloss",
                random_state=42,
                n_jobs=-1,
                tree_method="hist"
            )

        elif self.model_type == "lightgbm":
            import lightgbm as lgb
            return lgb.LGBMClassifier(
                n_estimators=kwargs.get("n_estimators", 400),
                max_depth=kwargs.get("max_depth", 8),
                learning_rate=kwargs.get("learning_rate", 0.05),
                num_leaves=kwargs.get("num_leaves", 31),
                objective="multiclass",
                num_class=3,
                random_state=42,
                n_jobs=-1
            )

        elif self.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=kwargs.get("n_estimators", 300),
                max_depth=kwargs.get("max_depth", 12),
                min_samples_split=kwargs.get("min_samples_split", 10),
                random_state=42,
                n_jobs=-1
            )

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        **model_params
    ):
        logger.info(f"Training {self.model_type} win-probability model...")
        logger.info(f"Train samples: {len(X_train)}, Features: {X_train.shape[1]}")

        self.model = self.create_model(**model_params)

        if X_val is not None and self.model_type == "xgboost":
            self.model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train)

        logger.info("Training complete!")
        return self.model

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        logger.info("Evaluating win probability model...")

        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)

        # Core metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        logloss = log_loss(y_test, y_proba, labels=[0, 1, 2])

        # Brier score (for calibration)
        brier = np.mean([
            brier_score_loss((y_test == c).astype(int), y_proba[:, c])
            for c in range(3)
        ])

        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(
            y_test,
            y_pred,
            labels=[0, 1, 2],
            zero_division=0,
            output_dict=True
        )

        self.metrics = {
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "log_loss": logloss,
            "brier_score": brier,
            "confusion_matrix": cm.tolist(),
            "classification_report": report
        }

        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"F1 Macro: {f1_macro:.4f}")
        logger.info(f"Log Loss: {logloss:.4f}")
        logger.info(f"Brier Score: {brier:.4f}")

        return self.metrics

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    def save_model(self, output_dir: str, version: str = None):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_str = f"_{version}" if version else f"_{timestamp}"

        model_path = output_path / f"{self.model_type}_winprob{version_str}.joblib"
        metrics_path = output_path / f"metrics{version_str}.json"

        joblib.dump(self.model, model_path)
        with open(metrics_path, "w") as f:
            json.dump(self.metrics, f, indent=2)

        logger.info(f"Saved model to {model_path}")
        logger.info(f"Saved metrics to {metrics_path}")

        return model_path, metrics_path


# ----------------------------------------------------------------------
# Full training pipeline
# ----------------------------------------------------------------------
def train_connect4_winprob(
    dataset_path: str,
    model_type: str = "xgboost",
    self_play: bool = True,
    output_dir: str = "models",
    version: str = None,
    **model_params
):
    logger.info("=" * 80)
    logger.info("CONNECT4 WIN PROBABILITY TRAINING")
    logger.info("=" * 80)
    logger.info(f"Dataset: {dataset_path}")
    logger.info(f"Model type: {model_type}")
    logger.info(f"Self-play mode: {self_play}")

    # Preprocessing
    preprocessor = Connect4WinProbPreprocessor(self_play=self_play)
    data = preprocessor.preprocess_pipeline(
        dataset_path=dataset_path,
        test_size=0.2,
        val_size=0.1,
        random_state=42,
        output_dir=f"{output_dir}/preprocessing"
    )

    # Training
    trainer = WinProbabilityTrainer(model_type=model_type)
    trainer.train(
        X_train=data["X_train"],
        y_train=data["y_train"],
        X_val=data["X_val"],
        y_val=data["y_val"],
        **model_params
    )

    # Evaluation
    metrics = trainer.evaluate(
        X_test=data["X_test"],
        y_test=data["y_test"]
    )
    tb = WinProbTensorBoardLogger(
        log_dir="tensorboard-logs",
        experiment_name="connect4-winprob"
    )

    tb.log_test_metrics(metrics)
    tb.log_confusion_matrix(
        np.array(metrics["confusion_matrix"]),
        class_names=["LOSS", "DRAW", "WIN"]
    )
    tb.log_prediction_confidence(
        trainer.model.predict_proba(data["X_test"])
    )
    tb.log_outcome_distribution(data["y_test"])
    tb.close()

    # MLflow logging
    if MLFLOW_AVAILABLE:
        tracker = MLflowTracker("connect4-win-probability")
        tracker.log_params(model_params)
        tracker.log_metrics(metrics)

    # Save
    model_path, metrics_path = trainer.save_model(
        output_dir=f"{output_dir}/{model_type}",
        version=version
    )

    logger.info("=" * 80)
    logger.info("TRAINING COMPLETE ✅")
    logger.info("=" * 80)

    return {
        "model": trainer.model,
        "metrics": metrics,
        "preprocessor": preprocessor,
        "model_path": model_path,
        "metrics_path": metrics_path
    }


# ----------------------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    results = train_connect4_winprob(
        dataset_path="data/dataset_v1.parquet",
        model_type="xgboost",
        self_play=True,
        output_dir="models/winprob_v1",
        version="v1",
        n_estimators=400,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8
    )

    print("\n" + "=" * 80)
    print("FINAL RESULTS — WIN PROBABILITY")
    print("=" * 80)
    print(f"Accuracy: {results['metrics']['accuracy']:.4f}")
    print(f"F1 Macro: {results['metrics']['f1_macro']:.4f}")
    print(f"Log Loss: {results['metrics']['log_loss']:.4f}")
    print(f"Brier Score: {results['metrics']['brier_score']:.4f}")
