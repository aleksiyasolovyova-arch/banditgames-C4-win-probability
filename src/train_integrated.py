import os
import joblib
import logging
from pathlib import Path

import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.metrics import brier_score_loss

import mlflow
import mlflow.sklearn

from dotenv import load_dotenv

from src.preprocessing import Connect4WinProbPreprocessor
from src.eda import Connect4WinProbEDA

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _brier_multiclass(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    scores = []
    for c in [0, 1, 2]:
        if np.any(y_true == c):
            scores.append(brier_score_loss((y_true == c).astype(int), y_proba[:, c]))
    return float(np.mean(scores)) if scores else float("nan")


def train_job(dataset_path: str, output_dir: str, version: str, self_play: bool = True):
    logger.info(f"Starting WIN-PROB training job for {version}")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1) Preprocess (authoritative)
    preprocessor = Connect4WinProbPreprocessor(self_play=self_play)
    try:
        data = preprocessor.preprocess_pipeline(
            dataset_path=dataset_path,
            test_size=0.2,
            val_size=0.1,
            random_state=42,
            output_dir=str(out_path / "preprocessing")
        )
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        return None

    # 2) EDA (on constructed target, not raw df)
    try:
        eda_output = out_path / "reports"
        eda = Connect4WinProbEDA(output_dir=str(eda_output))
        eda.generate_report(data["y_full"], version)
    except Exception as e:
        logger.warning(f"EDA generation failed (ignored): {e}")

    # 3) Train
    params = {
        "n_estimators": int(os.getenv("XGB_N_ESTIMATORS", "400")),
        "max_depth": int(os.getenv("XGB_MAX_DEPTH", "8")),
        "learning_rate": float(os.getenv("XGB_LEARNING_RATE", "0.05")),
        "subsample": float(os.getenv("XGB_SUBSAMPLE", "0.8")),
        "colsample_bytree": float(os.getenv("XGB_COLSAMPLE", "0.8")),
        # still 3-class, because your old logic creates {0,1,2}
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
    }

    logger.info("Training XGBoost win-prob model...")
    model = XGBClassifier(**params)
    model.fit(data["X_train"], data["y_train"])

    # 4) Evaluate
    preds = model.predict(data["X_test"])
    proba = model.predict_proba(data["X_test"])

    acc = accuracy_score(data["y_test"], preds)
    ll = log_loss(data["y_test"], proba, labels=[0, 1, 2])
    brier = _brier_multiclass(data["y_test"], proba)

    logger.info(f"Accuracy: {acc:.4f}")
    logger.info(f"LogLoss:  {ll:.4f}")
    logger.info(f"Brier:    {brier:.4f}")

    # IMPORTANT sanity warning
    unique, counts = np.unique(data["y_test"], return_counts=True)
    test_dist = dict(zip(unique.tolist(), counts.tolist()))
    logger.info(f"TEST LABEL DISTRIBUTION: {test_dist}")
    if len(test_dist) == 1:
        logger.warning(
            "⚠️ Test split contains only ONE class. Metrics will look artificially perfect.\n"
            "This is not a model success — it's a split/data issue."
        )

    # 5) Save artifacts locally
    joblib.dump(model, out_path / f"model_{version}.joblib")
    logger.info(f"Model saved to {out_path}")

    # 6) MLflow (opt-in)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME")

    if tracking_uri:
        try:
            mlflow.set_tracking_uri(tracking_uri)
            if experiment_name:
                mlflow.set_experiment(experiment_name)

            with mlflow.start_run(run_name=version):
                mlflow.log_params(params)
                mlflow.log_param("self_play", self_play)
                mlflow.log_metric("accuracy", acc)
                mlflow.log_metric("log_loss", ll)
                mlflow.log_metric("brier_score", brier)
                mlflow.sklearn.log_model(model, "model")

            logger.info(f"Logged run to MLflow (uri={tracking_uri}, experiment={experiment_name})")
        except Exception as e:
            logger.warning(f"MLflow logging failed (ignored): {e}")
    else:
        logger.info("MLflow disabled (MLFLOW_TRACKING_URI not set).")

    return {"accuracy": acc, "log_loss": ll, "brier_score": brier}


if __name__ == "__main__":
    load_dotenv()

    dataset_path = os.getenv("DATASET_PATH")
    output_dir = os.getenv("MODELS_DIR", "models")
    self_play = os.getenv("SELF_PLAY", "true").lower() == "true"

    if not dataset_path:
        raise ValueError("DATASET_PATH not set. Put it in .env (or export it).")

    version = Path(dataset_path).stem.split("_")[-1]
    train_job(dataset_path=dataset_path, output_dir=output_dir, version=version, self_play=self_play)
