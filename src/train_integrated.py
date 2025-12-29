import os
import joblib
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.metrics import brier_score_loss

import mlflow
import mlflow.sklearn

from dotenv import load_dotenv

from src.preprocessing import Connect4WinProbPreprocessor
from src.eda import Connect4WinProbEDA

# TensorBoard is optional (don’t crash if torch is missing)
try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
    TENSORBOARD_AVAILABLE = True
except Exception:
    SummaryWriter = None  # type: ignore
    TENSORBOARD_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _brier_multiclass(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    scores = []
    for c in [0, 1, 2]:
        if np.any(y_true == c):
            scores.append(brier_score_loss((y_true == c).astype(int), y_proba[:, c]))
    return float(np.mean(scores)) if scores else float("nan")


def _remap_labels_for_training(y: np.ndarray):
    """
    XGBClassifier can error if a class is missing in y_train AND labels are non-contiguous.
    Example: y_train has {0,2} (no draws), it expects {0,1} for binary.

    We remap ONLY for training so labels become contiguous:
      present classes sorted -> {0,2} => map {0->0, 2->1}
    Then we can still evaluate with original labels using predict_proba columns.

    Returns:
      y_mapped, forward_map(original->mapped), inverse_map(mapped->original)
    """
    present = np.unique(y)
    forward = {int(orig): int(i) for i, orig in enumerate(present)}
    inverse = {int(i): int(orig) for i, orig in enumerate(present)}
    y_mapped = np.vectorize(lambda v: forward[int(v)])(y).astype(int)
    return y_mapped, forward, inverse


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
            random_state=42
        )
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        return None

    # Sanity logs
    train_classes = np.unique(data["y_train"])
    val_classes = np.unique(data["y_val"])
    test_classes = np.unique(data["y_test"])
    logger.info(f"TRAIN classes present: {train_classes.tolist()}")
    logger.info(f"VAL classes present:   {val_classes.tolist()}")
    logger.info(f"TEST classes present:  {test_classes.tolist()}")

    # --- EDA (authoritative labels only) ---
    try:
        eda_output = out_path / "reports"
        eda = Connect4WinProbEDA(output_dir=str(eda_output))

        # ALWAYS use the labels produced by preprocessing
        y_full = pd.Series(data["y_full"])
        eda.generate_report(y_full, version)

    except Exception as e:
        logger.error(f"EDA generation failed: {e}")

    # 3) Train
    params = {
        "n_estimators": int(os.getenv("XGB_N_ESTIMATORS", "400")),
        "max_depth": int(os.getenv("XGB_MAX_DEPTH", "8")),
        "learning_rate": float(os.getenv("XGB_LEARNING_RATE", "0.05")),
        "subsample": float(os.getenv("XGB_SUBSAMPLE", "0.8")),
        "colsample_bytree": float(os.getenv("XGB_COLSAMPLE", "0.8")),
        "objective": "multi:softprob",
        "num_class": 3,  # we WANT 3 outputs conceptually
        "eval_metric": "mlogloss",
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
    }

    logger.info("Training XGBoost win-prob model...")

    X_train = data["X_train"]
    y_train = data["y_train"]

    # ✅ Critical: handle missing classes in TRAIN
    present = np.unique(y_train)
    if len(present) < 3:
        logger.warning(
            f"TRAIN is missing classes {sorted(set([0,1,2]) - set(present.tolist()))}. "
            f"Will remap labels for training to keep XGBoost happy."
        )
        y_train_mapped, forward_map, inverse_map = _remap_labels_for_training(y_train)

        # Train as binary or smaller-k multiclass internally
        # NOTE: We override objective/num_class based on how many labels exist in TRAIN.
        n_present = len(np.unique(y_train_mapped))
        train_params = dict(params)

        if n_present == 2:
            train_params["objective"] = "binary:logistic"
            train_params.pop("num_class", None)
        else:
            train_params["objective"] = "multi:softprob"
            train_params["num_class"] = n_present

        model = XGBClassifier(**train_params)
        model.fit(X_train, y_train_mapped)

        # For evaluation: we need 3-class probs [LOSS,DRAW,WIN]
        # Build a full 3-column proba with zeros for missing classes.
        def predict_proba_3(X):
            p = model.predict_proba(X)
            # p columns correspond to mapped classes [0..n_present-1]
            proba3 = np.zeros((p.shape[0], 3), dtype=float)
            for mapped_idx in range(p.shape[1]):
                orig_label = inverse_map[mapped_idx]
                proba3[:, orig_label] = p[:, mapped_idx]
            return proba3

        proba = predict_proba_3(data["X_test"])
        preds = np.argmax(proba, axis=1)

    else:
        # Normal 3-class training
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)

        preds = model.predict(data["X_test"])
        proba = model.predict_proba(data["X_test"])

    # 4) Evaluate (robust)
    acc = accuracy_score(data["y_test"], preds)
    ll = log_loss(data["y_test"], proba, labels=[0, 1, 2])
    brier = _brier_multiclass(data["y_test"], proba)

    logger.info(f"Accuracy: {acc:.4f}")
    logger.info(f"LogLoss:  {ll:.4f}")
    logger.info(f"Brier:    {brier:.4f}")

    # -------------------------------
    # WIN-PROBABILITY TELEMETRY (optional)
    # -------------------------------
    win_probs = proba[:, 2]
    mean_win_prob = float(win_probs.mean())
    predicted_win_rate = float((win_probs > 0.5).mean())
    actual_win_rate = float((data["y_test"] == 2).mean())

    logger.info(
        f"WinProb telemetry | mean={mean_win_prob:.3f} "
        f"predicted={predicted_win_rate:.3f} actual={actual_win_rate:.3f}"
    )

    if TENSORBOARD_AVAILABLE:
        tb_log_dir = os.getenv("TENSORBOARD_LOG_DIR", "/workspace/tensorboard_logs")
        writer = SummaryWriter(log_dir=os.path.join(tb_log_dir, "win_probability"))
        writer.add_scalar("Confidence/MeanWinProbability", mean_win_prob, 0)
        writer.add_scalar("Calibration/PredictedWinRate", predicted_win_rate, 0)
        writer.add_scalar("Calibration/ActualWinRate", actual_win_rate, 0)
        writer.close()
    else:
        logger.info("TensorBoard logging skipped (torch/tensorboard not installed).")

    # Label distribution sanity
    unique, counts = np.unique(data["y_test"], return_counts=True)
    test_dist = dict(zip(unique.tolist(), counts.tolist()))
    logger.info(f"TEST LABEL DISTRIBUTION: {test_dist}")
    if len(test_dist) == 1:
        logger.warning(
            "⚠️ Test split contains only ONE class. Metrics will look artificially perfect."
        )

    # 5) Save artifacts locally (names match your winprob naming)
    joblib.dump(model, out_path / f"winprob_model_{version}.joblib")
    joblib.dump(preprocessor, out_path / f"winprob_preprocessor_{version}.joblib")
    logger.info(f"Model and preprocessor saved to {out_path}")

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
