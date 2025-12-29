"""
TensorBoard Integration for Connect4 Win Probability Model

Logs:
- Training / validation metrics
- Probabilistic evaluation metrics
- Outcome distributions
- Confusion matrix
- Prediction confidence
- Calibration-relevant statistics
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Any, List
from datetime import datetime
import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    logging.warning("TensorBoard not available - install with: pip install tensorboard torch")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WinProbTensorBoardLogger:
    """TensorBoard logger for Connect4 win probability model"""

    def __init__(
        self,
        log_dir: str = "tensorboard-logs",
        experiment_name: str = "connect4-win-prob"
    ):
        if not TENSORBOARD_AVAILABLE:
            raise ImportError("TensorBoard not available")

        self.log_dir = (
            Path(log_dir)
            / experiment_name
            / datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.train_writer = SummaryWriter(self.log_dir / "train")
        self.val_writer = SummaryWriter(self.log_dir / "val")
        self.test_writer = SummaryWriter(self.log_dir / "test")

        logger.info(f"TensorBoard logs: {self.log_dir}")
        logger.info(f"Run TensorBoard with:")
        logger.info(f"  tensorboard --logdir {self.log_dir.parent.parent}")

    # ------------------------------------------------------------------
    # Training / validation logging
    # ------------------------------------------------------------------
    def log_training_metrics(
        self,
        metrics: Dict[str, float],
        step: int,
        phase: str = "train"
    ):
        writer = self.train_writer if phase == "train" else self.val_writer

        for name, value in metrics.items():
            writer.add_scalar(name, value, step)

    # ------------------------------------------------------------------
    # Final evaluation logging
    # ------------------------------------------------------------------
    def log_test_metrics(self, metrics: Dict[str, Any]):
        """
        Log only scalar test metrics to TensorBoard.
        """
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                self.test_writer.add_scalar(f"final/{name}", value, 0)

    # ------------------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------------------
    def log_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: Optional[List[str]] = None,
        step: int = 0
    ):
        import matplotlib.pyplot as plt
        import seaborn as sns

        if class_names is None:
            class_names = ["LOSS", "DRAW", "WIN"]

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
            cbar_kws={"label": "Count"}
        )
        ax.set_xlabel("Predicted Outcome")
        ax.set_ylabel("True Outcome")
        ax.set_title("Confusion Matrix — Win Probability")

        self.test_writer.add_figure("confusion_matrix", fig, step)
        plt.close()

    # ------------------------------------------------------------------
    # Prediction confidence / calibration
    # ------------------------------------------------------------------
    def log_prediction_confidence(
        self,
        y_proba: np.ndarray,
        step: int = 0
    ):
        """
        Logs average predicted probabilities for LOSS / DRAW / WIN.
        Useful to see model confidence and bias.
        """
        mean_probs = y_proba.mean(axis=0)

        labels = ["LOSS", "DRAW", "WIN"]
        for label, value in zip(labels, mean_probs):
            self.test_writer.add_scalar(f"confidence/mean_{label.lower()}", value, step)

    # ------------------------------------------------------------------
    # Outcome distribution
    # ------------------------------------------------------------------
    def log_outcome_distribution(
        self,
        y_true: np.ndarray,
        step: int = 0
    ):
        values, counts = np.unique(y_true, return_counts=True)
        total = counts.sum()

        mapping = {0: "LOSS", 1: "DRAW", 2: "WIN"}
        for v, c in zip(values, counts):
            self.test_writer.add_scalar(
                f"distribution/{mapping[v]}",
                c / total,
                step
            )

    # ------------------------------------------------------------------
    # Hyperparameters
    # ------------------------------------------------------------------
    def log_hyperparameters(
        self,
        hparams: Dict[str, Any],
        metrics: Dict[str, float]
    ):
        clean_hparams = {
            k: v for k, v in hparams.items()
            if isinstance(v, (int, float, str, bool))
        }
        self.train_writer.add_hparams(clean_hparams, metrics)

    # ------------------------------------------------------------------
    # Close
    # ------------------------------------------------------------------
    def close(self):
        self.train_writer.close()
        self.val_writer.close()
        self.test_writer.close()
        logger.info("TensorBoard writers closed")


def create_tensorboard_logger(config: Dict[str, Any]) -> Optional[WinProbTensorBoardLogger]:
    if not TENSORBOARD_AVAILABLE:
        logger.warning("TensorBoard not available")
        return None

    try:
        return WinProbTensorBoardLogger(
            log_dir=config.get("log_dir", "tensorboard-logs"),
            experiment_name=config.get("experiment_name", "connect4-win-prob")
        )
    except Exception as e:
        logger.error(f"Failed to create TensorBoard logger: {e}")
        return None
