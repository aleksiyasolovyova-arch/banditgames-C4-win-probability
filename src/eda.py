# src/eda.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class Connect4WinProbEDA:
    """
    EDA for Win Probability.

    Expects FINAL labels produced by preprocessing:
      0 = LOSS
      1 = DRAW
      2 = WIN
    """

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(self, y: pd.Series, version: str):
        if not isinstance(y, pd.Series):
            raise TypeError("EDA expects a pandas Series of labels")

        label_map = {0: "LOSS", 1: "DRAW", 2: "WIN"}
        y_named = y.map(label_map)

        plt.figure(figsize=(7, 5))
        sns.countplot(x=y_named, order=["LOSS", "DRAW", "WIN"])
        plt.title(f"Win Probability Target Distribution (Version {version})")
        plt.xlabel("Outcome")
        plt.ylabel("Count")
        plt.tight_layout()

        out_path = self.output_dir / f"target_dist_{version}.png"
        plt.savefig(out_path)
        plt.close()

        logger.info(f"EDA plot saved: {out_path}")
