"""
Exploratory Data Analysis (EDA) for Connect4 Win Probability (Self-Play)

This EDA mirrors the structure of the policy-imitation EDA, but:
- Handles self-play data
- Applies perspective augmentation
- Defines win/draw/loss relative to the side to move
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 10


class Connect4WinProbEDA:
    """EDA for Connect4 win probability (self-play data)"""

    def __init__(self, output_dir: str = "./eda_reports_winprob"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"EDA output directory: {self.output_dir}")

    # ======================================================================
    # Perspective augmentation (CRITICAL FOR SELF-PLAY)
    # ======================================================================
    def augment_selfplay_perspective(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Duplicate dataset with mirrored perspective:
        - swap current_player
        - swap board encoding (1 <-> 2)
        - mark mirrored rows
        """
        df = df.copy()
        df["perspective"] = "original"

        mirror = df.copy()
        mirror["perspective"] = "mirrored"

        # Swap current player
        mirror["current_player"] = mirror["current_player"].map({1: 2, 2: 1})

        # Swap board encoding
        board_cols = [c for c in df.columns if c.startswith("board_before_")]
        for col in board_cols:
            mirror[col] = mirror[col].map({1: 2, 2: 1, 0: 0})

        return pd.concat([df, mirror], ignore_index=True)

    # ======================================================================
    # Target construction (self-play)
    # ======================================================================
    def build_outcome_target(self, df: pd.DataFrame) -> pd.Series:
        """
        Self-play win probability target.

        2 = WIN  (side to move eventually wins)
        1 = DRAW
        0 = LOSS
        """
        y = pd.Series(0, index=df.index, dtype=int)

        # Draws
        y[df["winner"].isna()] = 1

        # In self-play:
        # original perspective = losing side
        # mirrored perspective = winning side
        y[(df["winner"].notna()) & (df["perspective"] == "mirrored")] = 2

        return y

    # ======================================================================
    # Main EDA report
    # ======================================================================
    def generate_full_report(
        self,
        df: pd.DataFrame,
        X_train: Optional[pd.DataFrame] = None,
        y_train: Optional[pd.Series] = None,
        save_plots: bool = True
    ) -> Dict:

        logger.info("=" * 80)
        logger.info("GENERATING WIN PROBABILITY EDA REPORT (SELF-PLAY)")
        logger.info("=" * 80)

        # Augment dataset FIRST
        df = self.augment_selfplay_perspective(df)

        report = {}

        logger.info("1. Dataset Overview")
        report["overview"] = self.dataset_overview(df)

        logger.info("2. Target Variable Analysis")
        report["target_analysis"] = self.analyze_target(df, save_plots)

        logger.info("3. Feature Statistics")
        report["feature_stats"] = self.feature_statistics(df)

        logger.info("4. MCTS Behavior Analysis")
        report["mcts_analysis"] = self.analyze_mcts_behavior(df, save_plots)

        logger.info("5. Game Phase Analysis")
        report["phase_analysis"] = self.analyze_game_phases(df, save_plots)

        logger.info("6. Board State Analysis")
        report["board_analysis"] = self.analyze_board_states(df, save_plots)

        logger.info("7. Correlation Analysis")
        if X_train is not None and y_train is not None:
            report["correlation"] = self.correlation_analysis(X_train, y_train, save_plots)

        logger.info("8. Data Quality Checks")
        report["quality"] = self.data_quality_checks(df)

        self.save_text_report(report)

        logger.info("=" * 80)
        logger.info("EDA COMPLETE")
        logger.info("=" * 80)

        return report

    # ======================================================================
    # 1. Dataset overview
    # ======================================================================
    def dataset_overview(self, df: pd.DataFrame) -> Dict:
        overview = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "unique_games": df["gameId"].nunique() if "gameId" in df.columns else None,
            "avg_moves_per_game": len(df) / df["gameId"].nunique(),
            "missing_values": int(df.isnull().sum().sum()),
            "duplicated_rows": int(df.duplicated().sum())
        }

        for k, v in overview.items():
            logger.info(f"{k:25s}: {v}")

        return overview

    # ======================================================================
    # 2. Target analysis
    # ======================================================================
    def analyze_target(self, df: pd.DataFrame, save_plots: bool = True) -> Dict:
        y = self.build_outcome_target(df)

        counts = y.value_counts().sort_index()
        pct = y.value_counts(normalize=True).sort_index() * 100

        labels = {0: "LOSS", 1: "DRAW", 2: "WIN"}

        for k in labels:
            logger.info(f"{labels[k]:>4}: {counts.get(k,0):,} ({pct.get(k,0):.2f}%)")

        if save_plots:
            plt.bar(
                [labels[k] for k in labels],
                [counts.get(k, 0) for k in labels],
                color=["firebrick", "gray", "steelblue"],
                alpha=0.8
            )
            plt.ylabel("Count")
            plt.title("Outcome Distribution (Self-Play, Side to Move)")
            plt.tight_layout()
            plt.savefig(self.output_dir / "target_distribution.png", dpi=300)
            plt.close()

        return {
            "value_counts": counts.to_dict(),
            "percentages": pct.to_dict()
        }

    # ======================================================================
    # 3. Feature statistics
    # ======================================================================
    def feature_statistics(self, df: pd.DataFrame) -> Dict:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        logger.info(f"Numeric features: {len(numeric_cols)}")
        return {
            "numeric_feature_count": len(numeric_cols),
            "summary": df[numeric_cols].describe().to_dict()
        }

    # ======================================================================
    # 4. MCTS behavior analysis
    # ======================================================================
    def analyze_mcts_behavior(self, df: pd.DataFrame, save_plots: bool = True) -> Dict:
        visit_cols = [c for c in df.columns if c.startswith("mcts_visits_")]

        if not visit_cols:
            logger.warning("No MCTS columns found")
            return {}

        df = df.copy()
        df["total_visits"] = df[visit_cols].sum(axis=1)

        visit_probs = df[visit_cols].div(df["total_visits"], axis=0).fillna(0)
        df["visit_entropy"] = -(visit_probs * np.log(visit_probs + 1e-10)).sum(axis=1)

        if save_plots:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            axes[0].hist(df["total_visits"], bins=50)
            axes[0].set_title("Total MCTS Visits")

            axes[1].hist(df["visit_entropy"], bins=50, color="coral")
            axes[1].set_title("MCTS Visit Entropy")

            plt.tight_layout()
            plt.savefig(self.output_dir / "mcts_behavior.png", dpi=300)
            plt.close()

        return {
            "avg_total_visits": float(df["total_visits"].mean()),
            "avg_visit_entropy": float(df["visit_entropy"].mean())
        }

    # ======================================================================
    # 5. Game phase analysis
    # ======================================================================
    def analyze_game_phases(self, df: pd.DataFrame, save_plots: bool = True) -> Dict:
        if "moveIndex" not in df.columns:
            return {}

        df = df.copy()
        df["outcome"] = self.build_outcome_target(df)

        df["phase_category"] = pd.cut(
            df["moveIndex"],
            bins=[0, 10, 25, float("inf")],
            labels=["Early (1–10)", "Mid (11–25)", "Late (26+)"],
            include_lowest=True
        )

        phase_outcomes = df.groupby(["phase_category", "outcome"]).size().unstack(fill_value=0)

        if save_plots:
            phase_outcomes.div(phase_outcomes.sum(axis=1), axis=0).plot(
                kind="bar", stacked=True
            )
            plt.title("Outcome Distribution by Game Phase")
            plt.ylabel("Percentage (%)")
            plt.tight_layout()
            plt.savefig(self.output_dir / "game_phases.png", dpi=300)
            plt.close()

        return {"moves_per_phase": df["phase_category"].value_counts().to_dict()}

    # ======================================================================
    # 6. Board state analysis
    # ======================================================================
    def analyze_board_states(self, df: pd.DataFrame, save_plots: bool = True) -> Dict:
        board_cols = [c for c in df.columns if c.startswith("board_before_")]
        if not board_cols:
            return {}

        df = df.copy()
        df["board_fullness"] = (df[board_cols] != 0).sum(axis=1)
        df["board_fullness_pct"] = df["board_fullness"] / 42 * 100

        if save_plots:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            axes[0].hist(df["board_fullness_pct"], bins=30)
            axes[0].set_title("Board Fullness (%)")

            axes[1].scatter(df["moveIndex"], df["board_fullness_pct"], s=5, alpha=0.1)
            axes[1].set_title("Board Fullness vs Move Index")

            plt.tight_layout()
            plt.savefig(self.output_dir / "board_states.png", dpi=300)
            plt.close()

        return {
            "avg_board_fullness_pct": float(df["board_fullness_pct"].mean())
        }

    # ======================================================================
    # 7. Correlation analysis
    # ======================================================================
    def correlation_analysis(self, X: pd.DataFrame, y: pd.Series, save_plots: bool = True, top_n: int = 20) -> Dict:
        correlations = {
            col: abs(X[col].corr(y))
            for col in X.columns
            if not np.isnan(X[col].corr(y))
        }

        correlations = dict(sorted(correlations.items(), key=lambda x: x[1], reverse=True))
        top = dict(list(correlations.items())[:top_n])

        if save_plots:
            plt.barh(range(len(top)), list(top.values()))
            plt.yticks(range(len(top)), [k[:40] for k in top.keys()])
            plt.gca().invert_yaxis()
            plt.title("Top Feature Correlations with Outcome")
            plt.tight_layout()
            plt.savefig(self.output_dir / "feature_correlations.png", dpi=300)
            plt.close()

        return {"top_correlations": top}

    # ======================================================================
    # 8. Data quality checks
    # ======================================================================
    def data_quality_checks(self, df: pd.DataFrame) -> Dict:
        return {
            "missing_columns": df.isnull().sum()[df.isnull().sum() > 0].to_dict(),
            "duplicated_rows": int(df.duplicated().sum())
        }

    # ======================================================================
    # Save text report
    # ======================================================================
    def save_text_report(self, report: Dict):
        path = self.output_dir / "eda_summary.txt"
        with open(path, "w") as f:
            f.write("CONNECT4 WIN PROBABILITY — SELF-PLAY EDA SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            for section, content in report.items():
                f.write(f"{section.upper()}\n")
                f.write("-" * 80 + "\n")
                f.write(f"{content}\n\n")
        logger.info(f"Saved text report: {path}")


# ======================================================================
# CLI entry point
# ======================================================================
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python eda.py <dataset_path>")
        sys.exit(1)

    df = pd.read_parquet(sys.argv[1])
    eda = Connect4WinProbEDA()
    eda.generate_full_report(df, save_plots=True)
