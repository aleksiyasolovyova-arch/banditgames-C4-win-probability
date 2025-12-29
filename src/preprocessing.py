"""
Connect4 Win Probability Preprocessing (NEW SCHEMA ONLY)

Target:
0 = LOSS
1 = DRAW
2 = WIN

Uses ONLY the new dataset schema:
- game_outcome
- game_winner
- player
- policy_col_*
- q_value_col_*
"""

import logging
from pathlib import Path
from typing import Dict, Tuple, List

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class Connect4WinProbPreprocessor:
    def __init__(self, self_play: bool = True):
        self.self_play = self_play
        self.scaler = StandardScaler()
        self.feature_columns: List[str] | None = None

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    def load_dataset(self, dataset_path: str) -> pd.DataFrame:
        logger.info(f"Loading dataset from {dataset_path}")
        df = pd.read_parquet(dataset_path)
        logger.info(f"Loaded {len(df):,} rows")
        logger.info(f"Unique games: {df['gameId'].nunique()}")
        return df

    # ------------------------------------------------------------------
    # Perspective augmentation (SELF-PLAY ONLY)
    # ------------------------------------------------------------------
    def augment_selfplay_perspective(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["perspective"] = "original"

        mirror = df.copy()
        mirror["perspective"] = "mirrored"

        # Swap board encoding (1 <-> 2)
        board_before = [c for c in df.columns if c.startswith("board_before_r")]
        board_after = [c for c in df.columns if c.startswith("board_after_r")]

        for cols in (board_before, board_after):
            arr = mirror[cols].to_numpy()
            arr = np.where(arr == 1, 2, np.where(arr == 2, 1, arr))
            mirror[cols] = arr

        return pd.concat([df, mirror], ignore_index=True)

    # ------------------------------------------------------------------
    # Target (NEW SCHEMA, OLD LOGIC)
    # ------------------------------------------------------------------
    def build_outcome_target(self, df: pd.DataFrame) -> pd.Series:
        y = pd.Series(0, index=df.index, dtype=int)  # LOSS default

        outcome = df["game_outcome"].astype(str).str.upper().str.strip()
        winner = df["game_winner"].astype(str).str.lower().str.strip()

        # Draws
        y[outcome == "DRAW"] = 1

        if self.self_play:
            non_draw = outcome != "DRAW"
            y[non_draw & (df["perspective"] == "original")] = 0
            y[non_draw & (df["perspective"] == "mirrored")] = 2
        else:
            non_draw = outcome != "DRAW"
            y[non_draw & winner.str.startswith("ai")] = 2

        return y

    # ------------------------------------------------------------------
    # Clean
    # ------------------------------------------------------------------
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Cleaning dataset...")

        # Required columns
        required = [
            "gameId", "moveIndex", "game_outcome", "game_winner"
        ]
        for c in required:
            if c not in df.columns:
                raise ValueError(f"Missing required column: {c}")

        # Fill policy / q_value
        fill_cols = [c for c in df.columns if c.startswith(("policy_col_", "q_value_col_"))]
        df[fill_cols] = df[fill_cols].fillna(0)

        df = df.dropna(subset=["moveIndex"])
        return df

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        board_cols = [f"board_before_r{r}c{c}" for r in range(6) for c in range(7)]
        board = df[board_cols].to_numpy()

        df["player1_pieces"] = (board == 1).sum(axis=1)
        df["player2_pieces"] = (board == 2).sum(axis=1)

        center_cols = [f"board_before_r{r}c{c}" for r in range(6) for c in [2, 3, 4]]
        center = df[center_cols].to_numpy()
        df["center_control_p1"] = (center == 1).sum(axis=1)
        df["center_control_p2"] = (center == 2).sum(axis=1)

        df["move_number"] = df["moveIndex"].astype(int)

        # Policy entropy
        policy_cols = [f"policy_col_{i}" for i in range(7)]
        probs = df[policy_cols].to_numpy(dtype=float) + 1e-10
        probs = probs / probs.sum(axis=1, keepdims=True)

        df["policy_entropy"] = -(probs * np.log(probs)).sum(axis=1)
        df["policy_max"] = probs.max(axis=1)

        # Q-value summary
        q_cols = [f"q_value_col_{i}" for i in range(7)]
        q = df[q_cols].to_numpy(dtype=float)

        df["qvalue_mean"] = q.mean(axis=1)
        df["qvalue_range"] = q.max(axis=1) - q.min(axis=1)

        return df

    # ------------------------------------------------------------------
    # Feature selection (STRICT, NO LEAKAGE)
    # ------------------------------------------------------------------
    def select_features(self, df: pd.DataFrame):
        board_cols = [f"board_before_r{r}c{c}" for r in range(6) for c in range(7)]
        policy_cols = [f"policy_col_{i}" for i in range(7)]
        q_cols = [f"q_value_col_{i}" for i in range(7)]

        engineered = [
            "player1_pieces", "player2_pieces",
            "center_control_p1", "center_control_p2",
            "move_number",
            "policy_entropy", "policy_max",
            "qvalue_mean", "qvalue_range",
        ]

        feature_cols = board_cols + policy_cols + q_cols + engineered
        self.feature_columns = feature_cols

        X = df[feature_cols]
        y = self.build_outcome_target(df)
        game_ids = df["gameId"]

        logger.info(f"X shape: {X.shape}")
        logger.info(f"Target distribution:\n{y.value_counts().sort_index()}")

        return X, y, game_ids

    # ------------------------------------------------------------------
    # Split + scale
    # ------------------------------------------------------------------
    def preprocess_pipeline(
        self,
        dataset_path: str,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
    ) -> Dict:

        logger.info("=" * 80)
        logger.info("WIN PROBABILITY PREPROCESSING START (NEW SCHEMA)")
        logger.info("=" * 80)

        df = self.load_dataset(dataset_path)

        if self.self_play:
            df = self.augment_selfplay_perspective(df)

        df = self.clean_data(df)
        df = self.engineer_features(df)

        X, y, game_ids = self.select_features(df)

        rng = np.random.default_rng(random_state)
        games = game_ids.unique()
        rng.shuffle(games)

        n_test = int(len(games) * test_size)
        n_val = int((len(games) - n_test) * val_size)

        test_g = games[:n_test]
        val_g = games[n_test:n_test + n_val]
        train_g = games[n_test + n_val:]

        train = game_ids.isin(train_g)
        val = game_ids.isin(val_g)
        test = game_ids.isin(test_g)

        X_train, X_val, X_test = X[train], X[val], X[test]
        y_train, y_val, y_test = y[train], y[val], y[test]

        logger.info(f"TRAIN dist:\n{y_train.value_counts().sort_index()}")
        logger.info(f"VAL dist:\n{y_val.value_counts().sort_index()}")
        logger.info(f"TEST dist:\n{y_test.value_counts().sort_index()}")

        self.scaler.fit(X_train)

        return {
            "X_train": self.scaler.transform(X_train),
            "X_val": self.scaler.transform(X_val),
            "X_test": self.scaler.transform(X_test),
            "y_train": y_train.values,
            "y_val": y_val.values,
            "y_test": y_test.values,
            "feature_names": self.feature_columns,
            "preprocessor": self,
            "df_processed": df,
            "y_full": y.values,
        }

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
