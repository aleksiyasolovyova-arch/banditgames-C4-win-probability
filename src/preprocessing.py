"""
Data Preprocessing Pipeline for Connect4 Win Probability

Supports:
- Self-play data (perspective augmentation ON)
- Live AI vs Human data (perspective augmentation OFF)

Target:
0 = LOSS
1 = DRAW
2 = WIN

WIN is always defined relative to the AI / side-to-move.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging
from pathlib import Path
from typing import Tuple, Dict
import joblib

logger = logging.getLogger(__name__)


class Connect4WinProbPreprocessor:
    """
    Preprocessor for Connect4 win probability.

    Args:
        self_play (bool):
            True  -> self-play data (apply perspective augmentation)
            False -> live AI vs human data (no augmentation)
    """

    def __init__(self, self_play: bool = True):
        self.self_play = self_play
        self.scaler = StandardScaler()
        self.feature_columns = None

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
        """
        Duplicate dataset with mirrored player perspective.
        Used ONLY for self-play data.
        """
        df = df.copy()
        df["perspective"] = "original"

        mirror = df.copy()
        mirror["perspective"] = "mirrored"

        # Swap current player
        mirror["current_player"] = mirror["current_player"].map({1: 2, 2: 1})

        # Swap board encoding (1 <-> 2)
        board_cols = [c for c in df.columns if c.startswith("board_before_")]
        for col in board_cols:
            mirror[col] = mirror[col].map({1: 2, 2: 1, 0: 0})

        return pd.concat([df, mirror], ignore_index=True)

    # ------------------------------------------------------------------
    # Target construction
    # ------------------------------------------------------------------
    def build_outcome_target(self, df: pd.DataFrame) -> pd.Series:
        """
        Build win/draw/loss target.

        Self-play:
            - original perspective = LOSS
            - mirrored perspective = WIN

        Live-play:
            - winner identifies AI
            - WIN if AI eventually wins and it's AI's turn
        """
        y = pd.Series(0, index=df.index, dtype=int)

        # Draws
        y[df["winner"].isna()] = 1

        if self.self_play:
            # Mirrored perspective corresponds to winning side
            y[(df["winner"].notna()) & (df["perspective"] == "mirrored")] = 2
        else:
            # Live play: winner must identify AI explicitly
            # ASSUMPTION: winner == "AI" when AI wins
            y[(df["winner"] == "AI")] = 2

        return y

    # ------------------------------------------------------------------
    # Clean
    # ------------------------------------------------------------------
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean dataset.

        Note:
        - No action filtering (not a policy model)
        """
        logger.info("Cleaning dataset...")
        initial_rows = len(df)

        # Fill MCTS columns with 0
        mcts_cols = [
            c for c in df.columns
            if c.startswith(("mcts_visits_", "mcts_qvalue_", "mcts_prob_"))
        ]
        if mcts_cols:
            df[mcts_cols] = df[mcts_cols].fillna(0)

        # Drop rows missing critical values
        df = df.dropna(subset=["current_player", "moveIndex"])

        logger.info(f"Cleaned: {initial_rows} → {len(df)} rows")
        return df

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Engineering features...")
        df = df.copy()

        board_cols = [c for c in df.columns if c.startswith("board_before_")]
        board_matrix = df[board_cols].values

        # Piece counts
        df["player1_pieces"] = (board_matrix == 1).sum(axis=1)
        df["player2_pieces"] = (board_matrix == 2).sum(axis=1)

        # Center control
        center_cols = [f"board_before_r{r}c{c}" for r in range(6) for c in [2, 3, 4]]
        center_matrix = df[center_cols].values
        df["center_control_p1"] = (center_matrix == 1).sum(axis=1)
        df["center_control_p2"] = (center_matrix == 2).sum(axis=1)

        # Move number
        df["move_number"] = df["moveIndex"]

        # MCTS statistics
        visit_cols = [f"mcts_visits_col{i}" for i in range(7) if f"mcts_visits_col{i}" in df.columns]
        qvalue_cols = [f"mcts_qvalue_col{i}" for i in range(7) if f"mcts_qvalue_col{i}" in df.columns]

        if visit_cols:
            visits = df[visit_cols].values + 1e-10
            probs = visits / visits.sum(axis=1, keepdims=True)
            df["visit_entropy"] = -(probs * np.log(probs)).sum(axis=1)
            df["top_visit_ratio"] = visits.max(axis=1) / visits.sum(axis=1)

        if qvalue_cols:
            qvalues = df[qvalue_cols].values
            df["qvalue_range"] = qvalues.max(axis=1) - qvalues.min(axis=1)
            df["qvalue_mean"] = qvalues.mean(axis=1)

        return df

    # ------------------------------------------------------------------
    # Feature selection
    # ------------------------------------------------------------------
    def select_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        logger.info("Selecting features...")

        board_cols = [f"board_before_r{r}c{c}" for r in range(6) for c in range(7)]
        mcts_visit_cols = [f"mcts_visits_col{i}" for i in range(7)]
        mcts_qvalue_cols = [f"mcts_qvalue_col{i}" for i in range(7)]
        mcts_prob_cols = [f"mcts_prob_col{i}" for i in range(7)]

        engineered_cols = [
            "player1_pieces", "player2_pieces",
            "center_control_p1", "center_control_p2",
            "move_number", "visit_entropy",
            "qvalue_range", "qvalue_mean",
            "top_visit_ratio",
            "current_player"
        ]

        feature_cols = (
            board_cols +
            mcts_visit_cols +
            mcts_qvalue_cols +
            mcts_prob_cols +
            engineered_cols
        )

        feature_cols = [c for c in feature_cols if c in df.columns]
        self.feature_columns = feature_cols

        X = df[feature_cols]
        y = self.build_outcome_target(df)
        game_ids = df["gameId"]

        logger.info(f"X shape: {X.shape}")
        logger.info(f"Target distribution:\n{y.value_counts()}")

        return X, y, game_ids

    # ------------------------------------------------------------------
    # GAME-level split (NO LEAKAGE)
    # ------------------------------------------------------------------
    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        game_ids: pd.Series,
        test_size: float,
        val_size: float,
        random_state: int
    ):
        logger.info("Splitting data by GAME (no leakage)...")

        unique_games = game_ids.unique()
        np.random.seed(random_state)
        shuffled = np.random.permutation(unique_games)

        n_test = int(len(unique_games) * test_size)
        n_val = int((len(unique_games) - n_test) * val_size)

        test_games = shuffled[:n_test]
        val_games = shuffled[n_test:n_test + n_val]
        train_games = shuffled[n_test + n_val:]

        train_mask = game_ids.isin(train_games)
        val_mask = game_ids.isin(val_games)
        test_mask = game_ids.isin(test_games)

        return (
            X[train_mask], X[val_mask], X[test_mask],
            y[train_mask], y[val_mask], y[test_mask]
        )

    # ------------------------------------------------------------------
    # Scaling
    # ------------------------------------------------------------------
    def scale_features(self, X_train, X_val, X_test):
        logger.info("Scaling features...")
        self.scaler.fit(X_train)
        return (
            self.scaler.transform(X_train),
            self.scaler.transform(X_val),
            self.scaler.transform(X_test)
        )

    def transform_new_data(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform new (unseen) game states for inference.

        IMPORTANT:
        - Assumes preprocess_pipeline() has already been run
        - Does NOT perform train/val/test split
        - Does NOT apply perspective augmentation automatically
        """

        if self.feature_columns is None:
            raise ValueError(
                "Preprocessor not fitted. "
                "Run preprocess_pipeline() before calling transform_new_data()."
            )

        df = df.copy()

        # Clean
        df = self.clean_data(df)

        # Feature engineering
        df = self.engineer_features(df)

        # Select features
        missing = [c for c in self.feature_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required feature columns: {missing}")

        X = df[self.feature_columns]

        # Scale (using already-fitted scaler)
        X_scaled = self.scaler.transform(X)

        return X_scaled

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------
    def preprocess_pipeline(
        self,
        dataset_path: str,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        output_dir: str = None
    ) -> Dict:

        logger.info("=" * 80)
        logger.info("WIN PROBABILITY PREPROCESSING START")
        logger.info(f"Self-play mode: {self.self_play}")
        logger.info("=" * 80)

        df = self.load_dataset(dataset_path)

        if self.self_play:
            df = self.augment_selfplay_perspective(df)

        df = self.clean_data(df)
        df = self.engineer_features(df)

        X, y, game_ids = self.select_features(df)

        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(
            X, y, game_ids, test_size, val_size, random_state
        )

        X_train_s, X_val_s, X_test_s = self.scale_features(
            X_train, X_val, X_test
        )

        if output_dir:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            joblib.dump(self, out / "preprocessor.joblib")
            logger.info(f"Saved preprocessor to {out / 'preprocessor.joblib'}")

        logger.info("=" * 80)
        logger.info("PREPROCESSING COMPLETE ✅")
        logger.info("=" * 80)

        return {
            "X_train": X_train_s,
            "X_val": X_val_s,
            "X_test": X_test_s,
            "y_train": y_train.values,
            "y_val": y_val.values,
            "y_test": y_test.values,
            "feature_names": self.feature_columns,
            "preprocessor": self
        }
