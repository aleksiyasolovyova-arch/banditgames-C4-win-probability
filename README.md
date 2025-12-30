# Connect4 Win Probability Model

This directory contains the **training, preprocessing, evaluation, and automation pipeline** for the **Connect4 Win Probability model**.

The model estimates the probability that the **current player** will:
- **Lose**
- **Draw**
- **Win**

given a board state and policy/Q-value information.  
Predictions are returned as **calibrated probabilities**, not just class labels.

---

##  Objective

The goal of this model is to provide **state-level win probabilities** for Connect4, enabling:

- Confidence-aware decision making
- Policy evaluation
- Self-play learning feedback
- Deployment-ready probabilistic outputs

### Target Encoding

| Label | Meaning |
|-----|--------|
| `0` | LOSS |
| `1` | DRAW |
| `2` | WIN |

---

## High-Level Pipeline

Parquet Dataset  
→ Preprocessing & Feature Engineering  
→ Train / Validation / Test Split (by gameId)  
→ XGBoost Probability Model  
→ Evaluation & Calibration Metrics  
→ Model + Preprocessor Artifacts  
→ (Optional) MLflow & TensorBoard Logging  
→ (Automated) Retraining on New Data

---

## Exploratory Data Analysis (EDA)

**Purpose:**  
Validate target distribution and detect dataset imbalance early.

**What it does:**
- Uses **final labels produced by preprocessing**
- Plots LOSS / DRAW / WIN distribution
- Saves report to disk per dataset version

---

##  Preprocessing & Feature Engineering

### Dataset Assumptions (New Schema Only)

Expected columns include:
- game_outcome
- game_winner
- player
- policy_col_0 … policy_col_6
- q_value_col_0 … q_value_col_6
- board_before_r{row}c{col}
- moveIndex
- gameId

---

###  Self-Play Perspective Augmentation

Thanks to EDA we realised that when self-play data is used, the positions need to be duplicated


When SELF_PLAY=true, each position is duplicated the following way:
- Original perspective
- Mirrored perspective (player 1 ↔ player 2)

This teaches player-invariant evaluation.

---

### Engineered Features

- Piece counts per player
- Center column control
- Policy entropy and max probability
- Q-value mean and range
- Move number

All features are derived strictly from the current state.

---

## Train / Validation / Test Split

- Split by gameId to avoid leakage
- Train 70%, Validation 10%, Test 20%

---

##  Model Training

- Algorithm: XGBoost Classifier
- Objective: multi:softprob
- Output: calibrated class probabilities

Handles missing classes safely via temporary label remapping.

---

##  Evaluation Metrics

- Accuracy
- Log Loss
- Brier Score
- Win probability calibration telemetry

Accuracy only checks whether the most likely class was correct.
It completely ignores confidence.

For a probability-based system, this is insufficient:

A 51% win prediction and a 99% win prediction are treated the same by accuracy

Overconfident mistakes are not distinguished from reasonable uncertainty

Log loss and Brier score solve this by evaluating how well the model’s confidence matches reality.

---

##  Artifacts

- winprob_model_<version>.joblib
- winprob_preprocessor_<version>.joblib
- EDA reports

---

##  Automated Retraining

A watcher process monitors new parquet files and retrains automatically, optionally notifying a deployment API.

---

##  Summary

This pipeline is production-ready, leakage-safe, probability-calibrated, and designed for continuous self-play learning.
