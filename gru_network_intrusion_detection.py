"""
Deep Learning-Based Network Intrusion Detection System Using GRU
================================================================
Dataset : CIC-IDS2017 (network-flow records, CSV files)
Task    : Binary classification  ->  0 = BENIGN, 1 = any attack
Model   : GRU over sliding windows of 10 consecutive flow records

Usage
-----
1. Put the CIC-IDS2017 CSV files (MachineLearningCSV or TrafficLabelling)
   in a folder called  data/
2. pip install numpy pandas scikit-learn tensorflow matplotlib
3. python gru_network_intrusion_detection.py

Outputs are written to results/ (metrics.csv, confusion_matrix.png,
training_curves.png).
"""

import os
import glob
import random

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # save plots to files, no display needed
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
DATA_DIR = "data"
RESULTS_DIR = "results"

SEQ_LEN = 10          # records per window
TRAIN_FRAC = 0.80     # first 80% of rows -> training part
VAL_FRAC = 0.10       # last 10% of the training part -> validation
BATCH_SIZE = 256
EPOCHS = 20
THRESHOLD = 0.5       # score >= 0.5 -> malicious
SEED = 42

LABEL_COLUMN = "Label"
ID_COLUMNS = [
    "Flow ID", "Source IP", "Src IP", "Destination IP", "Dst IP", "Timestamp",
]
DAY_ORDER = ["monday", "tuesday", "wednesday", "thursday", "friday"]


# --------------------------------------------------------------------------
# 1. Load data
# --------------------------------------------------------------------------
def day_rank(path):
    """Sort CSV files Monday -> Friday so row order follows the capture week."""
    name = os.path.basename(path).lower()
    for i, day in enumerate(DAY_ORDER):
        if day in name:
            return (i, name)
    return (len(DAY_ORDER), name)


def load_data(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")), key=day_rank)
    if not files:
        raise FileNotFoundError(f"No CSV files found in '{data_dir}/'.")
    frames = []
    for path in files:
        df = pd.read_csv(path, low_memory=False, encoding="latin1")
        df.columns = df.columns.str.strip()  # some headers have leading spaces
        frames.append(df)
        print(f"Loaded {os.path.basename(path):55s} {len(df):>8,} rows")
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------
# 2. Clean data and build binary labels
# --------------------------------------------------------------------------
def clean_and_label(df):
    # Drop identifier columns (they identify hosts/time, not behaviour)
    df = df.drop(columns=[c for c in ID_COLUMNS if c in df.columns])

    # BENIGN = 0, every attack type = 1
    text = df[LABEL_COLUMN].astype(str).str.strip().str.upper()
    labels = (text != "BENIGN").astype(np.int8).to_numpy()

    # Make every feature numeric; infinite values become missing values
    features = df.drop(columns=[LABEL_COLUMN])
    features = features.apply(pd.to_numeric, errors="coerce")
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.dropna(axis=1, how="all")  # drop empty columns
    return features, labels


# --------------------------------------------------------------------------
# 3. Split in order, fill missing values, scale
# --------------------------------------------------------------------------
def split_in_order(n_rows):
    """First 80% train, last 20% test; last 10% of train is validation.
    Rows are NOT shuffled, so the split respects the row (time) order."""
    n_train_total = int(n_rows * TRAIN_FRAC)
    n_val = int(n_train_total * VAL_FRAC)
    n_train = n_train_total - n_val
    return (
        slice(0, n_train),
        slice(n_train, n_train_total),
        slice(n_train_total, n_rows),
    )


def preprocess(features, labels):
    tr, va, te = split_in_order(len(features))
    X_train, X_val, X_test = features.iloc[tr], features.iloc[va], features.iloc[te]
    y_train, y_val, y_test = labels[tr], labels[va], labels[te]

    # Statistics come from the TRAINING rows only (no leakage from test)
    medians = X_train.median()
    X_train = X_train.fillna(medians).fillna(0.0)
    X_val = X_val.fillna(medians).fillna(0.0)
    X_test = X_test.fillna(medians).fillna(0.0)

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train).astype("float32")
    X_val = scaler.transform(X_val).astype("float32")
    X_test = scaler.transform(X_test).astype("float32")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# --------------------------------------------------------------------------
# 4. Sequence construction (sliding window of 10 consecutive records)
# --------------------------------------------------------------------------
def make_dataset(X, y, shuffle):
    """Windows are built INSIDE one part, so none crosses the split border.
    Window label = label of its LAST record. Only the order of the finished
    windows is shuffled for training; rows are never shuffled before windowing."""
    window_labels = y[SEQ_LEN - 1:]
    ds = keras.utils.timeseries_dataset_from_array(
        data=X,
        targets=window_labels,
        sequence_length=SEQ_LEN,
        sequence_stride=1,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=SEED,
    )
    return ds, window_labels


# --------------------------------------------------------------------------
# 5. Model
# --------------------------------------------------------------------------
def build_model(seq_len, n_features):
    model = keras.Sequential([
        layers.Input(shape=(seq_len, n_features)),
        layers.GRU(64),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# --------------------------------------------------------------------------
# 6. Evaluation
# --------------------------------------------------------------------------
def evaluate(model, test_ds, y_true):
    scores = model.predict(test_ds, verbose=0).ravel()
    y_pred = (scores >= THRESHOLD).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }
    return metrics, np.array([[tn, fp], [fn, tp]])


def save_confusion_matrix(cm, path):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Benign", "Malicious"])
    ax.set_yticks([0, 1], labels=["Benign", "Malicious"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix (test set)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_training_curves(history, path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for ax, key, title in zip(axes, ["loss", "accuracy"], ["Loss", "Accuracy"]):
        ax.plot(history.history[key], label="train")
        ax.plot(history.history["val_" + key], label="validation")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load, clean, split, scale
    df = load_data(DATA_DIR)
    features, labels = clean_and_label(df)
    del df
    print(f"\nRows: {len(features):,}   Features: {features.shape[1]}   "
          f"Attack share: {labels.mean():.2%}")

    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = preprocess(features, labels)

    # Windows (built separately inside train / validation / test)
    train_ds, y_train_win = make_dataset(X_tr, y_tr, shuffle=True)
    val_ds, _ = make_dataset(X_va, y_va, shuffle=False)
    test_ds, y_test_win = make_dataset(X_te, y_te, shuffle=False)

    # Class weights so the rarer class is not ignored
    classes = np.unique(y_train_win)
    weights = compute_class_weight("balanced", classes=classes, y=y_train_win)
    class_weight = {int(c): float(w) for c, w in zip(classes, weights)}
    print("Class weights:", class_weight)

    # Model + training with early stopping
    model = build_model(SEQ_LEN, X_tr.shape[1])
    model.summary()
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=class_weight,
        callbacks=[early_stop],
        verbose=2,
    )

    # Evaluate on data the model has never seen
    metrics, cm = evaluate(model, test_ds, y_test_win)
    print("\nTest results")
    for name, value in metrics.items():
        print(f"  {name:15s} {value:.4f}" if isinstance(value, float)
              else f"  {name:15s} {value:,}")

    pd.DataFrame([metrics]).to_csv(
        os.path.join(RESULTS_DIR, "metrics.csv"), index=False
    )
    save_confusion_matrix(cm, os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    save_training_curves(history, os.path.join(RESULTS_DIR, "training_curves.png"))
    model.save(os.path.join(RESULTS_DIR, "gru_nids.keras"))
    print(f"\nSaved metrics, plots and model to '{RESULTS_DIR}/'.")


if __name__ == "__main__":
    main()
