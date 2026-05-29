import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("Loading dataset...")
df = pd.read_csv("travistorrent.csv")
df["label"] = (df["tr_status"] == "failed").astype(int)
df = df.sort_values("tr_build_id").reset_index(drop=True)

# Normalise sequence features
df["churn_norm"] = df["gh_diff_src_churn"] / (df["gh_diff_src_churn"].max() + 1)
df["dur_norm"]   = df["tr_prev_build_dur"] / (df["tr_prev_build_dur"].max() + 1)

SEQ_LEN = 10
FEATURES_SEQ = ["label", "churn_norm", "dur_norm"]

print("Building sequences...")
sequences = []
labels = []

for project in df["gh_project_name"].unique():
    proj_df = df[df["gh_project_name"] == project].reset_index(drop=True)
    if len(proj_df) < SEQ_LEN + 1:
        continue
    for i in range(SEQ_LEN, len(proj_df)):
        seq = proj_df[FEATURES_SEQ].iloc[i-SEQ_LEN:i].values
        lbl = proj_df["label"].iloc[i]
        sequences.append(seq)
        labels.append(lbl)

X_seq = np.array(sequences, dtype=np.float32)
y_seq = np.array(labels, dtype=np.float32)

print(f"Total sequences: {len(X_seq)}")
print(f"Sequence shape: {X_seq.shape}")
print(f"Class balance — PASS: {(y_seq==0).sum()} FAIL: {(y_seq==1).sum()}")

# Temporal split
split = int(len(X_seq) * 0.7)
X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]

# Class weights
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
class_weight = {0: 1.0, 1: neg / pos}
print(f"\nClass weight for FAIL: {neg/pos:.2f}")

# ── Build LSTM model ─────────────────────────────────────────────────
print("\nBuilding LSTM model...")
model = keras.Sequential([
    layers.Input(shape=(SEQ_LEN, len(FEATURES_SEQ))),
    layers.LSTM(64, return_sequences=True),
    layers.Dropout(0.3),
    layers.LSTM(64, return_sequences=False),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["AUC"]
)

model.summary()

# ── Train ─────────────────────────────────────────────────────────────
print("\nTraining LSTM...")
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_AUC", patience=5,
    restore_best_weights=True, mode="max"
)

history = model.fit(
    X_train, y_train,
    validation_split=0.15,
    epochs=30,
    batch_size=64,
    class_weight=class_weight,
    callbacks=[early_stop],
    verbose=1
)

# ── Evaluate ──────────────────────────────────────────────────────────
print("\nEvaluating on test set...")
lstm_proba = model.predict(X_test, verbose=0).flatten()
lstm_pred  = (lstm_proba >= 0.5).astype(int)

auc = roc_auc_score(y_test, lstm_proba)
f1  = f1_score(y_test, lstm_pred)

print(f"\nLSTM Results:")
print(f"  AUC-ROC: {auc:.4f}")
print(f"  F1:      {f1:.4f}")

# ── Save model ────────────────────────────────────────────────────────
model.save("lstm_model.h5")
print("\nSaved: lstm_model.h5")

# ── Training curve ────────────────────────────────────────────────────
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history["loss"], label="Train loss")
plt.plot(history.history["val_loss"], label="Val loss")
plt.title("Loss")
plt.xlabel("Epoch")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history["AUC"], label="Train AUC")
plt.plot(history.history["val_AUC"], label="Val AUC")
plt.title("AUC-ROC")
plt.xlabel("Epoch")
plt.legend()

plt.tight_layout()
plt.savefig("lstm_training_curve.png", dpi=150)
print("Saved: lstm_training_curve.png")
print("\nAll done! Next step: run evaluate.py")