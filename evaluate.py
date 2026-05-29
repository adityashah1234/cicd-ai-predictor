import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import shap
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow import keras
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, matthews_corrcoef, confusion_matrix, roc_curve
)

print("Loading dataset and models...")

# ── Load data ─────────────────────────────────────────────────────────
df = pd.read_csv("travistorrent.csv")
df["label"] = (df["tr_status"] == "failed").astype(int)
df = df.sort_values("tr_build_id").reset_index(drop=True)

le_lang   = LabelEncoder()
le_branch = LabelEncoder()
df["gh_lang"]    = le_lang.fit_transform(df["gh_lang"].astype(str))
df["git_branch"] = le_branch.fit_transform(df["git_branch"].astype(str))

FEATURES = [
    "gh_diff_src_churn","gh_diff_test_churn","git_num_all_commits",
    "gh_team_size","gh_sloc","gh_test_lines_per_kloc",
    "gh_is_pr","tr_prev_build_status","tr_prev_build_dur",
    "build_hour","build_day_of_week","developer_experience",
    "files_changed","tests_changed","dep_depth",
    "gh_num_pr_comments","gh_pull_req_num",
    "git_commits_on_files_touched","gh_lang","git_branch"
]

X = df[FEATURES].fillna(0)
y = df["label"]

split     = int(len(df) * 0.7)
X_train   = X.iloc[:split]
X_test    = X.iloc[split:]
y_train   = y.iloc[:split]
y_test    = y.iloc[split:]

scale = (y_train==0).sum() / (y_train==1).sum()

# ── Load XGBoost ──────────────────────────────────────────────────────
import xgboost as xgb
xgb_model = joblib.load("xgb_model.pkl")
xgb_proba = xgb_model.predict_proba(X_test)[:,1]

# ── Load LSTM ─────────────────────────────────────────────────────────
lstm_model = keras.models.load_model("lstm_model.h5")

df["churn_norm"] = df["gh_diff_src_churn"] / (df["gh_diff_src_churn"].max()+1)
df["dur_norm"]   = df["tr_prev_build_dur"]  / (df["tr_prev_build_dur"].max()+1)
SEQ_LEN = 10
FEAT_SEQ = ["label","churn_norm","dur_norm"]

sequences, seq_indices = [], []
for project in df["gh_project_name"].unique():
    proj = df[df["gh_project_name"]==project].reset_index(drop=True)
    if len(proj) < SEQ_LEN+1:
        continue
    for i in range(SEQ_LEN, len(proj)):
        orig_idx = proj.index[i]
        sequences.append(proj[FEAT_SEQ].iloc[i-SEQ_LEN:i].values)
        seq_indices.append(orig_idx)

X_seq_all = np.array(sequences, dtype=np.float32)
seq_idx   = np.array(seq_indices)

test_start = split
seq_test_mask = seq_idx >= test_start
X_seq_test    = X_seq_all[seq_test_mask]
seq_test_idx  = seq_idx[seq_test_mask]

if len(X_seq_test) > 0:
    lstm_proba_seq = lstm_model.predict(X_seq_test, verbose=0).flatten()
    lstm_proba_full = np.full(len(X_test), 0.25)
    for i, idx in enumerate(seq_test_idx):
        local = idx - test_start
        if 0 <= local < len(lstm_proba_full):
            lstm_proba_full[local] = lstm_proba_seq[i]
else:
    lstm_proba_full = np.full(len(X_test), 0.25)

# ── Ensemble ──────────────────────────────────────────────────────────
ensemble_proba = 0.6 * xgb_proba + 0.4 * lstm_proba_full

# ── Baseline models ───────────────────────────────────────────────────
print("Training baselines for comparison...")
lr = LogisticRegression(max_iter=1000, class_weight="balanced")
lr.fit(X_train, y_train)
lr_proba = lr.predict_proba(X_test)[:,1]

rf = RandomForestClassifier(n_estimators=100, class_weight="balanced",
                             random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_proba = rf.predict_proba(X_test)[:,1]

lstm_proba_eval = lstm_proba_full

# ── Evaluation function ───────────────────────────────────────────────
def evaluate(proba, y_true, name):
    pred = (proba >= 0.5).astype(int)
    auc  = roc_auc_score(y_true, proba)
    f1   = f1_score(y_true, pred)
    prec = precision_score(y_true, pred)
    rec  = recall_score(y_true, pred)
    mcc  = matthews_corrcoef(y_true, pred)
    return {"name":name,"auc":auc,"f1":f1,"prec":prec,"rec":rec,"mcc":mcc,"proba":proba}

models = [
    evaluate(lr_proba,       y_test, "Logistic Regression"),
    evaluate(rf_proba,       y_test, "Random Forest"),
    evaluate(xgb_proba,      y_test, "XGBoost only"),
    evaluate(lstm_proba_eval,y_test, "LSTM only"),
    evaluate(ensemble_proba, y_test, "Hybrid XGBoost-LSTM"),
]

# ── Print comparison table ────────────────────────────────────────────
print("\n" + "="*65)
print(f"{'Model':<25} {'AUC-ROC':>8} {'F1':>8} {'Prec':>8} {'Rec':>8} {'MCC':>8}")
print("-"*65)
for m in models:
    marker = " <-- PROPOSED" if m["name"] == "Hybrid XGBoost-LSTM" else ""
    print(f"{m['name']:<25} {m['auc']:>8.4f} {m['f1']:>8.4f} "
          f"{m['prec']:>8.4f} {m['rec']:>8.4f} {m['mcc']:>8.4f}{marker}")
print("="*65)

# ── Figure 1: ROC Curves ──────────────────────────────────────────────
plt.figure(figsize=(8,6))
colors = ["steelblue","orange","green","purple","red"]
for m, color in zip(models, colors):
    fpr, tpr, _ = roc_curve(y_test, m["proba"])
    lw = 2.5 if m["name"]=="Hybrid XGBoost-LSTM" else 1.5
    plt.plot(fpr, tpr, color=color, linewidth=lw,
             label=f"{m['name']} (AUC={m['auc']:.3f})")
plt.plot([0,1],[0,1],"k--",alpha=0.4)
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.title("ROC Curves: All Models", fontsize=13)
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig("roc_curves.png", dpi=150)
print("\nSaved: roc_curves.png")

# ── Figure 2: Confusion Matrix ────────────────────────────────────────
ens_pred = (ensemble_proba >= 0.5).astype(int)
cm = confusion_matrix(y_test, ens_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["PASS","FAIL"],
            yticklabels=["PASS","FAIL"])
plt.title("Confusion Matrix: Hybrid XGBoost-LSTM", fontsize=12)
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("Saved: confusion_matrix.png")

# ── Figure 3: SHAP Summary ────────────────────────────────────────────
print("\nGenerating SHAP plots (this takes 1-2 minutes)...")
sample = X_test.sample(500, random_state=42)
explainer   = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(sample)

plt.figure()
shap.summary_plot(shap_values, sample, feature_names=FEATURES,
                  show=False, max_display=15)
plt.title("SHAP Feature Importance: XGBoost", fontsize=12)
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=150, bbox_inches="tight")
print("Saved: shap_summary.png")

# ── Figure 4: SHAP Waterfall — 3 examples ────────────────────────────
high_risk_idx = np.where((y_test.values==1) & (ensemble_proba>=0.7))[0]
low_risk_idx  = np.where((y_test.values==0) & (ensemble_proba<=0.3))[0]

def save_waterfall(idx_array, label, filename):
    if len(idx_array) == 0:
        print(f"  No {label} examples found — skipping")
        return
    row  = X_test.iloc[idx_array[0]:idx_array[0]+1]
    sv   = explainer.shap_values(row)[0]
    base = explainer.expected_value
    feat_names = FEATURES
    pairs = sorted(zip(sv, feat_names), key=lambda x: abs(x[0]), reverse=True)[:8]
    values, names = zip(*pairs)
    colors = ["#d73027" if v>0 else "#4575b4" for v in values]
    plt.figure(figsize=(8,5))
    plt.barh(range(len(values)), values, color=colors)
    plt.yticks(range(len(names)), names)
    plt.axvline(0, color="black", linewidth=0.8)
    prob = ensemble_proba[idx_array[0]]
    plt.title(f"SHAP Waterfall: {label}\nPredicted risk: {prob:.1%}", fontsize=12)
    plt.xlabel("SHAP value (impact on prediction)")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"Saved: {filename}")

save_waterfall(high_risk_idx, "HIGH RISK (FAIL)",  "shap_waterfall_fail.png")
save_waterfall(low_risk_idx,  "LOW RISK (PASS)",   "shap_waterfall_pass.png")

# ── Figure 5: Evaluate on microservices data ──────────────────────────
print("\nEvaluating on microservices pipeline_runs.csv...")
try:
    ms = pd.read_csv("pipeline_runs.csv")
    ms["label"] = (ms["actual_result"] == "fail").astype(int)

    ms_feat = pd.DataFrame()
    for f in FEATURES:
        if f in ms.columns:
            ms_feat[f] = ms[f]
        else:
            ms_feat[f] = 0
    ms_feat = ms_feat.fillna(0)

    ms_proba = xgb_model.predict_proba(ms_feat)[:,1]
    ms_auc   = roc_auc_score(ms["label"], ms_proba)
    ms_f1    = f1_score(ms["label"], (ms_proba>=0.5).astype(int))
    print(f"\nMicroservices evaluation set results:")
    print(f"  AUC-ROC: {ms_auc:.4f}")
    print(f"  F1:      {ms_f1:.4f}")
    print(f"  (These are your Table 6.3 results for Chapter 6)")
except Exception as e:
    print(f"  Could not evaluate microservices set: {e}")

print("\n" + "="*50)
print("ALL EVALUATION COMPLETE!")
print("Files saved:")
for f in ["roc_curves.png","confusion_matrix.png",
          "shap_summary.png","shap_waterfall_fail.png","shap_waterfall_pass.png"]:
    print(f"  {f}")
print("="*50)