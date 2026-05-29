import pandas as pd
import numpy as np
import xgboost as xgb
import mlflow
import mlflow.xgboost
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, matthews_corrcoef, confusion_matrix,
    roc_curve
)
import warnings
warnings.filterwarnings("ignore")

print("Loading dataset...")
df = pd.read_csv("travistorrent.csv")
print(f"Loaded {len(df)} records")

# Encode target
df["label"] = (df["tr_status"] == "failed").astype(int)

# Encode categorical columns
le_lang = LabelEncoder()
le_branch = LabelEncoder()
df["gh_lang"] = le_lang.fit_transform(df["gh_lang"].astype(str))
df["git_branch"] = le_branch.fit_transform(df["git_branch"].astype(str))

# Feature columns
FEATURES = [
    "gh_diff_src_churn", "gh_diff_test_churn", "git_num_all_commits",
    "gh_team_size", "gh_sloc", "gh_test_lines_per_kloc",
    "gh_is_pr", "tr_prev_build_status", "tr_prev_build_dur",
    "build_hour", "build_day_of_week", "developer_experience",
    "files_changed", "tests_changed", "dep_depth",
    "gh_num_pr_comments", "gh_pull_req_num",
    "git_commits_on_files_touched", "gh_lang", "git_branch"
]

X = df[FEATURES].fillna(0)
y = df["label"]

# Temporal split — 70% train, 30% test
split = int(len(df) * 0.7)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

print(f"Training: {len(X_train)} | Test: {len(X_test)}")
print(f"Class balance — PASS: {(y_train==0).sum()} FAIL: {(y_train==1).sum()}")

scale = (y_train == 0).sum() / (y_train == 1).sum()

def evaluate(model, X_test, y_test, name):
    proba = model.predict_proba(X_test)[:, 1]
    pred  = (proba >= 0.5).astype(int)
    auc   = roc_auc_score(y_test, proba)
    f1    = f1_score(y_test, pred)
    prec  = precision_score(y_test, pred)
    rec   = recall_score(y_test, pred)
    mcc   = matthews_corrcoef(y_test, pred)
    print(f"\n{name}")
    print(f"  AUC-ROC:   {auc:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  MCC:       {mcc:.4f}")
    return auc, f1, prec, rec, mcc, proba

results = {}

# ── Logistic Regression ───────────────────────────────────────────────
print("\nTraining Logistic Regression...")
lr = LogisticRegression(max_iter=1000, class_weight="balanced")
lr.fit(X_train, y_train)
results["Logistic Regression"] = evaluate(lr, X_test, y_test, "Logistic Regression")

# ── Random Forest ─────────────────────────────────────────────────────
print("\nTraining Random Forest...")
rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
results["Random Forest"] = evaluate(rf, X_test, y_test, "Random Forest")

# ── XGBoost ───────────────────────────────────────────────────────────
print("\nTraining XGBoost...")
mlflow.set_experiment("cicd_failure_prediction")
with mlflow.start_run(run_name="XGBoost"):
    params = {
        "max_depth": 5,
        "learning_rate": 0.1,
        "n_estimators": 300,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": scale,
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "random_state": 42,
    }
    xgb_model = xgb.XGBClassifier(**params)
    xgb_model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  verbose=False)
    mlflow.log_params(params)
    auc, f1, prec, rec, mcc, xgb_proba = evaluate(xgb_model, X_test, y_test, "XGBoost")
    mlflow.log_metrics({"auc_roc":auc,"f1":f1,"precision":prec,"recall":rec,"mcc":mcc})
    results["XGBoost"] = (auc, f1, prec, rec, mcc, xgb_proba)
    joblib.dump(xgb_model, "xgb_model.pkl")
    print("\n  Model saved as xgb_model.pkl")

# ── Comparison table ──────────────────────────────────────────────────
print("\n" + "="*60)
print(f"{'Model':<22} {'AUC-ROC':>8} {'F1':>8} {'Prec':>8} {'Rec':>8} {'MCC':>8}")
print("-"*60)
for name, (auc,f1,prec,rec,mcc,_) in results.items():
    print(f"{name:<22} {auc:>8.4f} {f1:>8.4f} {prec:>8.4f} {rec:>8.4f} {mcc:>8.4f}")
print("="*60)

# ── ROC Curve plot ────────────────────────────────────────────────────
plt.figure(figsize=(8,6))
colors = ["steelblue","orange","green"]
for (name, (auc,_,_,_,_,proba)), color in zip(results.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, proba)
    plt.plot(fpr, tpr, color=color, label=f"{name} (AUC={auc:.3f})")
plt.plot([0,1],[0,1],"k--",alpha=0.5)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves — All Models")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curves.png", dpi=150)
print("\nSaved: roc_curves.png")

# ── Confusion Matrix ──────────────────────────────────────────────────
xgb_pred = (xgb_proba >= 0.5).astype(int)
cm = confusion_matrix(y_test, xgb_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["PASS","FAIL"], yticklabels=["PASS","FAIL"])
plt.title("Confusion Matrix — XGBoost")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("Saved: confusion_matrix.png")

print("\nAll done! Next step: run train_lstm.py")