#!/usr/bin/env python3
"""
predict.py
----------
Called by GitHub Actions after every push.
Loads the trained XGBoost model, reads the latest
commit features from pipeline_runs.csv, and prints
a risk score with a plain-English SHAP explanation.
"""

import pandas as pd
import numpy as np
import joblib
import shap
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder

# ── Feature list must match train_xgboost.py exactly ─────────────────
FEATURES = [
    "gh_diff_src_churn","gh_diff_test_churn","git_num_all_commits",
    "gh_team_size","gh_sloc","gh_test_lines_per_kloc",
    "gh_is_pr","tr_prev_build_status","tr_prev_build_dur",
    "build_hour","build_day_of_week","developer_experience",
    "files_changed","tests_changed","dep_depth",
    "gh_num_pr_comments","gh_pull_req_num",
    "git_commits_on_files_touched","gh_lang","git_branch"
]

FEATURE_LABELS = {
    "gh_diff_src_churn":          "source code lines changed",
    "gh_diff_test_churn":         "test code lines changed",
    "git_num_all_commits":        "total project commits",
    "gh_team_size":               "team size",
    "gh_sloc":                    "project size (lines of code)",
    "gh_test_lines_per_kloc":     "test coverage density",
    "gh_is_pr":                   "is a pull request",
    "tr_prev_build_status":       "previous build failed",
    "tr_prev_build_dur":          "previous build duration",
    "build_hour":                 "hour of day commit was pushed",
    "build_day_of_week":          "day of week (5=Sat 6=Sun)",
    "developer_experience":       "developer experience (prior commits)",
    "files_changed":              "number of files changed",
    "tests_changed":              "test files were modified",
    "dep_depth":                  "microservice dependency depth",
    "gh_num_pr_comments":         "pull request comment count",
    "gh_pull_req_num":            "pull request number",
    "git_commits_on_files_touched":"commits on changed files",
    "gh_lang":                    "programming language",
    "git_branch":                 "branch type",
}

def load_latest_features():
    """Read the most recent row from pipeline_runs.csv."""
    try:
        df = pd.read_csv("pipeline_runs.csv")
        if len(df) == 0:
            return None
        latest = df.iloc[-1]

        # Map pipeline_runs columns to model feature names
        row = {
            "gh_diff_src_churn":           latest.get("src_churn", 50),
            "gh_diff_test_churn":          latest.get("tests_changed", 0) * 20,
            "git_num_all_commits":         latest.get("git_num_all_commits", 100),
            "gh_team_size":                latest.get("gh_team_size", 3),
            "gh_sloc":                     latest.get("gh_sloc", 500),
            "gh_test_lines_per_kloc":      latest.get("gh_test_lines_per_kloc", 60),
            "gh_is_pr":                    latest.get("is_pr", 0),
            "tr_prev_build_status":        latest.get("prev_build_status", 0),
            "tr_prev_build_dur":           latest.get("prev_build_dur", 90),
            "build_hour":                  latest.get("build_hour", 14),
            "build_day_of_week":           latest.get("build_day_of_week", 1),
            "developer_experience":        latest.get("developer_experience", 10),
            "files_changed":               latest.get("files_changed", 2),
            "tests_changed":               latest.get("tests_changed", 0),
            "dep_depth":                   latest.get("dep_depth", 1),
            "gh_num_pr_comments":          latest.get("gh_num_pr_comments", 0),
            "gh_pull_req_num":             latest.get("gh_pull_req_num", 0),
            "git_commits_on_files_touched":latest.get("git_commits_on_files_touched", 5),
            "gh_lang":                     1,
            "git_branch":                  0,
        }
        service  = latest.get("service", "unknown")
        hour     = int(row["build_hour"])
        day      = int(row["build_day_of_week"])
        dep      = int(row["dep_depth"])
        dev_exp  = int(row["developer_experience"])
        churn    = int(row["gh_diff_src_churn"])
        files    = int(row["files_changed"])
        prev_st  = int(row["tr_prev_build_status"])
        return row, service, hour, day, dep, dev_exp, churn, files, prev_st
    except Exception as e:
        print(f"Warning: could not read pipeline_runs.csv ({e})")
        return None

def day_name(d):
    days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    return days[min(d,6)]

def main():
    print("")
    print("=" * 55)
    print("  AI FAILURE PREDICTION")
    print("=" * 55)

    # Load model
    try:
        model = joblib.load("xgb_model.pkl")
    except FileNotFoundError:
        print("  Model not found (xgb_model.pkl)")
        print("  Run train_xgboost.py first")
        print("=" * 55)
        return

    # Load features
    result = load_latest_features()
    if result is None:
        print("  No pipeline data found")
        print("=" * 55)
        return

    row, service, hour, day, dep, dev_exp, churn, files, prev_st = result

    # Build feature vector
    X = pd.DataFrame([row])[FEATURES].fillna(0)

    # Predict
    proba = model.predict_proba(X)[0][1]
    risk_pct = int(proba * 100)

    if proba >= 0.65:
        level = "HIGH RISK"
        emoji = "FAIL"
    elif proba >= 0.40:
        level = "MEDIUM RISK"
        emoji = "WARN"
    else:
        level = "LOW RISK"
        emoji = "PASS"

    print(f"  Service:    {service}")
    print(f"  Risk score: {risk_pct}%")
    print(f"  Verdict:    {level}")
    print("")

    # SHAP explanation
    try:
        explainer   = shap.TreeExplainer(model)
        shap_vals   = explainer.shap_values(X)[0]
        pairs = sorted(
            zip(shap_vals, FEATURES),
            key=lambda x: abs(x[0]),
            reverse=True
        )[:5]

        print("  TOP FACTORS DRIVING THIS PREDICTION:")
        print("  " + "-" * 48)
        for val, feat in pairs:
            direction = "increases" if val > 0 else "reduces"
            label     = FEATURE_LABELS.get(feat, feat)
            actual    = X[feat].values[0]
            print(f"  {'(+)' if val>0 else '(-)'}  {label}")
            print(f"       value={actual:.0f}  |  {direction} failure risk by {abs(val):.3f}")
        print("  " + "-" * 48)
    except Exception as e:
        print(f"  SHAP explanation unavailable: {e}")

    # Plain English summary
    print("")
    print("  PLAIN ENGLISH SUMMARY:")
    reasons = []
    if prev_st == 1:
        reasons.append("the previous build already failed")
    if churn > 200:
        reasons.append(f"a large amount of code was changed ({churn} lines)")
    if dep >= 6:
        reasons.append(f"{service} has {dep} downstream services depending on it")
    if hour >= 20:
        reasons.append(f"this commit was pushed late at {hour:02d}:00")
    if day >= 5:
        reasons.append(f"it is {day_name(day)}, a historically risky day")
    if dev_exp <= 5:
        reasons.append(f"the developer has only {dev_exp} prior commits")
    if files >= 8:
        reasons.append(f"{files} files were changed at once")

    if reasons:
        summary = "  This build is predicted " + level + " because " + \
                  reasons[0]
        if len(reasons) > 1:
            summary += ", and " + reasons[1]
        summary += "."
        print(summary)
    else:
        print(f"  This build looks {level.lower()} based on current commit features.")

    print("")
    print("=" * 55)

if __name__ == "__main__":
    main()