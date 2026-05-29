# AI-Driven CI/CD Pipeline Optimisation for Microservices

> MSc Computing Dissertation | University of West London | Student ID: 34116912 | May 2026

---

## What this project does

This project builds a system that predicts whether a CI/CD pipeline run will fail before it actually fails. Most CI/CD tools are reactive: they run tests, find a problem, and tell you something went wrong. This system does the opposite. The moment a developer pushes code to GitHub, it reads facts about that commit and estimates the failure risk before any test has run. It then explains the prediction in plain English so the developer can see exactly why the build looks risky.

The system combines two machine learning models. XGBoost looks at 20 features of the current commit: files changed, source code churn, which service was modified, the hour of day, the developer's experience, and so on. An LSTM network looks at the sequence of the last ten builds in the same project and picks up on patterns over time. The two predictions are combined into a single risk score, and a SHAP explanation lists the top factors driving that score.

This approach fills three gaps confirmed by a 2025 systematic review of 92 papers:
- No prior study combined XGBoost and LSTM for CI/CD failure prediction
- Prior papers used features that are only available after a build finishes, inflating their accuracy figures. This study uses only features available before a build starts
- No prior CI/CD failure predictor explained its predictions per-prediction in plain English

---

## Live demo

The prediction runs automatically on every push to this repository. Go to the Actions tab and click on any recent workflow run. Open the **AI failure prediction** step and you will see output like this:

```
=======================================================
  AI FAILURE PREDICTION
=======================================================
  Service:     user-service
  Risk score:  99%
  Verdict:     HIGH RISK

  TOP FACTORS DRIVING THIS PREDICTION:
  -----------------------------------------------
  (+)  day of week (5=Sat 6=Sun)        +9.251
  (+)  hour of day commit was pushed    +0.885
  (+)  number of files changed          +0.345
  (+)  developer experience             +0.271
  (+)  source code lines changed        +0.197
  -----------------------------------------------

  PLAIN ENGLISH SUMMARY:
  This build is predicted HIGH RISK because the previous
  build already failed, and a large amount of code was
  changed (315 lines).
=======================================================
```

---

## Repository structure

```
cicd-ai-predictor/
├── user-service/
│   ├── app.py              Flask microservice: GET /users, GET /health
│   ├── test_app.py         3 pytest tests
│   └── __init__.py
├── order-service/
│   ├── app.py              Flask microservice: POST /orders, GET /orders, GET /health
│   ├── test_app.py         3 pytest tests
│   └── __init__.py
├── payment-service/
│   ├── app.py              Flask microservice: POST /pay, GET /health
│   ├── test_app.py         3 pytest tests
│   └── __init__.py
├── .github/
│   └── workflows/
│       └── ci.yml          GitHub Actions pipeline
├── train_xgboost.py        Trains XGBoost + baseline models, saves xgb_model.pkl
├── train_lstm.py           Trains LSTM on build sequences, saves lstm_model.h5
├── evaluate.py             Runs all models, generates all evaluation figures
├── predict.py              Called by GitHub Actions: outputs risk score + SHAP explanation
├── extract_features.py     Extracts 20 pre-execution commit features
├── generate_runs.py        Generates 100 controlled pipeline runs (50 pass, 50 fail)
├── service_graph.json      Microservice dependency depth map
├── travistorrent.csv       Synthetic training dataset (100,000 records)
├── pipeline_runs.csv       Microservices evaluation dataset (100 runs)
├── xgb_model.pkl           Trained XGBoost model
├── lstm_model.h5           Trained LSTM model
├── roc_curves.png          ROC curves comparing all five models
├── confusion_matrix.png    Confusion matrix for the hybrid ensemble
├── shap_summary.png        SHAP beeswarm plot: global feature importance
├── shap_waterfall_fail.png SHAP waterfall: example high-risk prediction
├── shap_waterfall_pass.png SHAP waterfall: example low-risk prediction
├── lstm_training_curve.png LSTM training and validation loss/AUC curves
└── requirements.txt        All Python dependencies with pinned versions
```

---

## How to reproduce the results

### 1. Clone the repository

```bash
git clone https://github.com/adityashah1234/cicd-ai-predictor.git
cd cicd-ai-predictor
```

### 2. Set up the Python environment

Python 3.11 is required. On Apple Silicon (M1/M2/M3) Mac use tensorflow-macos instead of tensorflow.

```bash
python3.11 -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade pip setuptools
python3 -m pip install -r requirements.txt
```

On Apple Silicon Mac only, replace the TensorFlow line:
```bash
python3 -m pip install tensorflow-macos==2.16.2 tensorflow-metal==1.1.0
```

### 3. Generate the training dataset

The synthetic dataset mimics the statistical properties of TravisTorrent (Beller et al., 2017). 100,000 records across 500 projects with realistic class balance (72% pass, 28% fail) and serial correlation between consecutive builds.

```bash
python3 -c "
import pandas as pd, numpy as np
random_state = np.random.RandomState(42)
# ... (see generate_runs.py for full dataset generation code)
"
```

Or simply use the travistorrent.csv already in this repository.

### 4. Train the XGBoost model

```bash
python3 train_xgboost.py
```

This trains logistic regression, random forest, and XGBoost on a temporal 70/30 split. Saves xgb_model.pkl. Takes about 5 minutes. Results are logged with MLflow — run `mlflow ui` and open localhost:5000 to view them.

### 5. Train the LSTM model

```bash
python3 train_lstm.py
```

This builds sequences of 10 consecutive builds per project and trains a two-layer LSTM with early stopping. Saves lstm_model.h5. Takes about 10 minutes.

### 6. Run the full evaluation

```bash
python3 evaluate.py
```

This loads both models, runs the weighted ensemble (XGBoost x 0.6 + LSTM x 0.4), generates all six figures, prints the comparison table, and evaluates on the microservices pipeline_runs.csv.

Expected output:
```
Model                     AUC-ROC       F1     Prec      Rec      MCC
-----------------------------------------------------------------
Logistic Regression        0.994    0.923    0.883    0.968    0.893
Random Forest              1.000    1.000    1.000    1.000    1.000
XGBoost only               1.000    1.000    1.000    1.000    1.000
LSTM only                  0.768    0.639    0.612    0.668    0.581
Hybrid XGBoost-LSTM        1.000    1.000    1.000    1.000    1.000  <-- PROPOSED
```

### 7. Run a single prediction

```bash
python3 predict.py
```

This reads the most recent row from pipeline_runs.csv, loads the trained models, and prints the risk score and SHAP explanation. Output appears in under 400 milliseconds.

### 8. Run the microservice tests

```bash
cd user-service && python3 -m pytest test_app.py -v && cd ..
cd order-service && python3 -m pytest test_app.py -v && cd ..
cd payment-service && python3 -m pytest test_app.py -v && cd ..
```

All 9 tests should pass.

---

## Results summary

| Dataset | Model | AUC-ROC | F1 |
|---|---|---|---|
| Synthetic test set | Logistic Regression | 0.994 | 0.923 |
| Synthetic test set | Random Forest | 1.000 | 1.000 |
| Synthetic test set | XGBoost only | 1.000 | 1.000 |
| Synthetic test set | LSTM only | 0.768 | 0.639 |
| Synthetic test set | **Hybrid XGBoost-LSTM** | **1.000** | **1.000** |
| Microservices (real) | Hybrid XGBoost-LSTM | 1.000 | 0.935 |

**Note on the synthetic results:** The near-perfect scores reflect the structured nature of the synthetic training data. On real-world data such as TravisTorrent, XGBoost-only models typically achieve 0.86 to 0.94 AUC-ROC (Rajesh Kumar et al., 2025). The LSTM score of 0.768 is more representative of real-world performance and is consistent with comparable studies.

---

## Technology stack

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.11 | Core language |
| Flask | 3.0 | Microservice framework |
| pytest | 8.2 | Test suite |
| XGBoost | 2.1 | Primary classifier |
| TensorFlow/Keras | 2.16 | LSTM model |
| SHAP | 0.45 | Per-prediction explanations |
| scikit-learn | 1.4 | Baseline classifiers and preprocessing |
| pandas | 2.2 | Data loading and feature engineering |
| MLflow | 2.12 | Experiment tracking |
| GitHub Actions | N/A | CI/CD pipeline and live prediction |

---

## Limitations

- The training dataset is synthetic. A proper evaluation on real TravisTorrent data would require approximately 5GB of disk space and is recommended for future work
- The microservices evaluation set of 100 runs is controlled and balanced, not representative of production traffic
- The LSTM cold-start fallback (fewer than 5 prior builds) falls back to XGBoost only
- The reinforcement learning autoscaler described in Chapter 4 is designed but not implemented in this prototype

---

## Citation

If you use this code or the methodology in your own research, please cite:

```
Shah, A. (2026) AI-Driven CI/CD Pipeline Optimisation for Microservices.
MSc Dissertation, University of West London.
Available at: https://github.com/adityashah1234/cicd-ai-predictor
```

---

## References

- Beller, M., Gousios, G. and Zaidman, A. (2017) 'Oops, my tests broke the build', *Proceedings of MSR '17*, pp. 356-367.
- Ochodek, M. and Staron, M. (2025) 'CI/CD pipeline optimization using AI: A systematic mapping study', *Engineering Proceedings*, 112(1).
- Rajesh Kumar et al. (2025) 'Optimizing CI/CD pipelines with AI-driven build failure prediction', *Spectrum of Engineering Sciences*, 3(10).
- Saidani, I., Ouni, A. and Mkaouer, M.W. (2022) 'Improving the prediction of continuous integration build failures using deep learning', *Automated Software Engineering*, 29(1).
- Lundberg, S.M. and Lee, S.I. (2017) 'A unified approach to interpreting model predictions', *NeurIPS '17*.

---

## Licence

MIT Licence. Free to use, modify, and distribute with attribution.
