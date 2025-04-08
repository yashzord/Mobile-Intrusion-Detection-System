#!/usr/bin/env python3
import os
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from urllib.parse import urlparse
from datetime import datetime

# Define paths
DB_PATH = "/home/kali/Mobile-Intrusion-Detection-System/traffic_data.db"
IMAGES_DIR = "/home/kali/Mobile-Intrusion-Detection-System/images"
METRICS_FILE = "/home/kali/Mobile-Intrusion-Detection-System/model_unsupervised_metrics.txt"
RUN_COUNTER_FILE = "/home/kali/Mobile-Intrusion-Detection-System/run_unsupervised_counter.txt"

# --- Step 1: Load Cumulative Raw Data from Database ---
def load_raw_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM flows", conn)
    conn.close()
    return df

print("Loading cumulative raw data from the database...")
raw_df = load_raw_data()
print(f"Loaded {raw_df.shape[0]} raw records.")

# --- Step 2: Feature Engineering ---
def feature_engineering(df):
    # Convert timestamp_start to datetime and extract features.
    df['timestamp_start'] = pd.to_datetime(df['timestamp_start'], unit='s', errors='coerce')
    df['hour'] = df['timestamp_start'].dt.hour
    df['day_of_week'] = df['timestamp_start'].dt.dayofweek

    # Extract URL features.
    def parse_url(url):
        try:
            parsed = urlparse(str(url))
            return pd.Series({
                "domain": parsed.netloc,
                "path_depth": len([p for p in parsed.path.strip("/").split("/") if p]),
                "query_length": len(parsed.query),
                "is_https": parsed.scheme.lower() == "https"
            })
        except Exception:
            return pd.Series({
                "domain": "",
                "path_depth": 0,
                "query_length": 0,
                "is_https": False
            })
    url_features = df['url'].apply(parse_url)
    df = pd.concat([df, url_features], axis=1)

    # One-hot encode the HTTP method.
    if "method" in df.columns:
        df = pd.get_dummies(df, columns=["method"], prefix="method")

    # Create a status category.
    df["status_category"] = df["status_code"].fillna(0).apply(lambda x: int(x / 100))

    # Log-scale sizes.
    df["log_request_size"] = np.log1p(df["request_content_length"].astype(float))
    df["log_response_size"] = np.log1p(df["response_content_length"].astype(float))

    # TLS indicators.
    df["tls_failed"] = df["tls_established"].apply(lambda x: True if str(x).lower() == "false" else False)
    df["tls_missing"] = df["tls_established"].isna()

    # Flag nonstandard port.
    if "server_port" in df.columns:
        df["nonstandard_port"] = ~df["server_port"].isin([80, 443])

    # Drop columns that are not needed.
    drop_cols = [
        "request_headers", "response_headers",
        "timestamp_end", "http_version",
        "request_content_length", "response_content_length",
        "client_ip", "client_port", "server_ip", "server_port",
        "tls_cipher", "alpn", "sni", "user_agent", "content_encoding"
    ]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    return df

print("Performing feature engineering on cumulative data...")
feature_df = feature_engineering(raw_df)
print(f"Engineered feature dataset shape: {feature_df.shape}")

# --- Step 3: Data Preprocessing (Scaling) ---
numeric_cols = feature_df.select_dtypes(include=["number"]).columns
feature_df[numeric_cols] = feature_df[numeric_cols].fillna(0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(feature_df[numeric_cols])
X = pd.DataFrame(X_scaled, columns=numeric_cols)

# --- Step 4: Load Ground Truth Labels for Evaluation (if available) ---
def load_labels():
    conn = sqlite3.connect(DB_PATH)
    try:
        df_labels = pd.read_sql_query("SELECT * FROM labeled", conn)
    except Exception:
        df_labels = pd.DataFrame()
    conn.close()
    return df_labels

labels_df = load_labels()
if not labels_df.empty and "label" in labels_df.columns:
    if len(labels_df) == len(X):
        y_true = labels_df["label"]
    else:
        print("Warning: Labeled data count does not match engineered data. Evaluation may be inaccurate.")
        y_true = np.zeros(len(X))
else:
    y_true = np.zeros(len(X))
    print("No labeled data found. Using zeros for ground truth evaluation.")

# --- Step 5: Unsupervised Modeling ---
results = {}  # To store performance metrics.
# -- RUN COUNTER for unsupervised --
if os.path.exists(RUN_COUNTER_FILE):
    with open(RUN_COUNTER_FILE, "r") as f:
        try:
            run_number = int(f.read().strip()) + 1
        except:
            run_number = 1
else:
    run_number = 1
with open(RUN_COUNTER_FILE, "w") as f:
    f.write(str(run_number))
run_label = f"Run{run_number}"

unique_suffix = run_label  # Use explicit run label instead of a timestamp.

# Isolation Forest
print("Training Isolation Forest...")
iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
iso_forest.fit(X)
scores_if = -iso_forest.decision_function(X)
scores_if_norm = (scores_if - scores_if.min()) / (scores_if.max() - scores_if.min())
if len(np.unique(y_true)) > 1:
    auc_if = roc_auc_score(y_true, scores_if_norm)
else:
    auc_if = float("nan")
results["Isolation Forest"] = auc_if
print(f"Isolation Forest ROC AUC Score: {auc_if:.3f}")
if len(np.unique(y_true)) > 1:
    fpr, tpr, _ = roc_curve(y_true, scores_if_norm)
else:
    fpr, tpr = [], []
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"Isolation Forest ROC (AUC = {auc_if:.2f})")
plt.plot([0,1], [0,1], "k--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"Isolation Forest ROC Curve - {run_label}")
plt.legend(loc="best")
iso_fig_name = f"unsupervised_roc_curve_if_{run_label}.png"
plt.savefig(os.path.join(IMAGES_DIR, iso_fig_name))
plt.close()

# One-Class SVM
print("Training One-Class SVM...")
ocsvm = OneClassSVM(nu=0.05, kernel="rbf", gamma="scale")
ocsvm.fit(X)
scores_ocsvm = -ocsvm.decision_function(X)
scores_ocsvm_norm = (scores_ocsvm - scores_ocsvm.min()) / (scores_ocsvm.max() - scores_ocsvm.min())
if len(np.unique(y_true)) > 1:
    auc_ocsvm = roc_auc_score(y_true, scores_ocsvm_norm)
else:
    auc_ocsvm = float("nan")
results["One-Class SVM"] = auc_ocsvm
print(f"One-Class SVM ROC AUC Score: {auc_ocsvm:.3f}")
if len(np.unique(y_true)) > 1:
    fpr, tpr, _ = roc_curve(y_true, scores_ocsvm_norm)
else:
    fpr, tpr = [], []
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"One-Class SVM ROC (AUC = {auc_ocsvm:.2f})")
plt.plot([0,1], [0,1], "k--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"One-Class SVM ROC Curve - {run_label}")
plt.legend(loc="best")
ocsvm_fig_name = f"unsupervised_roc_curve_ocsvm_{run_label}.png"
plt.savefig(os.path.join(IMAGES_DIR, ocsvm_fig_name))
plt.close()

# Local Outlier Factor (LOF)
print("Training Local Outlier Factor (LOF)...")
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=False)
lof.fit(X)
scores_lof = -lof.negative_outlier_factor_
scores_lof_norm = (scores_lof - scores_lof.min()) / (scores_lof.max() - scores_lof.min())
if len(np.unique(y_true)) > 1:
    auc_lof = roc_auc_score(y_true, scores_lof_norm)
else:
    auc_lof = float("nan")
results["LOF"] = auc_lof
print(f"LOF ROC AUC Score: {auc_lof:.3f}")
if len(np.unique(y_true)) > 1:
    fpr, tpr, _ = roc_curve(y_true, scores_lof_norm)
else:
    fpr, tpr = [], []
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"LOF ROC (AUC = {auc_lof:.2f})")
plt.plot([0,1], [0,1], "k--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"LOF ROC Curve - {run_label}")
plt.legend(loc="best")
lof_fig_name = f"unsupervised_roc_curve_lof_{run_label}.png"
plt.savefig(os.path.join(IMAGES_DIR, lof_fig_name))
plt.close()

# --- Step 6: Append Performance Metrics to Log File ---
metrics_text = f"Run: {run_label}\n"
for method, auc in results.items():
    metrics_text += f"{method} ROC AUC Score: {auc:.3f}\n"
metrics_text += f"Isolation Forest ROC Curve: {iso_fig_name}\n"
metrics_text += f"One-Class SVM ROC Curve: {ocsvm_fig_name}\n"
metrics_text += f"LOF ROC Curve: {lof_fig_name}\n\n"

with open(METRICS_FILE, "a") as f:
    f.write(metrics_text)

print("Unsupervised modeling complete.")
print(f"ROC curve images saved as:\n  {iso_fig_name}\n  {ocsvm_fig_name}\n  {lof_fig_name}")
print("Performance metrics appended to", METRICS_FILE)
