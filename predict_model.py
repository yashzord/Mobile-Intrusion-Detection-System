#!/usr/bin/env python3
"""
predict_model.py – Run the prediction pipeline on newly captured data.
This version logs the anomaly details before and after whitelist filtering.
"""

import os
import json
import datetime
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Paths and constants
FEATURES_CSV = os.path.join("secure_traffic_data", "flows_features.csv")
MODEL_PATH = os.path.join("results", "autoencoder_model.h5")
RUN_LOG_PATH = os.path.join("results", "run_log.json")
ALEXA_PATH = os.path.join("whitelist", "alexa_top_1m.csv")
MAJESTIC_PATH = os.path.join("whitelist", "majestic_million.csv")
THRESHOLD_PERCENTILE = 95

# Load the scaler and model (if needed for feature transformation, assume it is used in feature_engineering.py)
scaler = joblib.load(os.path.join("results", "scaler.joblib"))
model = load_model(MODEL_PATH)

# Load engineered features from new capture (assumes that process_flows.py & feature_engineering.py have updated the file)
print("Loading engineered features from new capture...")
df = pd.read_csv(FEATURES_CSV)
if df.isnull().sum().sum() > 0:
    df = df.fillna(0)

urls = df["url"] if "url" in df.columns else pd.Series(["N/A"] * len(df))
features = df.select_dtypes(include=[np.number])
print("Data shape for prediction:", features.shape)

# Predict reconstruction errors
reconstructions = model.predict(features.values, verbose=0)
mse = np.mean(np.power(features.values - reconstructions, 2), axis=1)
threshold = np.percentile(mse, THRESHOLD_PERCENTILE)
print("Anomaly threshold (95th percentile): {:.4f}".format(threshold))
df["reconstruction_error"] = mse
df["anomaly_flag"] = df["reconstruction_error"] > threshold

pre_whitelist_anomalies = df[df["anomaly_flag"]]
pre_whitelist_urls = pre_whitelist_anomalies["url"].tolist() if "url" in df.columns else []

# --- Whitelist Post-Processing ---
print("Applying whitelist filtering on new capture...")
# Load whitelists
df_alexa = pd.read_csv(ALEXA_PATH, header=None, names=["domain"])
df_majestic = pd.read_csv(MAJESTIC_PATH, header=None, names=["domain"])
whitelist_domains = set(df_alexa["domain"].tolist() + df_majestic["domain"].tolist())

# Extract domain and apply whitelist
import re
def extract_domain(url):
    match = re.search(r'://(www\.)?([^/]+)', str(url))
    return match.group(2) if match else url

df["domain"] = df["url"].apply(extract_domain) if "url" in df.columns else "N/A"
df["whitelisted"] = df["domain"].apply(lambda d: d in whitelist_domains)
post_whitelist = df[(df["anomaly_flag"]) & (~df["whitelisted"])]
post_whitelist_urls = post_whitelist["url"].tolist() if "url" in df.columns else []

# Save outputs
results_csv = os.path.join("results", "flows_with_anomalies_predict.csv")
suspicious_csv = os.path.join("results", "suspicious_flows_predict.csv")
df.to_csv(results_csv, index=False)
post_whitelist.to_csv(suspicious_csv, index=False)
print("Prediction results written to", results_csv)
print("Suspicious flows saved to", suspicious_csv)

# Optionally, compute confusion matrix if ground truth exists.
if "true_label" in df.columns:
    y_true = df["true_label"]
    y_pred = df["anomaly_flag"].astype(int)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix (Predict)")
    cm_fig_path = os.path.join("results", "confusion_matrix_predict.png")
    plt.savefig(cm_fig_path)
    plt.close()
    print("Confusion matrix saved to", cm_fig_path)
else:
    cm = None

# Save run summary details into a log file
run_summary = {
    "run_datetime": datetime.datetime.now().isoformat(),
    "mode": "predict",
    "n_total_flows": int(len(df)),
    "anomaly_threshold": threshold,
    "pre_whitelist": {
        "n_anomalies": int(len(pre_whitelist_anomalies)),
        "urls": pre_whitelist_urls
    },
    "post_whitelist": {
        "n_anomalies": int(len(post_whitelist)),
        "urls": post_whitelist_urls
    },
    "confusion_matrix": cm.tolist() if cm is not None else None
}

if os.path.exists(RUN_LOG_PATH):
    with open(RUN_LOG_PATH, "r") as infile:
        run_log = json.load(infile)
else:
    run_log = []

run_log.append(run_summary)
with open(RUN_LOG_PATH, "w") as outfile:
    json.dump(run_log, outfile, indent=4)

print("Run summary saved to", RUN_LOG_PATH)
