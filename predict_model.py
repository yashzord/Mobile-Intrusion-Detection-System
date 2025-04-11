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
import re
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Paths and constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES_CSV = os.path.join(BASE_DIR, "secure_traffic_data", "flows_features.csv")
MODEL_PATH = os.path.join(BASE_DIR, "results", "autoencoder_model.h5")
RUN_LOG_PATH = os.path.join(BASE_DIR, "results", "run_log.json")
ALEXA_PATH = os.path.join(BASE_DIR, "whitelist", "alexa_top_1m.csv")
MAJESTIC_PATH = os.path.join(BASE_DIR, "whitelist", "majestic_million.csv")
THRESHOLD_PERCENTILE = 95

# Load the scaler and model
scaler_path = os.path.join(BASE_DIR, "results", "scaler.joblib")
scaler = joblib.load(scaler_path)
print("Scaler loaded from", scaler_path)

print("Loading the trained model...")
# Use compile=False to avoid deserializing lost functions/metrics such as 'mse'
model = load_model(MODEL_PATH, compile=False)
print("Model loaded successfully.")

# Load engineered features from new capture
print("Loading engineered features from new capture...")
df = pd.read_csv(FEATURES_CSV)
if df.isnull().sum().sum() > 0:
    print("NaN values detected in features; replacing with zeros.")
    df = df.fillna(0)

# Ensure the 'url' column is present
if "url" not in df.columns:
    df["url"] = "N/A"
urls = df["url"]

# Select only numeric features for prediction.
# Keeping the DataFrame with its column names ensures that the order remains consistent.
features = df.select_dtypes(include=[np.number])
print("Data shape for prediction:", features.shape)

# Scale the features using the loaded scaler.
features_scaled = scaler.transform(features)
print("Data scaled using the loaded scaler.")

# Run predictions on the scaled features.
print("Running predictions on new data...")
reconstructions = model.predict(features_scaled, verbose=0)
mse = np.mean(np.power(features_scaled - reconstructions, 2), axis=1)
threshold = np.percentile(mse, THRESHOLD_PERCENTILE)
print("Anomaly threshold (95th percentile): {:.4f}".format(threshold))
df["reconstruction_error"] = mse
df["anomaly_flag"] = df["reconstruction_error"] > threshold

# Log pre‐whitelist anomalies
pre_whitelist_anomalies = df[df["anomaly_flag"]].copy()
def extract_domain(url):
    """Extract the domain from a given URL using regex."""
    match = re.search(r'://(?:www\.)?([^/]+)', str(url))
    return match.group(1).lower() if match else str(url).lower()
pre_whitelist_anomalies["domain"] = pre_whitelist_anomalies["url"].apply(extract_domain)
pre_whitelist_urls = pre_whitelist_anomalies["url"].unique().tolist()
print("Number of anomalies before whitelist post-processing: {} out of {} flows".format(
    int(df["anomaly_flag"].sum()), len(df)
))
print("Pre‑whitelist anomalies: {} unique URLs flagged.".format(len(pre_whitelist_urls)))

# --- Whitelist Post-Processing ---
print("Applying whitelist filtering on new capture...")
# Load and clean whitelist domains
df_alexa = pd.read_csv(ALEXA_PATH, header=None, names=["domain"])
df_majestic = pd.read_csv(MAJESTIC_PATH, header=None, names=["domain"])
whitelist_domains = set(df_alexa["domain"].dropna().str.strip().str.lower().tolist() +
                        df_majestic["domain"].dropna().str.strip().str.lower().tolist())
print("After merging whitelists: {} domains total.".format(len(whitelist_domains)))

# Apply whitelist filtering by checking extracted domain names
df["domain"] = df["url"].apply(extract_domain)
df["whitelisted"] = df["domain"].apply(lambda d: d in whitelist_domains)
post_whitelist = df[(df["anomaly_flag"]) & (~df["whitelisted"])].copy()
post_whitelist_urls = post_whitelist["url"].unique().tolist()
print("Post‑whitelist anomalies: {} unique URLs remain after filtering.".format(len(post_whitelist_urls)))
print("Number of anomalies after whitelist post-processing: {} out of {} flows".format(
    int(post_whitelist["anomaly_flag"].sum()), len(df)
))

# Save prediction outputs to CSV files.
results_csv = os.path.join(BASE_DIR, "results", "flows_with_anomalies_predict.csv")
suspicious_csv = os.path.join(BASE_DIR, "results", "suspicious_flows_predict.csv")
df.to_csv(results_csv, index=False)
post_whitelist.to_csv(suspicious_csv, index=False)
print("Prediction results written to", results_csv)
print("Suspicious flows saved to", suspicious_csv)

# Optionally, compute and save confusion matrix if ground truth labels exist.
if "true_label" in df.columns:
    y_true = df["true_label"]
    y_pred = df["anomaly_flag"].astype(int)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix (Predict)")
    cm_fig_path = os.path.join(BASE_DIR, "results", "confusion_matrix_predict.png")
    plt.savefig(cm_fig_path)
    plt.close()
    print("Confusion matrix saved to", cm_fig_path)
else:
    cm = None

# Log run summary details
run_summary = {
    "run_datetime": datetime.datetime.now().isoformat(),
    "mode": "predict",
    "n_total_flows": int(len(df)),
    "anomaly_threshold": threshold,
    "pre_whitelist": {
        "n_anomalies": int(df["anomaly_flag"].sum()),
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

print("Prediction run complete. Check the results/ directory for outputs and metrics.")
