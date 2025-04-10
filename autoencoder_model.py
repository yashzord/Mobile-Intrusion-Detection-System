#!/usr/bin/env python3
"""
autoencoder_model.py – Train the autoencoder and perform anomaly detection.
This version saves both pre‑whitelist and post‑whitelist metrics and URLs.
"""

import os
import json
import datetime
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tldextract
from tensorflow.keras import layers, Model, Input, callbacks
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Paths and Constants
FEATURES_CSV = os.path.join("secure_traffic_data", "flows_features.csv")
MODEL_PATH = os.path.join("results", "autoencoder_model.h5")
SCALER_PATH = os.path.join("results", "scaler.joblib")
RUN_LOG_PATH = os.path.join("results", "run_log.json")
ALEXA_PATH = os.path.join("whitelist", "alexa_top_1m.csv")
MAJESTIC_PATH = os.path.join("whitelist", "majestic_million.csv")
THRESHOLD_PERCENTILE = 95  # Change threshold if needed

# Create results directory if not exists
os.makedirs("results", exist_ok=True)

# -------------------------------
# Load Engineered Features
# -------------------------------
print("Loading engineered features...")
df = pd.read_csv(FEATURES_CSV)
if df.isnull().sum().sum() > 0:
    print("NaN values detected; replacing with zeros.")
    df = df.fillna(0)

# Ensure a "url" column exists; if missing, throw an error.
if "url" not in df.columns:
    raise ValueError("The 'url' column is missing from the engineered features file.")

# Save the original URL values (for reporting later)
urls = df["url"]

# Select only the numeric features for training
features = df.select_dtypes(include=[np.number])
print("Training data shape (numeric):", features.shape)

# -------------------------------
# Split into Training and Test Sets
# -------------------------------
X_train, X_test = train_test_split(features, test_size=0.2, random_state=42)

# -------------------------------
# Build the Autoencoder Model
# -------------------------------
input_dim = X_train.shape[1]
encoding_dim = input_dim // 2

print("Building autoencoder model...")
input_layer = Input(shape=(input_dim,))
encoded = layers.Dense(encoding_dim, activation='relu')(input_layer)
encoded = layers.Dense(encoding_dim // 2, activation='relu')(encoded)
bottleneck = layers.Dense(encoding_dim // 4, activation='relu')(encoded)
decoded = layers.Dense(encoding_dim // 2, activation='relu')(bottleneck)
decoded = layers.Dense(encoding_dim, activation='relu')(decoded)
output_layer = layers.Dense(input_dim, activation='linear')(decoded)

autoencoder = Model(inputs=input_layer, outputs=output_layer)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.summary()

# -------------------------------
# Train the Model
# -------------------------------
print("Training autoencoder...")
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = autoencoder.fit(
    X_train, X_train,
    epochs=100,
    batch_size=32,
    shuffle=True,
    validation_data=(X_test, X_test),
    callbacks=[early_stop],
    verbose=1
)

# Save the trained model
autoencoder.save(MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

# -------------------------------
# Evaluate Model on Test Data
# -------------------------------
print("Evaluating the model on test data...")
test_loss = autoencoder.evaluate(X_test, X_test, verbose=0)
print("Test loss: {:.4f}".format(test_loss))

# -------------------------------
# Full Dataset Evaluation
# -------------------------------
print("Loading engineered features for full dataset evaluation...")
X_full = features.values  # All numeric features
X_pred = autoencoder.predict(X_full, verbose=0)
mse = np.mean(np.power(X_full - X_pred, 2), axis=1)
threshold = np.percentile(mse, THRESHOLD_PERCENTILE)
print("Anomaly threshold (95th percentile): {:.4f}".format(threshold))

# Mark anomalies on the full dataset (before whitelist filtering)
df["reconstruction_error"] = mse
df["anomaly_flag"] = df["reconstruction_error"] > threshold

# -------------------------------
# Pre‑Whitelist Anomaly Logging
# -------------------------------
def extract_domain(url):
    ext = tldextract.extract(str(url))
    if ext.domain and ext.suffix:
        return f"{ext.domain}.{ext.suffix}".lower()
    return str(url).lower()

pre_whitelist = df[df["anomaly_flag"]].copy()
# Add domain extraction for logging
pre_whitelist["domain"] = pre_whitelist["url"].apply(extract_domain)
pre_whitelist_urls = pre_whitelist["url"].unique().tolist()
print("Number of anomalies before whitelist post-processing: {} out of {} flows".format(len(pre_whitelist), len(df)))
print("Pre‑whitelist anomalies: {} unique URLs flagged.".format(len(pre_whitelist_urls)))

# -------------------------------
# Whitelist Post-Processing
# -------------------------------
print("Applying whitelist filtering...")
# Load whitelists (both files are in the 'whitelist' folder)
df_alexa = pd.read_csv(ALEXA_PATH, header=None, names=["domain"])
df_majestic = pd.read_csv(MAJESTIC_PATH, header=None, names=["domain"])
# Create a set of whitelist domains (convert to lowercase and strip whitespace)
whitelist_domains = set(
    df_alexa["domain"].dropna().str.strip().str.lower().tolist() +
    df_majestic["domain"].dropna().str.strip().str.lower().tolist()
)
print(f"After merging Alexa list: {len(df_alexa['domain'].dropna())} domains total")
print(f"After merging Majestic Million: {len(df_majestic['domain'].dropna())} domains total")

# Add domain information to the full dataframe using the robust extractor
df["domain"] = df["url"].apply(extract_domain)
# Mark records as whitelisted if the extracted domain is found in the whitelist
df["whitelisted"] = df["domain"].apply(lambda d: d in whitelist_domains)

# Remove whitelisted flows from the anomalies
post_whitelist = df[(df["anomaly_flag"]) & (~df["whitelisted"])].copy()
post_whitelist_urls = post_whitelist["url"].unique().tolist()
print("Post‑whitelist anomalies: {} unique URLs remain after filtering.".format(len(post_whitelist_urls)))
print("Number of anomalies after whitelist post-processing: {} out of {} flows".format(len(post_whitelist), len(df)))

# -------------------------------
# Save Results and Plots
# -------------------------------
results_csv = os.path.join("results", "flows_with_anomalies.csv")
suspicious_csv = os.path.join("results", "suspicious_flows.csv")
df.to_csv(results_csv, index=False)
post_whitelist.to_csv(suspicious_csv, index=False)
print("Results written to", results_csv)
print("Suspicious flows saved to", suspicious_csv)

# Optionally create a confusion matrix if true labels exist in the data.
if "true_label" in df.columns:
    y_true = df["true_label"]
    y_pred = df["anomaly_flag"].astype(int)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix")
    cm_fig_path = os.path.join("results", "confusion_matrix.png")
    plt.savefig(cm_fig_path)
    plt.close()
    print("Confusion matrix saved to", cm_fig_path)
else:
    cm = None

# -------------------------------
# Save Run Summary Log
# -------------------------------
run_summary = {
    "run_datetime": datetime.datetime.now().isoformat(),
    "mode": "train",
    "n_total_flows": int(len(df)),
    "anomaly_threshold": float(threshold),
    "pre_whitelist": {
        "n_anomalies": int(len(pre_whitelist)),
        "urls": pre_whitelist_urls
    },
    "post_whitelist": {
        "n_anomalies": int(len(post_whitelist)),
        "urls": post_whitelist_urls
    },
    "test_loss": float(test_loss),
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
