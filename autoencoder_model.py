#!/usr/bin/env python3
"""
autoencoder_model.py

This script trains a deep autoencoder on the enriched numeric feature set (flows_features.csv)
to detect anomalies in mobile traffic flows. It computes reconstruction errors, sets an anomaly threshold,
and then post-processes the results by merging the original URLs and incorporating an extended whitelist.
The whitelist is merged from:
  - A base file (whitelist.txt)
  - Extended lists (alexa_top_1m.csv and majestic_million.csv in the whitelists/ folder).

This "soft" post-processing helps reduce false positives while the core model remains data‑driven.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tldextract
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping

# Define paths
base_dir = os.path.dirname(__file__)
features_csv = os.path.join(base_dir, "secure_traffic_data", "flows_features.csv")
raw_flows_csv = os.path.join(base_dir, "secure_traffic_data", "flows.csv")
RESULTS_DIR = os.path.join(base_dir, "results")
MODEL_PATH = os.path.join(RESULTS_DIR, "autoencoder_model.h5")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ----------------------------
# Load Engineered Features
# ----------------------------
print("Loading engineered features...")
df_engineered = pd.read_csv(features_csv)

# Select only numeric features for training
df_numeric = df_engineered.select_dtypes(include=[np.number])
X = df_numeric.values
print(f"Training data shape (numeric): {X.shape}")

# Fill NaN values if any
if np.any(np.isnan(X)):
    print("NaN values detected; replacing with zeros.")
    X = np.nan_to_num(X)

# ----------------------------
# Train/Test Split
# ----------------------------
print("Splitting data into training and test sets...")
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# ----------------------------
# Build the Autoencoder Model
# ----------------------------
input_dim = X_train.shape[1]
encoding_dim = input_dim // 2

print("Building autoencoder model...")
input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_layer)
encoded = Dense(encoding_dim // 2, activation='relu')(encoded)
bottleneck = Dense(encoding_dim // 4, activation='relu')(encoded)
decoded = Dense(encoding_dim // 2, activation='relu')(bottleneck)
decoded = Dense(encoding_dim, activation='relu')(decoded)
output_layer = Dense(input_dim, activation='linear')(decoded)

autoencoder = Model(inputs=input_layer, outputs=output_layer)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.summary()

# ----------------------------
# Train the Model
# ----------------------------
print("Training autoencoder...")
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = autoencoder.fit(
    X_train, X_train,
    epochs=100,
    batch_size=32,
    shuffle=True,
    validation_data=(X_test, X_test),
    callbacks=[early_stop]
)

# Save the trained model
autoencoder.save(MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

# ----------------------------
# Evaluate Model
# ----------------------------
print("Evaluating the model on test data...")
X_test_pred = autoencoder.predict(X_test)
reconstruction_errors = np.mean(np.square(X_test - X_test_pred), axis=1)
threshold = np.percentile(reconstruction_errors, 95)
print(f"Anomaly threshold (95th percentile): {threshold:.4f}")

print("Applying model to entire dataset...")
X_pred = autoencoder.predict(X)
full_errors = np.mean(np.square(X - X_pred), axis=1)
anomaly_flags = (full_errors > threshold).astype(int)

df_results = pd.DataFrame(X, columns=df_numeric.columns)
df_results["reconstruction_error"] = full_errors
df_results["anomaly"] = anomaly_flags

# ----------------------------
# Recover URL and Extract Domain
# ----------------------------
if "url" in df_engineered.columns:
    urls = df_engineered["url"]
else:
    df_raw = pd.read_csv(raw_flows_csv)
    urls = df_raw["url"] if "url" in df_raw.columns else pd.Series([""] * len(df_results))
df_results["url"] = urls.values

def extract_domain(url):
    try:
        ext = tldextract.extract(url)
        return f"{ext.domain}.{ext.suffix}".lower() if ext.domain and ext.suffix else url.lower()
    except Exception:
        return url.lower()

df_results["domain"] = df_results["url"].apply(extract_domain)

# ----------------------------
# Extended Whitelist Integration
# ----------------------------
whitelist = set()
# Load base whitelist
base_whitelist_file = os.path.join(base_dir, "whitelist.txt")
if os.path.exists(base_whitelist_file):
    with open(base_whitelist_file, "r") as f:
        whitelist.update({line.strip().lower() for line in f if line.strip()})
    print(f"Loaded base whitelist: {len(whitelist)} domains")
else:
    print("No base whitelist file found.")

# Load optional extended whitelists
whitelists_folder = os.path.join(base_dir, "whitelists")
alexa_file = os.path.join(whitelists_folder, "alexa_top_1m.csv")
majestic_file = os.path.join(whitelists_folder, "majestic_million.csv")

if os.path.exists(alexa_file):
    df_alexa = pd.read_csv(alexa_file, header=None)
    whitelist.update({str(domain).split(',')[-1].strip().lower() for domain in df_alexa[0] if pd.notna(domain)})
    print(f"After merging Alexa list: {len(whitelist)} domains total")
if os.path.exists(majestic_file):
    df_majestic = pd.read_csv(majestic_file)
    whitelist.update({str(row["Domain"]).strip().lower() for _, row in df_majestic.iterrows() if pd.notna(row["Domain"])})
    print(f"After merging Majestic Million: {len(whitelist)} domains total")

# ----------------------------
# Whitelist Post-Processing
# ----------------------------
def apply_whitelist(row):
    # If the flow is flagged as anomalous but its domain is in the whitelist, mark it as benign.
    if row["anomaly"] == 1 and row["domain"] in whitelist:
        return 0
    return row["anomaly"]

df_results["anomaly"] = df_results.apply(apply_whitelist, axis=1)
print(f"Number of anomalies after whitelist post-processing: {np.sum(df_results['anomaly'])} out of {len(df_results)} flows")

# ----------------------------
# Save Results and Plots
# ----------------------------
results_csv = os.path.join(RESULTS_DIR, "flows_with_anomalies.csv")
df_results.to_csv(results_csv, index=False)
print(f"Results written to {results_csv}")

# Plot histogram (validation set)
plt.figure(figsize=(8, 5))
plt.hist(reconstruction_errors, bins=50, edgecolor='black', alpha=0.75)
plt.xlabel("Reconstruction Error")
plt.ylabel("Count")
plt.title("Validation Reconstruction Errors")
plt.savefig(os.path.join(RESULTS_DIR, "reconstruction_error_hist.png"))
plt.close()

# Plot full error distribution
plt.figure(figsize=(8, 5))
plt.hist(full_errors, bins=50, edgecolor='black', alpha=0.75)
plt.axvline(threshold, color='red', linestyle='--', label=f"Threshold = {threshold:.4f}")
plt.xlabel("Reconstruction Error")
plt.ylabel("Count")
plt.legend()
plt.title("Final Reconstruction Error Distribution")
plt.savefig(os.path.join(RESULTS_DIR, "final_reconstruction_error_hist.png"))
plt.close()

# Save suspicious flows for further review
df_suspicious = df_results[df_results["anomaly"] == 1]
suspicious_csv = os.path.join(RESULTS_DIR, "suspicious_flows.csv")
df_suspicious.to_csv(suspicious_csv, index=False)
print(f"Suspicious flows saved to {suspicious_csv}")
