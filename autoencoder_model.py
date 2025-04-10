#!/usr/bin/env python3
"""
autoencoder_model.py

Train a deep autoencoder for anomaly detection on the enriched numeric feature set.
This script loads engineered features from flows_features.csv (selecting only numeric columns),
checks for and fills any NaN values, splits the data, builds and trains the autoencoder
with early stopping, and then performs internal analysis by computing reconstruction errors.

After anomaly detection, it merges the original URL for each flow (sourced from flows.csv)
so you can inspect the URLs of the detected anomalies. Additionally, it extracts the domain
from each URL, loads a whitelist from whitelist.txt, and for flows where the domain is whitelisted,
the anomaly flag is overridden (marked as benign).

Outputs:
 - The trained model is saved to results/autoencoder_model.h5.
 - Histograms and anomaly detection CSVs are saved to the results/ folder.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tldextract  # Make sure to install: pip install tldextract
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
WHITELIST_FILE = os.path.join(base_dir, "whitelist.txt")
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Load Engineered Features ---
print("Loading engineered features...")
df_engineered = pd.read_csv(features_csv)

# Select only numeric columns for training the autoencoder.
df_numeric = df_engineered.select_dtypes(include=[np.number])
X = df_numeric.values
print(f"Data shape for training (numeric only): {X.shape}")

# Fill any NaN values with zeros.
if np.any(np.isnan(X)):
    print("Warning: NaN values detected in training data. Filling NaNs with 0.")
    X = np.nan_to_num(X)

# --- Train/Test Split ---
print("Splitting data into training and testing sets...")
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# --- Build the Autoencoder Model ---
input_dim = X_train.shape[1]
encoding_dim = int(input_dim / 2)  # Experiment with this value if needed.

print("Building the autoencoder model...")
input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_layer)
encoded = Dense(int(encoding_dim / 2), activation='relu')(encoded)
bottleneck = Dense(int(encoding_dim / 4), activation='relu')(encoded)
decoded = Dense(int(encoding_dim / 2), activation='relu')(bottleneck)
decoded = Dense(encoding_dim, activation='relu')(decoded)
output_layer = Dense(input_dim, activation='linear')(decoded)

autoencoder = Model(inputs=input_layer, outputs=output_layer)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.summary()

# --- Train the Model with Early Stopping ---
print("Training the autoencoder model...")
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = autoencoder.fit(
    X_train, X_train,
    epochs=100,
    batch_size=32,
    shuffle=True,
    validation_data=(X_test, X_test),
    callbacks=[early_stop]
)

# Save the trained model.
autoencoder.save(MODEL_PATH)
print(f"✅ Autoencoder model saved to {MODEL_PATH}")

# --- Evaluate the Model on Test Data ---
print("Evaluating the model on the test set...")
X_test_pred = autoencoder.predict(X_test)
reconstruction_errors = np.mean(np.square(X_test - X_test_pred), axis=1)

# Print reconstruction error summary.
print("\n--- Reconstruction Error Summary on Test Data ---")
print(f"Mean Error: {np.mean(reconstruction_errors):.4f}")
print(f"Std Dev: {np.std(reconstruction_errors):.4f}")
print(f"Median Error: {np.median(reconstruction_errors):.4f}")
print(f"Min Error: {np.min(reconstruction_errors):.4f}")
print(f"Max Error: {np.max(reconstruction_errors):.4f}\n")

# Save histogram of reconstruction errors for test set.
plt.figure(figsize=(8, 5))
plt.hist(reconstruction_errors, bins=50, alpha=0.75, edgecolor='black')
plt.xlabel("Reconstruction Error")
plt.ylabel("Number of Samples")
plt.title("Histogram of Reconstruction Errors on Test Data")
hist_path = os.path.join(RESULTS_DIR, "reconstruction_error_hist.png")
plt.savefig(hist_path)
plt.close()
print(f"✅ Reconstruction error histogram saved at {hist_path}")

# --- Set Anomaly Threshold (95th Percentile) ---
threshold = np.percentile(reconstruction_errors, 95)
print(f"Recommended anomaly threshold (95th percentile): {threshold:.4f}")

# --- Apply Autoencoder to Entire Dataset ---
print("Applying the model to the entire dataset to flag anomalies...")
X_pred = autoencoder.predict(X)
full_errors = np.mean(np.square(X - X_pred), axis=1)
anomaly_flags = (full_errors > threshold).astype(int)

print("\n--- Overall Anomaly Detection Analysis ---")
print(f"Total Samples: {len(full_errors)}")
print(f"Initial Anomalies Detected: {np.sum(anomaly_flags)}")

# --- Prepare a DataFrame with Results ---
df_results = pd.DataFrame(X, columns=df_numeric.columns)
df_results["reconstruction_error"] = full_errors
df_results["anomaly"] = anomaly_flags

# --- Recover Original URL for Each Flow ---
if "url" in df_engineered.columns:
    urls = df_engineered["url"]
else:
    print("URL column not found in engineered data; loading from raw flows.csv...")
    df_raw = pd.read_csv(raw_flows_csv)
    urls = df_raw["url"] if "url" in df_raw.columns else pd.Series([""] * len(df_results))
df_results["url"] = urls.values

# --- Extract Domain from URL using tldextract ---
def extract_domain(url):
    try:
        ext = tldextract.extract(url)
        if ext.domain and ext.suffix:
            return f"{ext.domain}.{ext.suffix}".lower()
        return url.lower()
    except Exception:
        return url.lower()

df_results["domain"] = df_results["url"].apply(extract_domain)

# --- Load Whitelist and Override Anomaly Flag for Trusted Domains ---
if os.path.exists(WHITELIST_FILE):
    with open(WHITELIST_FILE, "r") as f:
        whitelist = {line.strip().lower() for line in f if line.strip()}
    print(f"Loaded whitelist with {len(whitelist)} domains.")
else:
    whitelist = set()
    print("⚠️ Whitelist file not found. No whitelisting will be applied.")

# Override anomaly flag if the domain is whitelisted.
def apply_whitelist(row):
    if row["anomaly"] == 1 and row["domain"] in whitelist:
        return 0  # Override: mark as benign.
    return row["anomaly"]

df_results["anomaly"] = df_results.apply(apply_whitelist, axis=1)
final_anomaly_count = np.sum(df_results["anomaly"])
print(f"✅ After whitelist override, total anomalies: {final_anomaly_count} out of {len(df_results)} flows.")

# Save the complete anomaly results (including URL and domain) to CSV.
results_csv = os.path.join(RESULTS_DIR, "flows_with_anomalies.csv")
df_results.to_csv(results_csv, index=False)
print(f"✅ Flows with anomaly flags saved to {results_csv}")

# --- Plot Final Error Distribution with the Threshold ---
plt.figure(figsize=(8, 5))
plt.hist(full_errors, bins=50, alpha=0.75, edgecolor='black')
plt.axvline(threshold, color='red', linestyle='--', label=f"Threshold = {threshold:.4f}")
plt.xlabel("Reconstruction Error")
plt.ylabel("Number of Samples")
plt.legend()
plt.title("Final Reconstruction Error Distribution")
final_hist_path = os.path.join(RESULTS_DIR, "final_reconstruction_error_hist.png")
plt.savefig(final_hist_path)
plt.close()
print(f"✅ Final error histogram with threshold saved at {final_hist_path}")

print("\n✅ Autoencoder training, evaluation, and whitelist post–processing complete!")

# --- Save Only Suspicious Flows for Manual Review ---
df_suspicious = df_results[df_results["anomaly"] == 1]
suspicious_csv = os.path.join(RESULTS_DIR, "suspicious_flows.csv")
df_suspicious.to_csv(suspicious_csv, index=False)
print(f"✅ Suspicious flows (after whitelist override) saved to {suspicious_csv}")
