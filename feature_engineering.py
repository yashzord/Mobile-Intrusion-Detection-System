#!/usr/bin/env python3
"""
feature_engineering.py

Performs advanced feature engineering on raw flows data (from secure_traffic_data/flows.csv).
This script extracts temporal features, URL properties, applies TF‑IDF to URLs, integrates external threat
intelligence, and performs encoding and scaling. The engineered features (with the original URL preserved)
are saved to secure_traffic_data/flows_features.csv. In addition, the fitted scaler and TF‑IDF vectorizer
are dumped to the results/ folder for use during prediction.
"""

import os
import pandas as pd
import numpy as np
from urllib.parse import urlparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import joblib

# Define file paths
base_dir = os.path.dirname(__file__)
raw_csv = os.path.join(base_dir, "secure_traffic_data", "flows.csv")
output_csv = os.path.join(base_dir, "secure_traffic_data", "flows_features.csv")
external_intel_csv = os.path.join(base_dir, "external_data", "external_threats.csv")
SCALER_OUT = os.path.join(base_dir, "results", "scaler.joblib")
TFIDF_OUT = os.path.join(base_dir, "results", "tfidf_vectorizer.joblib")

# Read raw flows data
df = pd.read_csv(raw_csv)

# --- Temporal Features ---
# Convert timestamp to datetime (adjust unit if needed)
df['timestamp_start'] = pd.to_datetime(df['timestamp_start'], unit='s', errors='coerce')
df['hour'] = df['timestamp_start'].dt.hour
df['day_of_week'] = df['timestamp_start'].dt.dayofweek
df = df.sort_values("timestamp_start")
df['time_gap'] = df['timestamp_start'].diff().dt.total_seconds().fillna(0)

# --- Basic URL Feature Extraction ---
def parse_url_basic(url):
    """
    Extracts basic URL properties:
      - domain (extracted using urlparse)
      - path depth (number of non-empty segments in the path)
      - query length (number of characters in the query)
      - is_https flag
    """
    try:
        parsed = urlparse(str(url))
        return pd.Series({
            "domain_extracted": parsed.netloc,
            "path_depth": len([p for p in parsed.path.strip("/").split("/") if p]),
            "query_length": len(parsed.query),
            "is_https": 1 if parsed.scheme.lower() == "https" else 0
        })
    except Exception:
        return pd.Series({"domain_extracted": "", "path_depth": 0, "query_length": 0, "is_https": 0})

# Ensure the 'url' column exists; otherwise, raise an error.
if "url" not in df.columns:
    raise ValueError("The 'url' column is missing from flows.csv. Please ensure raw flows include the URL.")

# Apply URL basic parsing
url_basic = df['url'].apply(parse_url_basic)
df = pd.concat([df, url_basic], axis=1)

# --- TF-IDF Extraction from URL ---
# Apply TF-IDF vectorization on the URL field (fill missing with empty string)
tfidf = TfidfVectorizer(ngram_range=(2, 4), max_features=50)
tfidf_features = tfidf.fit_transform(df['url'].fillna("")).toarray()
tfidf_feature_names = [f"tfidf_{i}" for i in range(tfidf_features.shape[1])]
df_tfidf = pd.DataFrame(tfidf_features, columns=tfidf_feature_names, index=df.index)
df = pd.concat([df, df_tfidf], axis=1)

# --- External Threat Intelligence Integration ---
if os.path.exists(external_intel_csv):
    df_intel = pd.read_csv(external_intel_csv)
    # Merge on the extracted domain; adjust the column name if necessary
    df = df.merge(df_intel[['domain', 'threat_score']], left_on="domain_extracted", right_on="domain", how="left")
    df["threat_score"] = df["threat_score"].fillna(0)
else:
    print("External threat intelligence file not found. Skipping integration.")
    df["threat_score"] = 0

# --- One-hot Encoding for HTTP Methods (if present) ---
if "method" in df.columns:
    df = pd.get_dummies(df, columns=["method"], prefix="method")

# --- Log-scale Sizes ---
if "request_content_length" in df.columns:
    df["log_request_size"] = np.log1p(df["request_content_length"].astype(float))
if "response_content_length" in df.columns:
    df["log_response_size"] = np.log1p(df["response_content_length"].astype(float))

# --- Nonstandard Port Flag ---
if "server_port" in df.columns:
    df["nonstandard_port"] = df["server_port"].apply(lambda x: 0 if x in [80, 443] else 1)

# --- Drop Unnecessary Columns ---
# Do not drop "url" so that it remains in the engineered file.
drop_cols = [
    "request_headers", "response_headers", "http_version", "user_agent",
    "content_encoding", "tls_cipher", "alpn", "sni", "client_ip", "client_port",
    "server_ip", "server_port"
]
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

# --- Scaling of Numeric Features ---
numeric_cols = df.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

joblib.dump(scaler, SCALER_OUT)
print(f"Scaler saved to {SCALER_OUT}")

joblib.dump(tfidf, TFIDF_OUT)
print(f"TF-IDF Vectorizer saved to {TFIDF_OUT}")

# Save the engineered features (with the original URL preserved)
df.to_csv(output_csv, index=False)
print(f"Engineered features saved to {output_csv}")
print(f"Final feature shape: {df.shape[0]} rows x {df.shape[1]} columns")
