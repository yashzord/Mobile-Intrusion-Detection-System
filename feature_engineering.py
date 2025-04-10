#!/usr/bin/env python3
"""
feature_engineering.py

This script generates an enriched feature set from raw flows by combining:
  - Temporal features extracted from timestamps.
  - URL parsing to derive domain, path depth, query length, and HTTPS flag.
  - Textual features using TF‑IDF on URLs.
  - Integration of external threat intelligence.
  - Additional encoding for HTTP methods, log-scaling of size values, and flagging nonstandard ports.

The final output is saved to secure_traffic_data/flows_features.csv.
"""

import os
import pandas as pd
import numpy as np
from urllib.parse import urlparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import joblib

# Define paths
base_dir = os.path.dirname(__file__)
raw_csv = os.path.join(base_dir, "secure_traffic_data", "flows.csv")
output_csv = os.path.join(base_dir, "secure_traffic_data", "flows_features.csv")
external_intel_csv = os.path.join(base_dir, "external_data", "external_threats.csv")

# Load raw flows data
df = pd.read_csv(raw_csv)

# --- Temporal Features ---
df['timestamp_start'] = pd.to_datetime(df['timestamp_start'], unit='s', errors='coerce')
df['hour'] = df['timestamp_start'].dt.hour
df['day_of_week'] = df['timestamp_start'].dt.dayofweek
df = df.sort_values("timestamp_start")
df['time_gap'] = df['timestamp_start'].diff().dt.total_seconds().fillna(0)

# --- Basic URL Feature Extraction ---
def parse_url_basic(url):
    try:
        parsed = urlparse(str(url))
        return pd.Series({
            "domain": parsed.netloc,
            "path_depth": len([p for p in parsed.path.strip("/").split("/") if p]),
            "query_length": len(parsed.query),
            "is_https": 1 if parsed.scheme.lower() == "https" else 0
        })
    except Exception:
        return pd.Series({"domain": "", "path_depth": 0, "query_length": 0, "is_https": 0})

url_basic = df['url'].apply(parse_url_basic)
df = pd.concat([df, url_basic], axis=1)

# --- TF-IDF Extraction from URL ---
tfidf = TfidfVectorizer(ngram_range=(2, 4), max_features=50)
tfidf_features = tfidf.fit_transform(df['url'].fillna("")).toarray()
tfidf_feature_names = [f"tfidf_{i}" for i in range(tfidf_features.shape[1])]
df_tfidf = pd.DataFrame(tfidf_features, columns=tfidf_feature_names, index=df.index)
df = pd.concat([df, df_tfidf], axis=1)

# --- External Threat Intelligence Integration ---
if os.path.exists(external_intel_csv):
    df_intel = pd.read_csv(external_intel_csv)
    df = df.merge(df_intel[['domain', 'threat_score']], on="domain", how="left")
    df["threat_score"] = df["threat_score"].fillna(0)
else:
    print("External threat intelligence file not found. Skipping integration.")
    df["threat_score"] = 0

# --- One-hot Encoding for HTTP Methods ---
if "method" in df.columns:
    df = pd.get_dummies(df, columns=["method"], prefix="method")

# --- Log-scale Request/Response Sizes ---
df["log_request_size"] = np.log1p(df["request_content_length"].astype(float))
df["log_response_size"] = np.log1p(df["response_content_length"].astype(float))

# --- Nonstandard Port Flag ---
if "server_port" in df.columns:
    df["nonstandard_port"] = df["server_port"].apply(lambda x: 0 if x in [80, 443] else 1)

# --- Drop Unnecessary Columns ---
drop_cols = ["request_headers", "response_headers", "http_version",
             "user_agent", "content_encoding", "tls_cipher", "alpn",
             "sni", "client_ip", "client_port", "server_ip", "server_port", "url"]
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

# --- Standardize Numeric Features ---
numeric_cols = df.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Save the scaler for future inference
scaler_path = os.path.join(base_dir, "results", "scaler.joblib")
os.makedirs(os.path.join(base_dir, "results"), exist_ok=True)
joblib.dump(scaler, scaler_path)

# Save engineered features
df.to_csv(output_csv, index=False)
print(f"Advanced feature engineering complete. Engineered features saved to {output_csv}")
print(f"Final feature shape: {df.shape[0]} rows x {df.shape[1]} columns")
