#!/usr/bin/env python3
import pandas as pd
import numpy as np
from urllib.parse import urlparse

# === Load the raw flows ===
df = pd.read_csv("/home/kali/Mobile-Intrusion-Detection-System/secure_traffic_data/flows.csv")

# === Time features ===
df['timestamp_start'] = pd.to_datetime(df['timestamp_start'], unit='s', errors='coerce')
df['hour'] = df['timestamp_start'].dt.hour
df['day_of_week'] = df['timestamp_start'].dt.dayofweek

# === URL features ===
def parse_url(url):
    try:
        parsed = urlparse(str(url))
        return pd.Series({
            "domain": parsed.netloc,
            "path_depth": len([p for p in parsed.path.strip("/").split("/") if p]),
            "query_length": len(parsed.query),
            "is_https": parsed.scheme.lower() == "https"
        })
    except:
        return pd.Series({
            "domain": "",
            "path_depth": 0,
            "query_length": 0,
            "is_https": False
        })

df = pd.concat([df, df['url'].apply(parse_url)], axis=1)

# === HTTP Method One-Hot ===
df = pd.get_dummies(df, columns=["method"], prefix="method")

# === Status Category ===
df["status_category"] = df["status_code"].fillna(0).apply(lambda x: int(x / 100))

# === Log-scaled Sizes ===
df["log_request_size"] = np.log1p(df["request_content_length"].astype(float))
df["log_response_size"] = np.log1p(df["response_content_length"].astype(float))

# === TLS indicators ===
df["tls_failed"] = df["tls_established"] == False
df["tls_missing"] = df["tls_established"].isna()

# === Nonstandard Port Flag ===
df["nonstandard_port"] = ~df["server_port"].isin([80, 443])

# === Drop unused or redundant columns ===
drop_cols = [
    "request_headers", "response_headers",
    "timestamp_end", "http_version",
    "request_content_length", "response_content_length",
    "client_ip", "client_port", "server_ip", "server_port",
    "tls_cipher", "alpn", "sni", "user_agent", "content_encoding"
]

df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

# === Save final features ===
output_path = "/home/kali/Mobile-Intrusion-Detection-System/secure_traffic_data/flows_features.csv"
df.to_csv(output_path, index=False)
print(f"✅ Feature engineering complete. Output saved to {output_path}")
print(f"🔢 Shape: {df.shape[0]} rows × {df.shape[1]} columns")
