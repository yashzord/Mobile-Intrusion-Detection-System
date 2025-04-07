#!/usr/bin/env python3
import pandas as pd

# === Load both datasets ===
raw_path = "/root/secure_traffic_data/flows.csv"
feat_path = "/root/secure_traffic_data/flows_features.csv"

df_raw = pd.read_csv(raw_path)
df_feat = pd.read_csv(feat_path)

# === Print shape comparison ===
print("📦 Raw Dataset (flows.csv)")
print(f"Rows: {df_raw.shape[0]}, Columns: {df_raw.shape[1]}\n")

print("🧪 Feature Dataset (flows_features.csv)")
print(f"Rows: {df_feat.shape[0]}, Columns: {df_feat.shape[1]}\n")

# === Column comparison ===
raw_cols = set(df_raw.columns)
feat_cols = set(df_feat.columns)

added_cols = feat_cols - raw_cols
removed_cols = raw_cols - feat_cols
common_cols = raw_cols & feat_cols

print("➕ Columns Added (in feature set only):")
print(sorted(list(added_cols)), "\n")

print("➖ Columns Removed (from raw data):")
print(sorted(list(removed_cols)), "\n")

print("✅ Columns Retained (shared in both):")
print(sorted(list(common_cols)), "\n")

# === Check row integrity ===
if df_raw.shape[0] == df_feat.shape[0]:
    print("✅ Row count is consistent. No data was dropped.\n")
else:
    print("⚠️ Row mismatch! Check for dropped or missing rows.\n")

# === Show 10 random rows from each for comparison ===
print("🔍 Sample from flows.csv (Raw Data):")
print(df_raw.sample(10).to_string(index=False), "\n")

print("🔍 Sample from flows_features.csv (Engineered Data):")
print(df_feat.sample(10).to_string(index=False))
