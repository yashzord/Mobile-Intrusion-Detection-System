#!/usr/bin/env python3
"""
process_external_threat_data.py

Processes the Kaggle malicious URLs dataset (malicious_phish.csv)
by extracting domains from URLs and assigning a threat score (1 for malicious).
Output is saved as external_threats.csv in the external_data folder.
"""

import os
import pandas as pd
from urllib.parse import urlparse

base_dir = os.path.dirname(__file__)
input_csv = os.path.join(base_dir, "external_data", "malicious_phish.csv")

if not os.path.exists(input_csv):
    print(f"Input file {input_csv} does not exist. Please verify the downloaded dataset.")
    exit(1)

df = pd.read_csv(input_csv)

if "url" not in df.columns and "URL" not in df.columns:
    print("No 'url' or 'URL' column found in the dataset. Please check the CSV format.")
    exit(1)
    
url_col = "url" if "url" in df.columns else "URL"

def extract_domain(url):
    try:
        parsed = urlparse(url)
        return parsed.netloc
    except Exception:
        return ""

df["domain"] = df[url_col].apply(extract_domain)
df["threat_score"] = 1
df_external = df[["domain", "threat_score"]].drop_duplicates()

output_csv = os.path.join(base_dir, "external_data", "external_threats.csv")
df_external.to_csv(output_csv, index=False)
print(f"✅ External threat intelligence data saved to {output_csv}")
