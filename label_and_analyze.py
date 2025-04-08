#!/usr/bin/env python3
import pandas as pd
import numpy as np

# Extended allow list with many mainstream domains
ALLOWLIST = [
    "google.com", "youtube.com", "facebook.com", "twitter.com", "instagram.com", "linkedin.com", 
    "amazon.com", "apple.com", "microsoft.com", "ebay.com", "netflix.com", "wikipedia.org", 
    "reddit.com", "bbc.co.uk", "bbc.com", "cnn.com", "nytimes.com", "reuters.com", "bloomberg.com", 
    "forbes.com", "washingtonpost.com", "theguardian.com", "time.com", "usatoday.com", "espn.com", 
    "imdb.com", "tripadvisor.com", "yahoo.com", "live.com", "msn.com", "pinterest.com", "tumblr.com", 
    "flickr.com", "vimeo.com", "soundcloud.com", "spotify.com", "paypal.com", "dropbox.com", 
    "wordpress.com", "blogger.com", "medium.com", "huffpost.com", "vice.com", "buzzfeed.com", 
    "cnet.com", "techcrunch.com", "engadget.com", "mashable.com", "dailymail.co.uk", "washingtonexaminer.com",
    "usatoday.com", "abcnews.go.com", "nbcnews.com", "msnbc.com", "foxnews.com", "drudgereport.com",
    "cbsnews.com", "npr.org", "economist.com", "financialtimes.com", "fortune.com", "usatoday.com",
    "bbc.co.uk", "bbc.com", "guardian.co.uk", "independent.co.uk", "telegraph.co.uk", 
    "aljazeera.com", "dw.com", "rt.com", "theverge.com", "wired.com", "arstechnica.com"
]

INPUT_PATH = "/home/kali/Mobile-Intrusion-Detection-System/secure_traffic_data/flows_features.csv"
OUTPUT_LABELED = "/home/kali/Mobile-Intrusion-Detection-System/secure_traffic_data/flows_labeled.csv"
OUTPUT_SUSPICIOUS = "/home/kali/Mobile-Intrusion-Detection-System/secure_traffic_data/suspicious_flows.csv"

print("📥 Loading dataset...")
df = pd.read_csv(INPUT_PATH)

# Compute domain frequency to use as a proxy for rarity.
domain_freq = df['domain'].value_counts().to_dict()
max_freq = max(domain_freq.values()) if domain_freq else 1

# Initialize scoring columns.
df["score"] = 0
df["label"] = 0
df["suspicion_level"] = "Low"
df["reasons"] = ""

def add_reason(idx, reason, score):
    df.at[idx, "reasons"] += f"{reason}; "
    df.at[idx, "score"] += score

# Scoring rules applied row by row.
for i, row in df.iterrows():
    domain = row.get("domain", "")
    
    # Skip scoring if the domain is in the allowlist (or a subdomain of an allowed domain)
    if any(allowed in domain for allowed in ALLOWLIST):
        continue

    # Rule: Not HTTPS.
    if not row.get("is_https", False):
        add_reason(i, "Not HTTPS", 2)
    # Rule: TLS failure.
    if row.get("tls_failed", False):
        add_reason(i, "TLS failed/missing", 2)
    # Rule: HTTP error status.
    if row.get("status_code", 200) >= 400:
        add_reason(i, "HTTP error status", 2)
    # Rule: Suspiciously large response size.
    if row.get("log_response_size", 0) > 13:
        add_reason(i, "Suspiciously large response", 2)
    # Rule: Long query string.
    if row.get("query_length", 0) > 150:
        add_reason(i, "Long query string", 2)
    # Rule: Domain rarity – if frequency is low (less than 20% of the maximum frequency), add a point.
    freq = domain_freq.get(domain, 0)
    if freq < 0.2 * max_freq:
        add_reason(i, "Rare domain", 1)

# Label as suspicious if score >= 2.
df["label"] = (df["score"] >= 2).astype(int)

# Assign suspicion levels based on score.
df["suspicion_level"] = pd.cut(
    df["score"],
    bins=[-1, 1, 3, np.inf],
    labels=["Low", "Medium", "High"]
)

# Save outputs.
df.to_csv(OUTPUT_LABELED, index=False)
suspicious = df[df["label"] == 1].sort_values(by="score", ascending=False).head(20)
suspicious.to_csv(OUTPUT_SUSPICIOUS, index=False)

print(f"✅ Labeled dataset saved to: {OUTPUT_LABELED}")
print("📊 Label distribution:")
print(df["label"].value_counts())
print(f"🧪 Top 20 suspicious flows saved to: {OUTPUT_SUSPICIOUS}")
