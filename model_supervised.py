#!/usr/bin/env python3
import os
import sqlite3
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
from datetime import datetime

# Define paths
DB_PATH = "/home/kali/Mobile-Intrusion-Detection-System/traffic_data.db"
CHECKPOINT_PATH = "/home/kali/Mobile-Intrusion-Detection-System/model_checkpoint.pkl"
LAST_TRAINED_PATH = "/home/kali/Mobile-Intrusion-Detection-System/last_trained.txt"
METRICS_FILE = "/home/kali/Mobile-Intrusion-Detection-System/model_supervised_metrics.txt"
IMAGES_DIR = "/home/kali/Mobile-Intrusion-Detection-System/images"
RUN_COUNTER_FILE = "/home/kali/Mobile-Intrusion-Detection-System/run_supervised_counter.txt"

# Function to load labeled data from the database, optionally filtering by last trained timestamp.
def load_labeled_data(min_timestamp=None):
    conn = sqlite3.connect(DB_PATH)
    if min_timestamp is None:
        query = "SELECT * FROM labeled"
    else:
        query = f"SELECT * FROM labeled WHERE timestamp_start > '{min_timestamp}'"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

# Preprocessing function (simplified)
def preprocess_data(df):
    # Columns that should not be used for training.
    drop_cols = [
        "label", "url", "domain", "domain_root", "reasons", "timestamp_start",
        "suspicion_level", "content_type", "user_agent", "tls_established",
        "tls_cipher", "alpn", "sni", "http_version", "request_headers",
        "response_headers", "client_ip", "client_port", "server_ip",
        "server_port", "score"
    ]
    if "label" not in df.columns:
        raise Exception("Dataframe does not contain label column")
    y = df["label"]
    X = df.drop(columns=[col for col in drop_cols if col in df.columns])
    
    # Convert boolean columns to integers.
    for col in X.columns:
        if X[col].dtype == bool:
            X[col] = X[col].astype(int)
    non_numeric_cols = []
    for col in X.columns:
        try:
            X[col] = pd.to_numeric(X[col])
        except Exception:
            non_numeric_cols.append(col)
    if non_numeric_cols:
        X.drop(columns=non_numeric_cols, inplace=True)
    X = X.fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)
    return X, y, scaler

# Retrieve last trained timestamp if it exists.
last_trained_timestamp = None
if os.path.exists(LAST_TRAINED_PATH):
    with open(LAST_TRAINED_PATH, "r") as f:
        try:
            last_trained_timestamp = f.read().strip()  # Expect string datetime
        except Exception:
            last_trained_timestamp = None

# Load new labeled data from the database.
df_labeled = load_labeled_data(min_timestamp=last_trained_timestamp)

if df_labeled.empty:
    print("No new labeled data to train on.")
    exit(0)
else:
    print(f"Loaded {df_labeled.shape[0]} new labeled records for training.")

# Preprocess new data.
X_new, y_new, scaler = preprocess_data(df_labeled)

# Initialize or load the incremental model.
if os.path.exists(CHECKPOINT_PATH):
    model = joblib.load(CHECKPOINT_PATH)
    print("Loaded model checkpoint.")
else:
    model = SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-3)
    # Perform an initial partial_fit with the defined classes.
    model.partial_fit(X_new, y_new, classes=np.array([0, 1]))
    print("Initialized new model and performed initial training.")

# Update the model incrementally with new data.
model.partial_fit(X_new, y_new)
print("Updated model with new data.")

# Evaluate on a holdout split from the new data.
X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, test_size=0.3, random_state=42)
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred)
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
print("Classification Report on new data:")
print(report)
print(f"ROC AUC Score on new data: {roc_auc:.3f}")

# -- RUN COUNTER --
# Check for existing run counter and increment it.
if os.path.exists(RUN_COUNTER_FILE):
    with open(RUN_COUNTER_FILE, "r") as f:
        try:
            run_number = int(f.read().strip()) + 1
        except:
            run_number = 1
else:
    run_number = 1
with open(RUN_COUNTER_FILE, "w") as f:
    f.write(str(run_number))
run_label = f"Run{run_number}"

# Create and save confusion matrix plot.
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title(f"Confusion Matrix - {run_label}")
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ["Benign", "Suspicious"])
plt.yticks(tick_marks, ["Benign", "Suspicious"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if cm[i, j] > cm.max()/2. else "black")
conf_mat_filename = f"confusion_matrix_{run_label}.png"
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, conf_mat_filename))
plt.close()

# Create and save ROC curve plot.
fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0,1], [0,1], "k--", label="Random Chance")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve - {run_label}")
plt.legend(loc="best")
roc_curve_filename = f"roc_curve_{run_label}.png"
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, roc_curve_filename))
plt.close()

# Append performance metrics to the supervised metrics log file.
metrics_text = f"Run: {run_label}\n"
metrics_text += f"Loaded new labeled records: {df_labeled.shape[0]}\n"
metrics_text += f"ROC AUC Score: {roc_auc:.3f}\n"
metrics_text += "Classification Report:\n" + report + "\n"
metrics_text += f"Confusion Matrix Image: {conf_mat_filename}\n"
metrics_text += f"ROC Curve Image: {roc_curve_filename}\n\n"

with open(METRICS_FILE, "a") as f:
    f.write(metrics_text)

print("Performance metrics appended to", METRICS_FILE)

# Save the updated model checkpoint.
joblib.dump(model, CHECKPOINT_PATH)
print(f"Model checkpoint saved to {CHECKPOINT_PATH}.")

# Update the last trained timestamp using the maximum timestamp_start in the new labeled data.
new_last_timestamp = df_labeled["timestamp_start"].max()
with open(LAST_TRAINED_PATH, "w") as f:
    f.write(str(new_last_timestamp))
print(f"Updated last training timestamp to {new_last_timestamp}.")
