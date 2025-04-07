#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

# 1. Data Loading
data_path = "/root/secure_traffic_data/flows_features.csv"
data = pd.read_csv(data_path)

# Drop columns not needed for unsupervised detection.
cols_to_drop = [
    "timestamp_start", "url", "domain", "domain_root", 
    "content_type", "user_agent", "tls_cipher", "alpn", "sni",
    "http_version", "request_headers", "response_headers"
]
data = data.drop(columns=[col for col in cols_to_drop if col in data.columns])

# 2. Data Conversion & Scaling
for col in data.columns:
    if data[col].dtype == bool:
        data[col] = data[col].astype(int)
    else:
        try:
            data[col] = pd.to_numeric(data[col])
        except Exception as e:
            data.drop(columns=[col], inplace=True)
data = data.fillna(0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)
X = pd.DataFrame(X_scaled, columns=data.columns)

# 3. Load Labels for Evaluation (heuristic labels)
labels_path = "/root/secure_traffic_data/flows_labeled.csv"
labels_df = pd.read_csv(labels_path)
y_true = labels_df["label"]

# 4. Isolation Forest (baseline)
iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
iso_forest.fit(X)
scores_if = -iso_forest.decision_function(X)
scores_if_norm = (scores_if - scores_if.min()) / (scores_if.max() - scores_if.min())
auc_if = roc_auc_score(y_true, scores_if_norm)
print(f"Isolation Forest ROC AUC Score: {auc_if:.3f}")

# ROC Plot for Isolation Forest
fpr, tpr, thresholds = roc_curve(y_true, scores_if_norm)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"IF ROC (AUC = {auc_if:.2f})")
plt.plot([0,1],[0,1],'k--', label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Isolation Forest ROC Curve")
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("/root/Mobile-Intrusion-Detection-System/images/unsupervised_roc_curve_if.png")
plt.show()

# 5. One-Class SVM
ocsvm = OneClassSVM(nu=0.05, kernel="rbf", gamma="scale")
ocsvm.fit(X)
scores_ocsvm = -ocsvm.decision_function(X)
scores_ocsvm_norm = (scores_ocsvm - scores_ocsvm.min()) / (scores_ocsvm.max() - scores_ocsvm.min())
auc_ocsvm = roc_auc_score(y_true, scores_ocsvm_norm)
print(f"One-Class SVM ROC AUC Score: {auc_ocsvm:.3f}")

# ROC Plot for One-Class SVM
fpr, tpr, thresholds = roc_curve(y_true, scores_ocsvm_norm)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"OCSV ROC (AUC = {auc_ocsvm:.2f})")
plt.plot([0,1],[0,1],'k--', label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("One-Class SVM ROC Curve")
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("/root/Mobile-Intrusion-Detection-System/images/unsupervised_roc_curve_ocsvm.png")
plt.show()

# 6. Local Outlier Factor (LOF)
# Note: LOF does not have a separate fit and predict_proba, but we use negative_outlier_factor_.
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=False)
lof.fit(X)
scores_lof = -lof.negative_outlier_factor_
scores_lof_norm = (scores_lof - scores_lof.min()) / (scores_lof.max() - scores_lof.min())
auc_lof = roc_auc_score(y_true, scores_lof_norm)
print(f"Local Outlier Factor ROC AUC Score: {auc_lof:.3f}")

# ROC Plot for LOF
fpr, tpr, thresholds = roc_curve(y_true, scores_lof_norm)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"LOF ROC (AUC = {auc_lof:.2f})")
plt.plot([0,1],[0,1],'k--', label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Local Outlier Factor ROC Curve")
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("/root/Mobile-Intrusion-Detection-System/images/unsupervised_roc_curve_lof.png")
plt.show()

# 7. Optional: Compare Anomaly Score Distributions for Isolation Forest
plt.figure(figsize=(6,5))
plt.hist(scores_if_norm[y_true==0], bins=30, alpha=0.5, label="Benign (IF)")
plt.hist(scores_if_norm[y_true==1], bins=30, alpha=0.5, label="Suspicious (IF)")
plt.xlabel("Normalized Anomaly Score")
plt.ylabel("Frequency")
plt.title("Isolation Forest Anomaly Score Distribution")
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("/root/Mobile-Intrusion-Detection-System/images/anomaly_score_distribution_if.png")
plt.show()
