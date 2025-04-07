#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler

# 1. Data Loading
data_path = "/root/secure_traffic_data/flows_labeled.csv"
data = pd.read_csv(data_path)

# 2. Define Features and Target (drop leakage-prone columns)
cols_to_drop = [
    "label", "url", "domain", "domain_root", "reasons", "timestamp_start",
    "suspicion_level", "content_type", "user_agent", "tls_established",
    "tls_cipher", "alpn", "sni", "http_version", "request_headers",
    "response_headers", "client_ip", "client_port", "server_ip",
    "server_port", "score"
]
y = data["label"]
X = data.drop(columns=[col for col in cols_to_drop if col in data.columns])

# 3. Data Conversion & Scaling
for col in X.columns:
    if X[col].dtype == bool:
        X[col] = X[col].astype(int)
non_numeric_cols = []
for col in X.columns:
    try:
        X[col] = pd.to_numeric(X[col])
    except Exception as e:
        non_numeric_cols.append(col)
if non_numeric_cols:
    X.drop(columns=non_numeric_cols, inplace=True)
X = X.fillna(0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)

print("Final feature columns used for modeling:")
print(X.columns.tolist())

# 4. Train-Test Split and CV Setup
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 5. Hyperparameter Tuning with GridSearchCV
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'class_weight': ['balanced']
}
clf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(clf, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_clf = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

# 6. Predictions and Threshold Tuning
y_proba = best_clf.predict_proba(X_test)[:, 1]

# Set your desired decision threshold (e.g., 0.4 to favor recall)
decision_threshold = 0.4
y_pred = (y_proba >= decision_threshold).astype(int)

print("\n--- Classification Report (Threshold = {:.2f}) ---".format(decision_threshold))
print(classification_report(y_test, y_pred))
roc_auc = roc_auc_score(y_test, y_proba)
print(f"ROC AUC Score: {roc_auc:.3f}")

# 7. Visualization: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion Matrix (Threshold = {:.2f})".format(decision_threshold))
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ["Benign", "Suspicious"])
plt.yticks(tick_marks, ["Benign", "Suspicious"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")
plt.tight_layout()
plt.savefig("/root/Mobile-Intrusion-Detection-System/images/confusion_matrix_threshold.png")
plt.show()

# 8. Visualization: ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], "k--", label="Random Chance")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (Threshold = {:.2f})".format(decision_threshold))
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("/root/Mobile-Intrusion-Detection-System/images/roc_curve_threshold.png")
plt.show()

# 9. Visualization: Feature Importances
importances = best_clf.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns
plt.figure(figsize=(8, 6))
plt.title("Feature Importances")
plt.bar(range(len(features)), importances[indices])
plt.xticks(range(len(features)), features[indices], rotation=90)
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig("/root/Mobile-Intrusion-Detection-System/images/feature_importances_threshold.png")
plt.show()
