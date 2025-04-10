#!/usr/bin/env python3
"""
metrics_logger.py

Logs run metrics (timestamp, run duration, number of flows, anomalies, average reconstruction error,
and anomaly threshold) into results/metrics_log.csv.
"""

import os
import csv
from datetime import datetime

METRICS_FILE = os.path.join(os.path.dirname(__file__), "results", "metrics_log.csv")
os.makedirs(os.path.dirname(METRICS_FILE), exist_ok=True)

if not os.path.exists(METRICS_FILE):
    with open(METRICS_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "run_timestamp", "run_duration_sec", "num_flows", "num_anomalies",
            "avg_reconstruction_error", "anomaly_threshold", "model_version"
        ])

def log_metrics(run_duration, num_flows, num_anomalies, avg_error, threshold, model_version="v1"):
    run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(METRICS_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            run_timestamp, run_duration, num_flows, num_anomalies, avg_error, threshold, model_version
        ])
    print(f"Metrics logged for run at {run_timestamp}")
