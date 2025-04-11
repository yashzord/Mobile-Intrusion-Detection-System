#!/usr/bin/env python3
import os
import json
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Paths
BASE_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(BASE_DIR, "results")
RUN_LOG_PATH = os.path.join(RESULTS_DIR, "run_log.json")
FEATURES_CSV = os.path.join(BASE_DIR, "secure_traffic_data", "flows_features.csv")

# Set Streamlit page config
st.set_page_config(page_title="Mobile Intrusion Detection Dashboard", layout="wide")

# ----- Helper Functions -----
@st.cache_data
def load_run_log():
    if os.path.exists(RUN_LOG_PATH):
        with open(RUN_LOG_PATH, "r") as f:
            return json.load(f)
    else:
        return []

@st.cache_data
def load_results():
    results_csv = os.path.join(RESULTS_DIR, "flows_with_anomalies.csv")
    suspicious_csv = os.path.join(RESULTS_DIR, "suspicious_flows.csv")
    df_results = pd.read_csv(results_csv)
    df_suspicious = pd.read_csv(suspicious_csv)
    return df_results, df_suspicious

def plot_loss_curve():
    # Dummy example data. Replace with your actual training logs if available.
    epochs = np.arange(1, 101)
    loss = np.linspace(0.65, 0.03, 100)  # Dummy decreasing loss
    val_loss = np.linspace(0.16, 0.02, 100)  # Dummy decreasing validation loss

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=loss, mode="lines", name="Training Loss"))
    fig.add_trace(go.Scatter(x=epochs, y=val_loss, mode="lines", name="Validation Loss"))
    fig.update_layout(title="Training and Validation Loss Curves",
                      xaxis_title="Epoch",
                      yaxis_title="Loss",
                      height=400)
    st.plotly_chart(fig, use_container_width=True)

def plot_reconstruction_error_histogram(df_results, threshold):
    fig = px.histogram(df_results, x="reconstruction_error", nbins=50,
                       title="Distribution of Reconstruction Errors")
    fig.add_vline(x=threshold, line_dash="dash", line_color="red",
                  annotation_text=f"Threshold = {threshold:.4f}")
    st.plotly_chart(fig, use_container_width=True)

def plot_anomaly_counts(run_log):
    # Aggregate counts per run
    run_dates = [datetime.datetime.fromisoformat(r["run_datetime"]) for r in run_log]
    pre_counts = [r["pre_whitelist"]["n_anomalies"] for r in run_log]
    post_counts = [r["post_whitelist"]["n_anomalies"] for r in run_log]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=run_dates, y=pre_counts, mode="lines+markers", name="Pre-Whitelist Anomalies"))
    fig.add_trace(go.Scatter(x=run_dates, y=post_counts, mode="lines+markers", name="Post-Whitelist Anomalies"))
    fig.update_layout(title="Anomaly Counts Over Runs",
                      xaxis_title="Run Date",
                      yaxis_title="Number of Anomalies",
                      height=400)
    st.plotly_chart(fig, use_container_width=True)

def plot_unique_urls(run_log):
    pre_unique = [len(r["pre_whitelist"]["urls"]) for r in run_log]
    post_unique = [len(r["post_whitelist"]["urls"]) for r in run_log]
    run_dates = [datetime.datetime.fromisoformat(r["run_datetime"]) for r in run_log]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=run_dates, y=pre_unique, name="Unique URLs (Pre)"))
    fig.add_trace(go.Bar(x=run_dates, y=post_unique, name="Unique URLs (Post)"))
    fig.update_layout(title="Unique URL Counts Over Runs",
                      xaxis_title="Run Date",
                      yaxis_title="Unique URL Count",
                      barmode="group",
                      height=400)
    st.plotly_chart(fig, use_container_width=True)

def plot_run_summary(run_log):
    df_log = pd.DataFrame(run_log)
    st.dataframe(df_log)

def plot_box_error_by_feature(df_results, feature):
    if feature in df_results.columns:
        fig = px.box(df_results, x=feature, y="reconstruction_error",
                     title=f"Reconstruction Error by {feature}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write(f"Feature '{feature}' not found in results.")

# ----- Dashboard Layout -----
st.title("Mobile Intrusion Detection Dashboard")
st.markdown("This dashboard displays the results and metrics of the mobile intrusion detection system.")

# Load data
run_log = load_run_log()
df_results, df_suspicious = load_results()

# Display some run summary info
if run_log:
    latest_run = run_log[-1]
    st.markdown(f"**Latest Run:** {latest_run['run_datetime']}")
    st.markdown(f"- **Total Flows:** {latest_run['n_total_flows']}")
    st.markdown(f"- **Anomaly Threshold:** {latest_run['anomaly_threshold']:.4f}")
    st.markdown(f"- **Anomalies before Whitelist:** {latest_run['pre_whitelist']['n_anomalies']}")
    st.markdown(f"- **Anomalies after Whitelist:** {latest_run['post_whitelist']['n_anomalies']}")
    # Only display test_loss if it exists in the run summary
    if "test_loss" in latest_run:
        st.markdown(f"- **Test Loss:** {latest_run['test_loss']:.4f}")
    else:
        st.markdown(f"- **Test Loss:** N/A")
else:
    st.write("No run log data available.")

st.header("1. Loss Curves")
plot_loss_curve()

st.header("2. Reconstruction Error Distribution")
threshold = latest_run["anomaly_threshold"] if run_log else 0.07
plot_reconstruction_error_histogram(df_results, threshold)

st.header("3. Anomaly Counts Across Runs")
plot_anomaly_counts(run_log)

st.header("4. Unique URL Count Before vs. After Whitelist")
plot_unique_urls(run_log)

st.header("5. Run Summary")
plot_run_summary(run_log)

# Additional plots – customize as needed:
st.header("6. Distribution of Threat Scores")
if "threat_score" in df_results.columns:
    fig = px.histogram(df_results, x="threat_score", nbins=50,
                       title="Threat Score Distribution")
    st.plotly_chart(fig, use_container_width=True)

st.header("7. Reconstruction Error vs. Duration")
if "duration" in df_results.columns:
    fig = px.scatter(df_results, x="duration", y="reconstruction_error",
                     title="Reconstruction Error vs. Flow Duration",
                     trendline="ols")
    st.plotly_chart(fig, use_container_width=True)

st.header("8. Box Plot of Reconstruction Error by Day of Week")
if "day_of_week" in df_results.columns:
    fig = px.box(df_results, x="day_of_week", y="reconstruction_error",
                 title="Reconstruction Error Distribution by Day of Week")
    st.plotly_chart(fig, use_container_width=True)

st.header("9. Histogram of Log-scaled Request Sizes")
if "log_request_size" in df_results.columns:
    fig = px.histogram(df_results, x="log_request_size", nbins=50,
                       title="Distribution of Log-scaled Request Sizes")
    st.plotly_chart(fig, use_container_width=True)

st.header("10. Interactive Table of Suspicious Flows")
st.dataframe(df_suspicious)

# Sidebar filter for additional box plot by feature
st.sidebar.header("Dashboard Filters")
selected_feature = st.sidebar.selectbox("Select feature for box plot", df_results.columns.tolist())
plot_box_error_by_feature(df_results, selected_feature)
