# Mobile Intrusion Detection System – Real-time Dashboard Extension

This project demonstrates a complete pipeline for mobile network traffic intrusion detection that:
- Captures mobile traffic using an iOS device (via mitmproxy) and a Kali Linux workstation.
- Stores data securely in an encrypted container.
- Performs feature engineering, threat intelligence integration, and trains a deep autoencoder
  model for anomaly detection.
- Now supports subsequent “test” runs: new captured traffic is processed and predicted using the
  trained model.
- Logs performance metrics (e.g. number of anomalies, error statistics) for every run.
- Provides a Streamlit dashboard that shows tables, graphs and performance metrics from each run.

## Directory Structure

