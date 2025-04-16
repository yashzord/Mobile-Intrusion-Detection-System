# Mobile Intrusion Detection System

This project implements a Mobile Intrusion Detection System that processes network traffic captured from mobile devices, performs feature engineering, and uses an autoencoder-based deep learning model to detect anomalous network flows. External threat intelligence and whitelist data are utilized to reduce false positives and enhance detection accuracy.

## Table of Contents

- [Features](#features)
- [Architecture Overview](#architecture-overview)
- [Installation and Environment Setup](#installation-and-environment-setup)
- [Usage](#usage)
  - [Train Mode](#train-mode)
  - [Predict Mode](#predict-mode)
  - [Dashboard Mode](#dashboard-mode)
- [Data Sources and External References](#data-sources-and-external-references)
- [Conclusion and Future Work](#conclusion-and-future-work)
- [References](#references)
- [License](#license)

## Features

- **Network Flow Capture:** Uses MITMProxy to capture HTTP flows from mobile devices.
- **Flow Processing and Archiving:** Archives previous flow captures by renaming `flows.mitm` to `flows_1.mitm` to avoid overwriting, then processes new captures.
- **Feature Engineering:** Extracts, scales, and vectorizes features (using TF-IDF) from the captured flows.
- **Anomaly Detection:** Trains an autoencoder model on historical data to learn normal behavior and flag anomalous flows based on reconstruction errors.
- **Whitelist Filtering:** Applies whitelist filtering using data from sources such as the Majestic Million and Top 1M domains (Kaggle) to filter out benign anomalies.
- **Prediction Mode:** Processes new captured data, applies the trained model, and outputs anomalies along with metrics.
- **Dashboard Mode:** Provides an interactive dashboard (built with Streamlit and Plotly) for visualization and analysis of results.

## Architecture Overview

1. **Flow Capture and Archiving**
   - MITMProxy captures network flows from mobile devices.
   - The existing capture file (`flows.mitm`) is renamed to `flows_1.mitm` before new capture to avoid overwriting.
   - Captured flows are then converted into CSV and JSON formats.

2. **Feature Engineering**
   - Features are extracted from the processed flows.
   - Data is scaled using a scaler and vectorized using TF-IDF.
   - Engineered features are saved to a CSV file and used for model training and prediction.

3. **Model Training and Prediction**
   - **Train Mode:** Processes historical data to train the autoencoder model. Model artifacts, such as the trained model and scaler, are saved.
   - **Predict Mode:** Processes new captured data, runs predictions with the trained model, and applies whitelist filtering to reduce false positives.

4. **Visualization and Dashboard (Optional)**
   - An interactive dashboard displays metrics such as loss curves, reconstruction error distributions, anomaly counts, and more.

## Installation and Environment Setup

1. **Clone the Repository**

    ```bash
    git clone https://github.com/yourusername/Mobile-Intrusion-Detection-System.git
    cd Mobile-Intrusion-Detection-System
    ```

2. Create and Activate a Virtual Environment
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install the Required Packages
    ```bash
    pip install tensorflow pandas numpy scikit-learn matplotlib plotly streamlit
    ```

## Usage

### Train Mode

Run the training pipeline to process historical data, perform feature engineering, and train the autoencoder model:

```bash
./run_project.sh train
```

This command performs the following steps:
   - Processes the captured network flows.
   - Archives the previous capture file (i.e., flows.mitm is renamed to flows_1.mitm).
   - Engineers and scales the features.
   - Trains the autoencoder model and evaluates it.
   - Saves the model, scaler, and run summary in the results/ directory (including run_log.json).

### Predict Mode

Before running prediction, the existing flows_1.mitm file is archived (with a timestamp) to preserve previous capture data. Then, new flows are captured and processed:

```bash
./run_project.sh predict
```

This command performs the following steps:
   - Archives the current flows_1.mitm file with a timestamp.
   - Processes new captured flows and converts them to CSV and JSON formats.
   - Copies the new CSV file to the expected location and performs feature engineering.
   - Loads the trained scaler and model.
   - Runs predictions on the new data and applies whitelist filtering.
   - Outputs the prediction results and saves them in the results/ directory along with an updated run_log.json.

### Dashboard Mode

To visualize the results and performance metrics, run the dashboard mode using Streamlit:

```bash
./run_project.sh dashboard
```

The dashboard displays:
   - Training and validation loss curves.
   - Distribution of reconstruction errors with the set threshold.
   - Anomaly counts over multiple runs.
   - Comparison of unique URL counts before and after whitelist filtering.
   - Additional interactive plots and summary tables for further analysis.

## Data Sources and External References

-   **Whitelist Data:**
    -   [Majestic Million Report](https://majestic.com/reports/majestic-million)
    -   [Top 1M Domains Dataset (Kaggle)](https://www.kaggle.com/datasets/cheedcheed/top1m)

## Conclusion and Future Work

The Mobile Intrusion Detection System demonstrates effective processing of network flow data from mobile devices and leverages an autoencoder-based approach for anomaly detection. The integration of whitelist filtering significantly reduces false positives; however, the prediction stage revealed challenges—particularly in distinguishing benign Google search-related flows from truly malicious activities. This indicates that further refinement of the anomaly detection logic, especially within the whitelist filtering phase, is required. Future work will focus on improving the filtering algorithms, incorporating additional threat intelligence sources, and expanding the dashboard functionalities for deeper insights.

## References

-   [Majestic Million Report](https://majestic.com/reports/majestic-million)
-   [Top 1M Domains Dataset (Kaggle)](https://www.kaggle.com/datasets/cheedcheed/top1m)
-   [MITMProxy](https://mitmproxy.org/)
-   [TensorFlow](https://www.tensorflow.org/)
-   [Scikit-Learn](https://scikit-learn.org/)
-   [Plotly](https://plotly.com/)
-   [Streamlit](https://streamlit.io/)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.