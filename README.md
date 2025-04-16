# Mobile Intrusion Detection System

This project implements a Mobile Intrusion Detection System that processes network traffic captured from mobile devices, performs feature engineering, and uses an autoencoder-based deep learning model to detect anomalous network flows. External threat intelligence and whitelist data are utilized to reduce false positives and enhance detection accuracy.

## Table of Contents

-   [Features](#features)
-   [Architecture Overview](#architecture-overview)
-   [Installation and Environment Setup](#installation-and-environment-setup)
-   [Usage](#usage)
    -   [Train Mode](#train-mode)
    -   [Predict Mode](#predict-mode)
    -   [Dashboard Mode](#dashboard-mode)
-   [Data Sources and External References](#data-sources-and-external-references)
-   [Conclusion and Future Work](#conclusion-and-future-work)
-   [References](#references)
-   [License](#license)

## Features

-   **Network Flow Capture:** Captures HTTP flows from mobile devices using MITMProxy.
-   **Flow Processing and Archiving:** Archives previous flow captures (renaming flows.mitm to flows\_1.mitm) to prevent overwrites before new captures.
-   **Feature Engineering:** Extracts, scales, and vectorizes features (via TF-IDF) from the captured flows.
-   **Anomaly Detection:** Trains an autoencoder model on historical data and flags anomalous flows based on reconstruction errors.
-   **Whitelist Filtering:** Uses whitelists from sources such as the Majestic Million and Top 1M domains (Kaggle) to filter out benign anomalies.
-   **Prediction Mode:** Processes new captured data, applies the trained model, and outputs anomalies.
-   **Dashboard Mode:** Provides an interactive dashboard (using Streamlit and Plotly) for visualization and analysis of results.

## Architecture Overview

1.  **Flow Capture and Archiving**
    -   MITMProxy captures network flows.
    -   The existing capture file (flows.mitm) is renamed to flows\_1.mitm before new captures to prevent file overwriting.
    -   Captured flows are processed and converted into CSV and JSON formats.
2.  **Feature Engineering**
    -   Features are extracted from the processed flows.
    -   The data is scaled (using a scaler) and vectorized (using TF-IDF) for model input.
    -   Engineered features are saved into a CSV file for further use.
3.  **Model Training and Prediction**
    -   **Train Mode:** The autoencoder model is trained on engineered features to learn the normal behavior of network flows. Artifacts such as the trained model and scaler are saved.
    -   **Predict Mode:** New capture data is processed similarly; the trained model predicts anomalies based on reconstruction errors. Whitelist filtering is then applied to remove false positives.
4.  **Visualization and Dashboard (Optional)**
    -   An interactive dashboard displays loss curves, reconstruction error distributions, anomaly counts, and other key performance indicators.

## Installation and Environment Setup

1.  bashCopygit clone https://github.com/yourusername/Mobile-Intrusion-Detection-System.gitcd Mobile-Intrusion-Detection-System
2.  bashCopypython3 -m venv venvsource venv/bin/activate
3.  bashCopypip install tensorflow pandas numpy scikit-learn matplotlib plotly streamlit

## Usage

### Train Mode

Run the training pipeline to process historical data, engineer features, and train the autoencoder:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`bashCopy./run_project.sh train`

This command performs the following steps:

-   Processes the captured network flows.
-   Engineers and scales features.
-   Trains the autoencoder model.
-   Evaluates the model and logs results in the results/run\_log.json file.

### Predict Mode

Prior to running prediction, the existing capture file is archived (renamed from flows.mitm to flows\_1.mitm) to preserve historical data. New flows are then captured and processed.

Run the prediction pipeline using:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`bashCopy./run_project.sh predict`

This command executes:

1.  **Archiving:** Renames and archives the previous flows\_1.mitm file with a timestamp.
2.  **Flow Processing:** Processes new capture flows, saving outputs as flows\_new.csv and JSON.
3.  **Feature Engineering:** Copies flows\_new.csv to flows.csv and extracts/scales features.
4.  **Prediction Execution:** Loads the trained scaler and model, computes reconstruction errors, and flags anomalies based on a defined threshold.
5.  **Whitelist Filtering:** Applies whitelist filtering (with data from external sources) to remove known benign domains.
6.  **Output Generation:** Saves prediction results and suspicious flows to the results/ directory, and logs the run summary.

### Dashboard Mode

Launch the interactive dashboard to visualize and interact with the results:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`bashCopystreamlit run dashboard.py`

The dashboard displays:

-   **Loss Curves:** Graphs of training and validation loss.
-   **Reconstruction Error Distribution:** Histogram with the anomaly threshold indicated.
-   **Anomaly Counts Over Runs:** Trends showing the number of anomalies before and after whitelist filtering.
-   **Unique URL Comparisons:** Bar charts comparing unique URLs pre- and post-filtering.
-   **Additional Visualizations:** Scatter plots, box plots, and interactive tables with various metrics.
-   **Sidebar Filters:** Options to filter and analyze specific features.

## Data Sources and External References

-   **Whitelist Data:**
    -   [Majestic Million Report](https://majestic.com/reports/majestic-million)
    -   [Top 1M Domains Dataset (Kaggle)](https://www.kaggle.com/datasets/cheedcheed/top1m)
-   **External Threat Intelligence:** _(Source URL pending; to be updated once available.)_

## Conclusion and Future Work

The Mobile Intrusion Detection System effectively processes network flows and utilizes an autoencoder to flag anomalous activity in mobile network traffic. By integrating external whitelist data, the system reduces false positives, although preliminary results in prediction mode indicate that some benign Google search queries may be inaccurately flagged as anomalies. This may suggest that the anomaly threshold or feature engineering requires further refinement.

Future work will focus on:

-   Fine-tuning model parameters and anomaly detection thresholds.
-   Enhancing feature extraction techniques for improved accuracy.
-   Integrating additional sources of external threat intelligence.
-   Expanding dashboard capabilities for more in-depth real-time analysis.

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
