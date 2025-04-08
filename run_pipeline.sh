#!/bin/bash
# run_pipeline.sh
# This script runs the entire pipeline to process a new mobile data capture.
# It executes:
#   1. process_flows.py   (Extracts flows and appends data to the SQLite database)
#   2. feature_engineering.py   (Computes engineered features from the CSV)
#   3. label_and_analyze.py     (Labels flows and updates the labeled table in the DB)
#   4. model_supervised.py      (Incrementally trains the supervised model)
#   5. model_unsupervised.py    (Retrains unsupervised models on full cumulative data)

echo "-------------------------------------------------"
echo "Starting pipeline for new mobile data capture..."
echo "-------------------------------------------------"

echo "Step 1: Running process_flows.py..."
python3 process_flows.py
if [ $? -ne 0 ]; then
    echo "Error encountered in process_flows.py. Exiting."
    exit 1
fi

echo "Step 2: Running feature_engineering.py..."
python3 feature_engineering.py
if [ $? -ne 0 ]; then
    echo "Error encountered in feature_engineering.py. Exiting."
    exit 1
fi

echo "Step 3: Running label_and_analyze.py..."
python3 label_and_analyze.py
if [ $? -ne 0 ]; then
    echo "Error encountered in label_and_analyze.py. Exiting."
    exit 1
fi

echo "Step 4: Running model_supervised.py..."
python3 model_supervised.py
if [ $? -ne 0 ]; then
    echo "Error encountered in model_supervised.py. Exiting."
    exit 1
fi

echo "Step 5: Running model_unsupervised.py..."
python3 model_unsupervised.py
if [ $? -ne 0 ]; then
    echo "Error encountered in model_unsupervised.py. Exiting."
    exit 1
fi

echo "-------------------------------------------------"
echo "Pipeline execution complete!"
echo "-------------------------------------------------"
