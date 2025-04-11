#!/bin/bash
# run_project.sh - Master script for the Mobile Intrusion Detection Pipeline.
# Usage:
#   ./run_project.sh train    -> For initial training (using existing flows.mitm)
#   ./run_project.sh predict  -> For subsequent runs (new data prediction; archives flows.mitm before processing)
#   ./run_project.sh dashboard -> To launch the Streamlit dashboard

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 [train|predict|dashboard]"
    exit 1
fi

MODE=$1

if [ "$MODE" == "train" ]; then
    echo "Step 1: Processing flows from initial capture..."
    python3 process_flows.py "flows.csv"
    
    echo "Step 2: Performing feature engineering on training data..."
    python3 feature_engineering.py
    
    echo "Step 3: Processing external threat intelligence data..."
    python3 process_external_threat_data.py
    
    echo "Step 4: Training the autoencoder model..."
    python3 autoencoder_model.py
    
    echo "Initial training pipeline complete. Model and artifacts saved in results/."

elif [ "$MODE" == "predict" ]; then
    ARCHIVE_DIR="secure_traffic_data/archive"
    mkdir -p "$ARCHIVE_DIR"
    timestamp=$(date +"%Y%m%d_%H%M%S")
    cp secure_traffic_data/flows_1.mitm "$ARCHIVE_DIR/flows_$timestamp.mitm"
    echo "Archived current flows_1.mitm as flows_$timestamp.mitm"
    
    echo "Step 1: Processing flows from new capture..."
    python3 process_flows.py "flows_new.csv"
    
    echo "Step 2: Performing feature engineering on new data..."
    cp secure_traffic_data/flows_new.csv secure_traffic_data/flows.csv
    python3 feature_engineering.py
    
    echo "Step 3: Running prediction on new data..."
    python3 predict_model.py
    
    echo "Prediction run complete. Check results/ for outputs and metrics."

elif [ "$MODE" == "dashboard" ]; then
    echo "Launching Streamlit dashboard..."
    streamlit run dashboard.py
else
    echo "Invalid mode. Use 'train', 'predict', or 'dashboard'."
    exit 1
fi
