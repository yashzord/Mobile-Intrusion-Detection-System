#!/bin/bash
# run_project.sh - Master script to run the complete Mobile Intrusion Detection Pipeline

echo "Step 2: Perform feature engineering..."
python3 feature_engineering.py

echo "Step 3: Process external threat intelligence data..."
python3 process_external_threat_data.py

echo "Step 4: Train the autoencoder and perform anomaly detection..."
python3 autoencoder_model.py

echo "Pipeline execution complete. Check the results/ directory for outputs."
