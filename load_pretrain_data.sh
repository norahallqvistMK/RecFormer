#!/bin/bash

# Navigate to the pretrain_data folder
cd pretrain_data || { echo "Folder 'pretrain_data' not found!"; exit 1; }

# Run the Python scripts in sequence
echo "Running data_load.py..."
python3 download_data.py || { echo "Error running download_data.py"; exit 1; }

echo "Running meta_data_process.py..."
python meta_data_process.py || { echo "Error running meta_data_process.py"; exit 1; }

echo "Running interaction_data_process.py..."
python interaction_data_process.py || { echo "Error running interaction_data_process.py"; exit 1; }


echo "All scripts executed successfully!"