#!/bin/bash

# Step 1: Download and process the data
echo "Downloading and processing the data..."
python load_data.py

# Step 2: Define output directories and parameters
OUTPUT_DIRS=("pretrain_data" "classification_data" "finetune_data")
NUMBER_ITEMS=20000

# Step 3: Run metadata extraction for each output directory
for OUTPUT_DIR in "${OUTPUT_DIRS[@]}"; do
    echo "Running metadata extraction for $OUTPUT_DIR..."
    python meta_data_process.py \
        --number_items "$NUMBER_ITEMS" \
        --output_dir "$OUTPUT_DIR"
    echo "Metadata extraction for $OUTPUT_DIR completed."
done

# Step 4: Process interaction data in each output directory
for OUTPUT_DIR in "${OUTPUT_DIRS[@]}"; do
    echo "Processing interaction data in $OUTPUT_DIR..."
    cd "$OUTPUT_DIR" || { echo "Error: Failed to cd into $OUTPUT_DIR"; exit 1; }
    python interaction_data_process.py
    cd - > /dev/null || { echo "Error: Failed to return to the previous directory"; exit 1; }
    echo "Interaction data processing completed in $OUTPUT_DIR."
done