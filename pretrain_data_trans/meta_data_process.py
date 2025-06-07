import pandas as pd
import json
from typing import Dict, Any
import os
import sys
from helper import save_metadata_to_json

def extract_meta_data_from_file(file_path: str) -> Dict[int, Dict[str, Any]]:
    """
    Extract metadata for each unique transaction_type_id from a CSV file.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file with transaction data
        
    Returns:
    --------
    Dict[int, Dict[str, Any]]
        Dictionary where keys are transaction_type_id and values are metadata dicts
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded data from {file_path}. Shape: {df.shape}")
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        sys.exit(1)
    
    if "transaction_type_id" not in df.columns:
        print("Error: 'transaction_type_id' column not found in data")
        sys.exit(1)
    
    metadata_columns = {
        "amount": "amt_bin",
        "merchant": "merchant", 
        "month": "month",
        "day": "day",
        "weekday": "day_of_week"
    }
    
    available_columns = {key: col for key, col in metadata_columns.items() if col in df.columns}
    missing_columns = [col for col in metadata_columns.values() if col not in df.columns]
    if missing_columns:
        print(f"Warning: Missing columns in data: {missing_columns}")
    
    meta_data = {}
    
    for transaction_type_id, group in df.groupby("transaction_type_id"):
        first_row = group.iloc[0]
        # Convert numpy types to native Python types using .item() where available
        type_metadata = {
            key: first_row[col].item() if hasattr(first_row[col], 'item') else first_row[col]
            for key, col in available_columns.items()
        }
        meta_data[transaction_type_id] = type_metadata
    
    print(f"Extracted metadata for {len(meta_data)} unique transaction_type_id values.")
    return meta_data

def main():
    input_file_path = '../data/02_process/credit_card_transaction_train_processed.csv'
    output_file_path = 'meta_data.json'
    
    meta_data = extract_meta_data_from_file(input_file_path)
    save_metadata_to_json(meta_data, output_file_path)

if __name__ == "__main__":
    main()
