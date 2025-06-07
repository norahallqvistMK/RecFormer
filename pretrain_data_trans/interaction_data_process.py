import pandas as pd
import sys
from typing import List
from helper import save_metadata_to_json
import json

def extract_interaction_from_file(
    file_path: str, 
    json_meta_path: str,
    group_col: str = "cc_num"
) -> List[List[int]]:
    """
    Extract ordered lists of transaction_type_id per group from a CSV or Parquet file,
    filtered by transaction_type_ids provided in a JSON metadata file.

    Parameters:
    -----------
    file_path : str
        Path to the CSV or Parquet file containing the transaction data.
    json_meta_path : str
        Path to a JSON file containing a dictionary where keys are valid transaction_type_ids.
    group_col : str
        Column name to group by (default: "cc_num").

    Returns:
    --------
    List[List[int]]
        List of ordered transaction_type_id sequences (one per group), 
        only including groups with at least one transaction_type_id in the JSON keys.
    """
    # Load metadata
    try:
        with open(json_meta_path, "r") as f:
            meta_data = json.load(f)
            valid_type_ids = set(str(k) for k in meta_data.keys())
        print(f"Loaded metadata from {json_meta_path}. Found {len(valid_type_ids)} valid transaction_type_ids.")
    except Exception as e:
        print(f"Error reading metadata file {json_meta_path}: {e}")
        sys.exit(1)

    # Load transaction data
    try:
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif file_path.endswith(".parquet"):
            df = pd.read_parquet(file_path)
        else:
            print(f"Unsupported file format for {file_path}. Please use CSV or Parquet.")
            sys.exit(1)
        print(f"Loaded data from {file_path}. Shape: {df.shape}")
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        sys.exit(1)
    
    required_cols = {group_col, 'transaction_type_id', 'trans_date_trans_time'}
    if not required_cols.issubset(df.columns):
        print(f"Missing one or more required columns: {required_cols}")
        sys.exit(1)
    
    # Group, sort, filter, and extract sequences
    sorted_sequences = []
    for _, group in df.groupby(group_col):
        sorted_group = group.sort_values(by='trans_date_trans_time')
        sequence = sorted_group['transaction_type_id'].tolist()
        if any(tid in valid_type_ids for tid in sequence):
            sorted_sequences.append(sequence)

    print(f"Extracted {len(sorted_sequences)} filtered sequences based on '{group_col}'.")
    return sorted_sequences

def main():
    input_file_path = '../data/02_process/credit_card_transaction_train_processed.csv'
    meta_data_path = "meta_data.json"

    all_sequences = extract_interaction_from_file(input_file_path, meta_data_path)
    train_idx = int(len(all_sequences) * 0.85)
    train_sequences = all_sequences[:train_idx]
    val_sequences = all_sequences[train_idx:]
    
    print(f"Total sequences: {len(all_sequences)}")
    print(f"Training sequences: {len(train_sequences)}, Validation sequences: {len(val_sequences)}")

    # Save the sequences to files
    save_metadata_to_json(train_sequences, 'train.json')
    save_metadata_to_json(val_sequences, 'dev.json')

    # input_file_path_test = '../data/credit_card_transaction_train_processed.csv'


if __name__ == "__main__":
    main()
