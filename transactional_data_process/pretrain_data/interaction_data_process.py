import pandas as pd
import sys
from typing import List
from helper import save_metadata_to_json
import json

def extract_interaction_from_file(
    df: pd.DataFrame, 
    json_meta_path: str,
    group_col: str = "cc_num"
) -> List[List[int]]:
    """
    Extract ordered lists of transaction_type_id per group from a CSV or Parquet file,
    filtered by transaction_type_ids provided in a JSON metadata file.

    Parameters:
    -----------
    df : dataframe 
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

    
    required_cols = {group_col, 'transaction_type_id', 'trans_date_trans_time'}
    if not required_cols.issubset(df.columns):
        print(f"Missing one or more required columns: {required_cols}")
        sys.exit(1)
    
    # Group, sort, filter, and extract sequences
    sorted_sequences = []
    for _, group in df.groupby(group_col):
        sorted_group = group.sort_values(by='trans_date_trans_time')
        sequence = sorted_group['transaction_type_id'].tolist()
        valid_sequence_ids = [s for s in sequence if s in valid_type_ids]
        sorted_sequences.append(valid_sequence_ids)

    print(f"Extracted {len(sorted_sequences)} filtered sequences based on '{group_col}'.")
    return sorted_sequences


def get_dataframe(train_path:str ,test_path:str = None):
    # Read both datasets
    df = pd.read_csv(train_path)
    if test_path:
        df_test = pd.read_csv(test_path)
        # Combine datasets for fitting
        df = pd.concat([df, df_test], ignore_index=True)

    return df


def main():
    input_file_path_train = '../../data/02_process/credit_card_transaction_train_processed.csv'
    # input_file_path_test = '../../data/02_process/credit_card_transaction_test_processed.csv'

    df = get_dataframe(train_path=input_file_path_train)
    meta_data_path = "meta_data.json"

    all_sequences = extract_interaction_from_file(df, meta_data_path)
    train_idx = int(len(all_sequences) * 0.85)
    train_sequences = all_sequences[:train_idx]
    val_sequences = all_sequences[train_idx:]
    
    print(f"Total sequences: {len(all_sequences)}")
    print(f"Training sequences: {len(train_sequences)}, Validation sequences: {len(val_sequences)}")

    # Save the sequences to files
    save_metadata_to_json(train_sequences, 'train.json')
    save_metadata_to_json(val_sequences, 'dev.json')

if __name__ == "__main__":
    main()
