import json
import os
import random
import argparse
from collections import defaultdict
import pandas as pd
from tqdm import tqdm

def extract_sequences(df, meta_dict=None, group_col="cc_num"):
    """Extract transaction sequences per group from CSV file - optimized version."""
    
    if meta_dict is None:
        valid_transactions = set()
    else:
        valid_transactions = set(meta_dict.keys())
    
    # Pre-filter the dataframe if metadata is provided
    if valid_transactions:
        df_filtered = df[df['transaction_type_id'].isin(valid_transactions)]
    else:
        df_filtered = df
    
    # Sort once for the entire dataframe
    df_sorted = df_filtered.sort_values(['cc_num', 'trans_date_trans_time'])
    
    # Group and process using vectorized operations
    sorted_sequences = []
    for group_id, group in df_sorted.groupby(group_col):
        sorted_sequence = group['transaction_type_id'].tolist()
        #need to be at least 2 to be a sequence
        if len(sorted_sequence)>2:
            sorted_sequences.append(sorted_sequence)
    
    print(f"Extracted {len(sorted_sequences)} sequences")
    return sorted_sequences


def save_json(data, filepath):
    """Save data to JSON file."""
    with open(filepath, 'w', encoding='utf8') as f:
        json.dump(data, f)

def get_dataframe(train_path:str ,test_path:str = None):
    # Read both datasets
    df = pd.read_csv(train_path)
    if test_path:
        df_test = pd.read_csv(test_path)
        # Combine datasets for fitting
        df = pd.concat([df, df_test], ignore_index=True)

    return df

def main():
    # Parse arguments
    file_path_train = "../../data/02_process/credit_card_transaction_train_processed.csv"
    # file_path_test = "../../data/02_process/credit_card_transaction_test_processed.csv"
    output_path = "."
    meta_data_path = "meta_data.json"

    df = get_dataframe(file_path_train)
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Extract data meta data
    print("Getting metadata...")
    with open(meta_data_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    print("Extracting sequences...")
    all_sequences = extract_sequences(df, metadata)

    # Encode labels

    train_idx = int(len(all_sequences) * 0.85)
    train_sequences = all_sequences[:train_idx]
    val_sequences = all_sequences[train_idx:]

    print(f'Train: {len(train_sequences)}, '
          f'Dev: {len(val_sequences)}')
    
    # Save all files
    output_files = {
        'train.json': train_sequences,
        'dev.json': val_sequences}
    
    for filename, data in output_files.items():
        filepath = os.path.join(output_path, filename)
        save_json(data, filepath)
        print(f"Saved {filename}")

if __name__ == "__main__":
    main()