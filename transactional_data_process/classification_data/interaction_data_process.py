import json
import os
import random
import argparse
from collections import defaultdict
import pandas as pd
from tqdm import tqdm


class LabelField:
    def __init__(self):
        self.label2id = dict()
        self.label_num = 0

    def get_id(self, label):
        
        if label in self.label2id:
            return self.label2id[label]
        
        self.label2id[label] = self.label_num
        self.label_num += 1

        return self.label2id[label]

def extract_metadata(file_path):
    """Extract metadata for each transaction type from CSV file."""
    df = pd.read_csv(file_path, nrows=100000)
    print(f"Loaded metadata from {file_path}, shape: {df.shape}")
    
    # Define metadata columns to extract
    meta_columns = {
        "amount": "amt_bin",
        "merchant": "merchant", 
        "year": "year",
        "month": "month",
        "day": "day",
        "weekday": "day_of_week"
    }
    
    # Get available columns
    available_cols = {k: v for k, v in meta_columns.items() if v in df.columns}
    missing_cols = [v for v in meta_columns.values() if v not in df.columns]
    
    if missing_cols:
        print(f"Warning: Missing columns: {missing_cols}")
    
    # Extract metadata per transaction type
    metadata = {}
    for trans_id, group in df.groupby("transaction_type_id"):
        first_row = group.iloc[0]
        metadata[trans_id] = {
            key: str(first_row[col]) for key, col in available_cols.items()
        }
    
    print(f"Extracted metadata for {len(metadata)} transaction types")
    return metadata


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
    sequences = {}
    
    for group_id, group in df_sorted.groupby(group_col):
        trans_ids = group['transaction_type_id'].tolist()
        fraud_flags = group['is_fraud'].tolist()
        
        if trans_ids and fraud_flags:
            # Use any() for faster fraud detection
            has_fraud = 1 if any(fraud_flags) else 0
            sequences[group_id] = [trans_ids, [has_fraud]]
    
    print(f"Extracted {len(sequences)} sequences")
    return sequences


def split_test_data(test_sequences, dev_ratio=0.5, seed=42):
    """Split test data into dev and test sets."""
    keys = list(test_sequences.keys())
    random.seed(seed)
    random.shuffle(keys)
    
    split_point = int(len(keys) * dev_ratio)
    dev_keys = keys[:split_point]
    test_keys = keys[split_point:]
    
    dev_dict = {k: test_sequences[k] for k in dev_keys}
    test_dict = {k: test_sequences[k] for k in test_keys}
    
    return dev_dict, test_dict

def split_data(sequences, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """Split sequences into train/val/test sets."""
    keys = list(sequences.keys())
    random.seed(seed)
    random.shuffle(keys)
    
    total = len(keys)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    train_keys = keys[:train_end]
    val_keys = keys[train_end:val_end]
    test_keys = keys[val_end:]
    
    train_dict = {k: sequences[k] for k in train_keys}
    val_dict = {k: sequences[k] for k in val_keys}
    test_dict = {k: sequences[k] for k in test_keys}
    
    return train_dict, val_dict, test_dict

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
    sequences = extract_sequences(df, metadata)

    # Encode labels
    cc_encoder = LabelField()
    transaction_encoder = LabelField()
    
    # Process training data
    train_dict = {}
    for cc_id, (trans_ids, label) in sequences.items():
        encoded_cc = cc_encoder.get_id(cc_id)
        encoded_trans = [transaction_encoder.get_id(t) for t in trans_ids]
        train_dict[encoded_cc] = [encoded_trans, label]

    train_dict, val_dict, test_dict = split_data(train_dict)
    
    print(f'Users: {len(cc_encoder.label2id)}, '
          f'Transactions: {len(transaction_encoder.label2id)}, '
          f'Train: {len(train_dict)}, '
          f'Dev: {len(val_dict)}, '
          f'Test: {len(test_dict)}')
    
    # Save all files
    output_files = {
        'train.json': train_dict,
        'val.json': val_dict,
        'test.json': test_dict,
        'umap.json': cc_encoder.label2id,
        'smap.json': transaction_encoder.label2id}
    
    for filename, data in output_files.items():
        filepath = os.path.join(output_path, filename)
        save_json(data, filepath)
        print(f"Saved {filename}")

if __name__ == "__main__":
    main()