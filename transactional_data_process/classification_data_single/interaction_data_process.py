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

def extract_sequences(df, meta_dict=None, group_col="cc_num", debug=True):
    """
    Extract transaction sequences per group from a DataFrame.

    Parameters:
    - df (pd.DataFrame): Input dataframe containing transaction data.
    - meta_dict (dict, optional): Dictionary of valid transaction types.
    - group_col (str): Column to group transactions by.
    - debug (bool): Whether to print debug information.

    Returns:
    - dict: Dictionary mapping each group ID to a tuple (list of transaction_type_ids, list of fraud flags).
    """
    # Filter based on metadata
    if meta_dict:
        valid_transactions = set(meta_dict.keys())
        df = df[df['transaction_type_id'].isin(valid_transactions)]

    # Sort the DataFrame once
    df_sorted = df.sort_values([group_col, 'trans_date_trans_time'])

    if debug:
        total_fraud = df_sorted['is_fraud'].sum()
        print(f"Total fraud transactions: {total_fraud}")

    # Group and extract sequences
    sequences = {}
    grouped = df_sorted.groupby(group_col)

    for group_id, group in grouped:
        trans_ids = group['transaction_type_id'].tolist()
        fraud_flags = group['is_fraud'].tolist()
        if trans_ids:
            for id, (i, f) in enumerate(zip(trans_ids, fraud_flags)):
                sequences[f"{group_id}_{id}"] = [[i], f]

    if debug:
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

def split_data(sequences, train_ratio=0.8, val_ratio=0.10, test_ratio=0.10, seed=42):
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
    file_path_test = "../../data/02_process/credit_card_transaction_test_processed.csv"
    output_path = "."
    meta_data_path = "meta_data.json"

    df = get_dataframe(file_path_train, file_path_test)
    
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