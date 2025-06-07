import json
from collections import defaultdict
import gzip
import random
from tqdm import tqdm
import argparse
import os
import pandas as pd
import sys
from typing import Dict, Any, List

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


parser = argparse.ArgumentParser()
    
parser.add_argument('--file_path', default='../data/02_process/credit_card_transaction_train_processed.csv', help='Processing file path (.gz file).')
# parser.add_argument('--meta_file_path', default='../data/02_raw/Industrial_and_Scientific_metadata.jsonl.gz', help='Processing file path (.gz file).')
parser.add_argument('--output_path', default='.', help='Output directory')
args = parser.parse_args()


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
        "year" : "year",
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

        # CONVERTING EVERYTHIGN TO STRING TO WORK WITH CURRETN MODEL
        type_metadata = {
            key: str(first_row[col].item()) if hasattr(first_row[col], 'item') else first_row[col]
            for key, col in available_columns.items()
        }
        meta_data[transaction_type_id] = type_metadata
    
    print(f"Extracted metadata for {len(meta_data)} unique transaction_type_id values.")
    return meta_data

def extract_interaction_from_file(file_path: str, group_col: str = "cc_num") -> List[List[int]]:
    """
    Extract ordered lists of transaction_type_id per group from a CSV or Parquet file.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV or Parquet file containing the transaction data
    group_col : str
        Column name to group by (default: "cc_num")
        
    Returns:
    --------
    List[List[int]]
        List of ordered transaction_type_id lists, one per group
    """
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
    
    if group_col not in df.columns or 'transaction_type_id' not in df.columns or 'trans_date_trans_time' not in df.columns:
        print(f"Required columns ('{group_col}', 'transaction_type_id', 'trans_date_trans_time') not found in data")
        sys.exit(1)
    
    # Group, sort, and extract sequences
    sorted_sequences = {}
    groups = df.groupby(group_col)
    for name, group in groups:
        sorted_group = group.sort_values(by='trans_date_trans_time')
        sorted_sequence = sorted_group['transaction_type_id'].tolist()
        sorted_sequences[name] = sorted_sequence
    
    print(f"Extracted {len(sorted_sequences)} sequences based on '{group_col}'.")
    return sorted_sequences

meta_dict = extract_meta_data_from_file(args.file_path)       
raw_sequences = extract_interaction_from_file(args.file_path)       

output_path = args.output_path
if not os.path.exists(output_path):
    os.mkdir(output_path)

input_file = args.file_path
train_file = os.path.join(output_path, 'train.json')
dev_file = os.path.join(output_path, 'val.json')
test_file = os.path.join(output_path, 'test.json')
umap_file = os.path.join(output_path, 'umap.json')
smap_file = os.path.join(output_path, 'smap.json')
meta_file = os.path.join(output_path, 'meta_data.json')

CC_field = LabelField()
transaction_field = LabelField()
sequences = defaultdict(list)

for k, v in raw_sequences.items():
    if len(raw_sequences[k]) > 3: 
        sequences[CC_field.get_id(k)] = [transaction_field.get_id(ele) for ele in v]

train_dict = dict()
dev_dict = dict()
test_dict = dict()

intersections = 0

for k, v in tqdm(sequences.items()):
    length = len(sequences[k])
    intersections += length
    if length<3:
        train_dict[k] = sequences[k]
    else:
        train_dict[k] = sequences[k][:length-2]
        dev_dict[k] = [sequences[k][length-2]]
        test_dict[k] = [sequences[k][length-1]]

print(f'Users: {len(CC_field.label2id)}, Items: {len(transaction_field.label2id)}, Intersects: {intersections}')

f_u = open(umap_file, 'w', encoding='utf8')
json.dump(CC_field.label2id, f_u)
f_u.close()

f_s = open(smap_file, 'w', encoding='utf8')
json.dump(transaction_field.label2id, f_s)
f_s.close()

train_f = open(train_file, 'w', encoding='utf8')
json.dump(train_dict, train_f)
train_f.close()

dev_f = open(dev_file, 'w', encoding='utf8')
json.dump(dev_dict, dev_f)
dev_f.close()

test_f = open(test_file, 'w', encoding='utf8')
json.dump(test_dict, test_f)
test_f.close()

meta_f = open(meta_file, 'w', encoding='utf8')
json.dump(meta_dict, meta_f)
meta_f.close()

