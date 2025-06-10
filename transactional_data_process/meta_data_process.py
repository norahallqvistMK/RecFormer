import pandas as pd
import json
import os 
from itertools import islice
import argparse

def convert_to_serializable(value):
    """Convert pandas/numpy types to JSON-serializable Python types"""
    if pd.isna(value):
        return None
    elif pd.api.types.is_integer_dtype(type(value)):
        return int(value)
    elif pd.api.types.is_float_dtype(type(value)):
        return float(value)
    elif pd.api.types.is_bool_dtype(type(value)):
        return bool(value)
    else:
        return str(value)

def extract_metadata(df):
    """Extract metadata for each transaction type from CSV file."""
    
    # Define metadata columns to extract
    meta_columns = {
        "amount": "amt",
        "merchant": "merchant", 
        "month": "month",
        "day": "day",
        "weekday": "day_of_week"
    }
    
    # Get available columns
    available_cols = {k: v for k, v in meta_columns.items() if v in df.columns}
    missing_cols = [v for v in meta_columns.values() if v not in df.columns]
    
    if missing_cols:
        print(f"Warning: Missing columns: {missing_cols}")
    

    metadata = {}
    for trans_id, group in df.groupby("transaction_type_id"):
        first_row = group.iloc[0]
        metadata[trans_id] = {
            key: convert_to_serializable(first_row[col])
            for key, col in available_cols.items()
        }
    
    print(f"Extracted metadata for {len(metadata)} transaction types")
    return metadata

def save_json(data, filepath):
    """Save data to JSON file."""
    with open(filepath, 'w', encoding='utf8') as f:
        json.dump(data, f)

def save_meta_data(train_path:str ,test_path:str = None, number_items: int = None, output_dir:str = ""):
    # Read both datasets

    df = pd.read_csv(train_path)
    if test_path:
        df_test = pd.read_csv(test_path)
        # Combine datasets for fitting
        df = pd.concat([df, df_test], ignore_index=True)

    meta_dict = extract_metadata(df)
    if number_items:
        meta_dict = dict(islice(meta_dict.items(), number_items))

    save_json(meta_dict, os.path.join(output_dir, "meta_data.json"))
    print("Saved meta_dict in current directory")

if __name__ == "__main__":

    TRAIN_PATH ='../data/02_process/credit_card_transaction_train_processed.csv'
    # TEST_PATH='../data/02_process/credit_card_transaction_test_processed.csv'

    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Extract metadata from transaction datasets and save as JSON.")
    parser.add_argument("--train_path", type=str, default=TRAIN_PATH, help="Path to the training dataset CSV file.")
    parser.add_argument("--test_path", type=str, default=None, help="Path to the test dataset CSV file (optional).")
    parser.add_argument("--number_items", type=int, default=50000, help="Number of items to include in the metadata (optional).")
    parser.add_argument("--output_dir", type=str, default="", help="Directory to save the output JSON file (optional).")

    # Parse the arguments
    args = parser.parse_args()

    # Call the function with the parsed arguments
    save_meta_data(
        train_path=args.train_path,
        test_path=args.test_path,
        number_items=args.number_items,
        output_dir=args.output_dir
    )