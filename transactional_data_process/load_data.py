import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
import pickle
import os
import json
from pathlib import Path


def save_metadata_to_json(metadata, output_path):
    dir_name = os.path.dirname(output_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata saved to {output_path}")

def get_amt_bins(df: pd.DataFrame, number_bins: int = 100, min_amt: int = 0, max_amt: int = 30000):
    """
    Create a mapping of amount bins to token IDs that matches the tokenizer intervals.
    
    Parameters:
    - df: DataFrame containing the 'amt' column.
    - number_bins: Number of bins to create for the amount (default: 100).
    - min_amt: Minimum amount for binning (default: 0).
    - max_amt: Maximum amount for binning (default: 30000).
    
    Returns:
    - amount_bins: Array of bin edges
    - bin_labels: List of bin labels matching tokenizer format
    """
    if 'amt' not in df.columns:
        raise ValueError("DataFrame must contain an 'amt' column.")

    # Calculate interval size to match tokenizer
    interval_size = (max_amt - min_amt) // number_bins  # 30000 / 100 = 300
    
    # Create bin edges with fixed intervals
    amount_bins = []
    for i in range(number_bins + 1):
        amount_bins.append(min_amt + i * interval_size)
    
    # Add infinity for the final open-ended bin
    amount_bins.append(np.inf)
    amount_bins = np.array(amount_bins)

    # Generate bin labels to match tokenizer format
    bin_labels = []
    for i in range(number_bins):
        left = amount_bins[i]
        right = amount_bins[i + 1]
        # Format: [AMOUNT_start_end] to match tokenizer
        label = f"[AMOUNT_{left}_{right}]"
        bin_labels.append(label)
    
    # Add the final 30000-Plus bin to match tokenizer
    bin_labels.append("[AMOUNT_30000_PLUS]")

    # Create token dictionary
    token_dict = {label: f"{idx}" for idx, label in enumerate(bin_labels)}
    save_metadata_to_json(token_dict, 'data/amt_bins.json')
    
    return amount_bins, bin_labels


def save_raw_data(save_dir: str):
    """
    Reads raw data from Hugging Face and saves it locally as CSV files.
    Skips downloading if the files already exist.

    Args:
        save_dir (str): Directory to save the raw data.
    """
    splits = {'train': 'credit_card_transaction_train.csv', 'test': 'credit_card_transaction_test.csv'}
    base_url = "hf://datasets/pointe77/credit-card-transaction/"

    # Ensure the directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Directory '{save_dir}' created.")

    # Define file paths
    train_path = os.path.join(save_dir, 'credit_card_transaction_train_raw.csv')
    test_path = os.path.join(save_dir, 'credit_card_transaction_test_raw.csv')

    # Check if files already exist
    if os.path.exists(train_path) and os.path.exists(test_path):
        print(f"Raw data already exists at '{train_path}' and '{test_path}'. Skipping download.")
        return

    # Read the data
    print("Downloading raw data from Hugging Face...")
    df_train = pd.read_csv(base_url + splits["train"])
    df_test = pd.read_csv(base_url + splits["test"])

    # Save raw data
    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)

    print(f"Raw data saved as '{train_path}' and '{test_path}'.")


def fit_global_label_encoder(train_path: str, test_path: str, save_dir: str = '../data'):
    """
    Fit label encoder on combined train and test data for transaction signatures.
    If the encoder and bins already exist, load and return them instead of recomputing.

    Parameters:
    - train_path: Path to training data
    - test_path: Path to test data
    - save_dir: Directory to save the fitted encoder

    Returns:
    - Fitted LabelEncoder and amount bins information
    """
    # Define paths for saved encoder and bins
    encoder_path = os.path.join(save_dir, 'transaction_type_encoder.pkl')
    bins_path = os.path.join(save_dir, 'amt_bins_info.pkl')

    # Check if the encoder and bins already exist
    if os.path.exists(encoder_path) and os.path.exists(bins_path):
        print("Encoder and bins already exist. Loading from disk...")
        with open(encoder_path, 'rb') as f:
            le = pickle.load(f)
        with open(bins_path, 'rb') as f:
            amt_bins, bin_labels = pickle.load(f)
        print(f"Loaded encoder from {encoder_path}")
        print(f"Loaded amount bins info from {bins_path}")
        return le, amt_bins, bin_labels

    # If not, fit the encoder and compute bins
    print("Fitting global label encoder on combined dataset...")

    # Read both datasets
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # Combine datasets for fitting
    df_combined = pd.concat([df_train, df_test], ignore_index=True)

    # Get amount bins from combined data
    amt_bins, bin_labels = get_amt_bins(df_combined)

    # Process combined data to get all possible transaction signatures
    df_combined_processed = preprocess_data_without_encoding(df_combined, amt_bins, bin_labels)

    # Fit label encoder on all unique transaction signatures
    le = LabelEncoder()
    all_signatures = df_combined_processed["transaction_signature"].unique()
    le.fit(all_signatures)

    # Ensure the save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Directory '{save_dir}' created.")

    # Save the fitted encoder and bins
    with open(encoder_path, 'wb') as f:
        pickle.dump(le, f)
    with open(bins_path, 'wb') as f:
        pickle.dump((amt_bins, bin_labels), f)

    print(f"Global label encoder saved to {encoder_path}")
    print(f"Amount bins info saved to {bins_path}")
    print(f"Total unique transaction types: {len(le.classes_)}")

    return le, amt_bins, bin_labels


def preprocess_data_without_encoding(
    df: pd.DataFrame, 
    amt_bins: np.ndarray,
    bin_labels: list,
    drop_na: bool = True,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Preprocess data without applying label encoding (for fitting the encoder).
    """
    required_columns = ['trans_date_trans_time', 'amt', 'merchant']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    df_processed = df.copy()
    
    if drop_na:
        initial_rows = len(df_processed)
        df_processed = df_processed.dropna(subset=required_columns)
        if verbose and initial_rows > len(df_processed):
            print(f"Dropped {initial_rows - len(df_processed)} rows with missing values")
    
    try:
        df_processed['trans_date_trans_time'] = pd.to_datetime(df_processed['trans_date_trans_time'], errors='coerce')
        invalid_dates = df_processed['trans_date_trans_time'].isnull().sum()
        if invalid_dates > 0 and drop_na:
            df_processed = df_processed.dropna(subset=['trans_date_trans_time'])
    except Exception as e:
        raise ValueError(f"Error converting trans_date_trans_time to datetime: {e}")
    
    if not pd.api.types.is_numeric_dtype(df_processed['amt']):
        df_processed['amt'] = pd.to_numeric(df_processed['amt'], errors='coerce')
    
    df_processed["amt_bin"] = pd.cut(
        df_processed["amt"].abs(),
        bins=amt_bins,
        labels=bin_labels,
        include_lowest=True,
        right=False  # [left, right) intervals
    )

    df_processed['year'] = df_processed['trans_date_trans_time'].dt.year.fillna(0).astype(int)
    df_processed['month'] = df_processed['trans_date_trans_time'].dt.month.fillna(0).astype(int)
    df_processed['day'] = df_processed['trans_date_trans_time'].dt.day.fillna(0).astype(int)
    df_processed['day_of_week'] = df_processed['trans_date_trans_time'].dt.dayofweek.fillna(0).astype(int)
    df_processed['hour'] = df_processed['trans_date_trans_time'].dt.hour.fillna(0).astype(int)
    
    features_for_transaction_type = ["amt_bin", "merchant", "month", "day", "day_of_week"]
    concat_parts = [df_processed[feature].astype(str).replace('nan', 'missing') for feature in features_for_transaction_type]
    df_processed["transaction_signature"] = pd.concat(concat_parts, axis=1).agg('_'.join, axis=1)
    
    return df_processed


def preprocess_data(
    df: pd.DataFrame, 
    label_encoder: LabelEncoder = None,
    amt_bins: np.ndarray = None,
    bin_labels: list = None,
    drop_na: bool = True,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Preprocess data with pre-fitted label encoder.
    """
    df_processed = preprocess_data_without_encoding(df, amt_bins, bin_labels, drop_na, verbose)
    
    if label_encoder is not None:
        # Apply the pre-fitted label encoder
        df_processed["transaction_type_id"] = label_encoder.transform(df_processed["transaction_signature"])
        df_processed["transaction_type_id"] = df_processed["transaction_type_id"].apply(lambda x: f"TRANSACTION_{x}")

    if verbose:
        print(f"Processed data shape: {df_processed.shape}")
    
    return df_processed


def preprocess_and_save_data(
    input_path: str, 
    output_path: str, 
    label_encoder: LabelEncoder,
    amt_bins: np.ndarray,
    bin_labels: list,
    drop_na: bool = True, 
    verbose: bool = False
):
    """
    Reads raw data from input_path, processes it using pre-fitted encoder, and saves processed data to output_path.
    """

    df = pd.read_csv(input_path)
    df_processed = preprocess_data(df, label_encoder, amt_bins, bin_labels, drop_na, verbose)
    df_processed.to_csv(output_path, index=False)
    if verbose:
        print(f"Processed data saved to {output_path}")


def load_fitted_encoder_and_bins(save_dir: str = '../data'):
    """
    Load the pre-fitted label encoder and amount bins.
    """
    encoder_path = os.path.join(save_dir, 'transaction_type_encoder.pkl')
    bins_path = os.path.join(save_dir, 'amt_bins_info.pkl')
    
    with open(encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    
    with open(bins_path, 'rb') as f:
        amt_bins, bin_labels = pickle.load(f)
    
    return label_encoder, amt_bins, bin_labels


def create_folder(folder_path):
    """
    Creates a folder at the specified path if it doesn't already exist.

    Args:
        folder_path (str): The path of the folder to create.

    Returns:
        str: A message indicating whether the folder was created or already exists.
    """
    try:
        # Check if the folder already exists
        if not os.path.exists(folder_path):
            # Create the folder
            os.makedirs(folder_path)
            return f"Folder created at: {folder_path}"
        else:
            return f"Folder already exists at: {folder_path}"
    except Exception as e:
        return f"An error occurred while creating the folder: {e}"
    
if __name__ == "__main__":
    # Save the raw files

    RAW_FILE_PATH = "../data/01_raw"
    PROCESSED_FILE_PATH = "../data/02_process"

    create_folder(RAW_FILE_PATH)
    create_folder(PROCESSED_FILE_PATH)

    save_raw_data(save_dir = RAW_FILE_PATH)

    # Step 1: Fit global label encoder on combined dataset
    label_encoder, amt_bins, bin_labels = fit_global_label_encoder(
        train_path=f'{RAW_FILE_PATH}/credit_card_transaction_train_raw.csv',
        test_path=f'{RAW_FILE_PATH}/credit_card_transaction_test_raw.csv'
    )

    # Step 2: Process both datasets using the fitted encoder
    preprocess_and_save_data(
        input_path=f'{RAW_FILE_PATH}/credit_card_transaction_train_raw.csv',
        output_path=f'{PROCESSED_FILE_PATH}/credit_card_transaction_train_processed.csv',
        label_encoder=label_encoder,
        amt_bins=amt_bins,
        bin_labels=bin_labels,
        drop_na=True,
        verbose=True
    )
    
    preprocess_and_save_data(
        input_path=f'{RAW_FILE_PATH}/credit_card_transaction_test_raw.csv',
        output_path=f'{PROCESSED_FILE_PATH}/credit_card_transaction_test_processed.csv',
        label_encoder=label_encoder,
        amt_bins=amt_bins,
        bin_labels=bin_labels,
        drop_na=True,
        verbose=True
    )
    
    print("Processing complete! Both train and test sets use consistent transaction_type_id encoding.")