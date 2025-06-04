import requests
import os
import logging
import os
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

import requests
import os
import time
from pathlib import Path

def download_gzip_file(url_dict, category, log_every_n_seconds=10):
    """
    Download a gzip file from a URL and save it locally.

    Args:
        url_dict (dict): dictionary of URLs with categories as keys.
        category (str): Category of the data to download.
        log_every_n_seconds (int, optional): Log progress every n seconds. Defaults to 10.
    """
    url = url_dict[category]

    suffix = "metadata" if "meta" in url else "reviews"

    output_path = (
        Path(__file__).parents[1] / "data" / "01_raw" / f"{category}_{suffix}.jsonl.gz"
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Check if file already exists
    if output_path.exists():
        # Make a HEAD request to get the expected file size
        head_response = requests.head(url)
        if head_response.status_code == requests.codes.ok:
            expected_size = int(head_response.headers.get("content-length", 0))
            actual_size = output_path.stat().st_size

            if expected_size > 0 and actual_size == expected_size:
                print(
                    f"Skipping download: {category}_{suffix} already exists with correct size ({actual_size / 1024 / 1024:.1f}MB)"
                )
                return
            else:
                print(
                    f"File exists but size differs (local: {actual_size / 1024 / 1024:.1f}MB, remote: {expected_size / 1024 / 1024:.1f}MB). Re-downloading..."
                )
        else:
            print(
                f"Unable to verify remote file size. Re-downloading {category}_{suffix}..."
            )

    print(f"Starting download: {category} from {url}")

    # Make the request with stream=True to handle large files
    response = requests.get(url, stream=True)

    # Check if the request was successful
    if response.status_code == requests.codes.ok:
        # Get total file size if available
        total_size = int(response.headers.get("content-length", 0))

        downloaded = 0
        start_time = time.time()
        last_log_time = start_time

        # Write the file to disk
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # Filter out keep-alive chunks
                    f.write(chunk)
                    downloaded += len(chunk)

                    # Log progress every 5 seconds
                    current_time = time.time()
                    if current_time - last_log_time >= log_every_n_seconds:
                        if total_size > 0:
                            percent = downloaded * 100 / total_size
                            speed = (
                                downloaded / (current_time - start_time) / 1024 / 1024
                            )  # MB/s
                            print(
                                f"{category}_{suffix}: {percent:.1f}% complete ({downloaded / 1024 / 1024:.1f}MB/{total_size / 1024 / 1024:.1f}MB) at {speed:.2f} MB/s"
                            )
                        else:
                            print(
                                f"{category}_{suffix}: {downloaded / 1024 / 1024:.1f}MB downloaded"
                            )
                        last_log_time = current_time

        print(f"Download complete: {category}_{suffix} saved to {output_path}")
    else:
        print(
            f"Failed to download {category}_{suffix}. Status code: {response.status_code}"
        )

if __name__ == "__main__":

    DATA_LINKS_REVIEWS = {
        "Automotive": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFiles/Automotive.json.gz",
                        # "Cell_Phones_and_Accessories": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFiles/Cell_Phones_and_Accessories.json.gz",
                        #     "Electronics": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFiles/Electronics.json.gz",
                        "CDs_and_Vinyl": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFiles/CDs_and_Vinyl.json.gz", 
                        "Industrial_and_Scientific": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFiles/Industrial_and_Scientific.json.gz"}

    DATA_LINKS_META = {
        # "Automotive": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/meta_Automotive.json.gz",
        #             # "Cell_Phones_and_Accessories": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/meta_Cell_Phones_and_Accessories.json.gz",
        #                 # "Electronics": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/meta_Electronics.json.gz",
        #             "CDs_and_Vinyl": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/meta_CDs_and_Vinyl.json.gz", 
                    "Industrial_and_Scientific": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/meta_Industrial_and_Scientific.json.gz"}


    for cat in DATA_LINKS_REVIEWS.keys():
        download_gzip_file(DATA_LINKS_REVIEWS, cat)
    
    for cat in DATA_LINKS_META.keys():
        download_gzip_file(DATA_LINKS_META, cat)