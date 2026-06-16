# Paste your utility functions here
import pandas as pd
import numpy as np
import os
import subprocess
import paths
from paths import PATHS
import sys
import importlib

# Force reload both to be safe
def reload_files():
  importlib.reload(paths)
  importlib.reload(utils)



print("reloaded")
def ensure_image_extracted(article_id, zip_name='h-and-m-personalized-fashion-recommendations.zip'):
    """
    Checks if an image exists. If not, extracts its parent shard from the ZIP.
    """
    sys.path.append("/content/drive/MyDrive/hm_fashion_project/src")
    from paths import RAW_DATA_PATH

    # 1. Setup paths
    formatted_id = str(article_id).zfill(10)
    subfolder = formatted_id[:3]
    
    # Path where the file SHOULD be
    target_path = os.path.join(PATHS['images'], subfolder, f"{formatted_id}.jpg")
    
    # 2. Check existence
    if os.path.exists(target_path):
        return target_path

    # 3. If missing, extract the entire shard (e.g., images/072/*)
    print(f"📦 Shard {subfolder} missing. Extracting from ZIP...")
    zip_full_path = os.path.join(RAW_DATA_PATH, zip_name)
    
    # Internal ZIP path
    internal_path = f"images/{subfolder}/*"
    
    # Run the unzip command
    # -j junk paths (doesn't create the 'images/' parent folder)
    # -d extract into the specific shard folder
    extract_to = os.path.join(PATHS['images'], subfolder)
    os.makedirs(extract_to, exist_ok=True)
    
    try:
        # Use subprocess for cleaner execution inside a function
        subprocess.run([
            "unzip", "-q", "-j", zip_full_path, 
            internal_path, "-d", extract_to
        ], check=True)
        print(f"✅ Shard {subfolder} extracted successfully.")
        return target_path
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return None
def reduce_mem_usage(df):
    """ Iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    
    return df
