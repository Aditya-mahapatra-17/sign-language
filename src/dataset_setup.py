import os
import sys
import zipfile
import subprocess
from config import RAW_DATA_DIR, NUM_CLASSES

def check_kaggle_installed():
    try:
        import kaggle
        return True
    except ImportError:
        return False

def download_dataset():
    """Downloads the ASL Alphabet dataset from Kaggle if not present."""
    dataset_name = "grassknoted/asl-alphabet"
    target_zip = os.path.join(RAW_DATA_DIR, "asl-alphabet.zip")
    
    if os.path.exists(os.path.join(RAW_DATA_DIR, "asl_alphabet_train")):
        print("Dataset already appears to be extracted in data/raw/.")
        return True

    if not os.path.exists(target_zip):
        print("Downloading dataset from Kaggle...")
        if not check_kaggle_installed():
            print("Error: The 'kaggle' python package is not installed.")
            print("Run 'pip install kaggle' and ensure your kaggle.json is in ~/.kaggle/")
            print("Alternatively, download manually from: https://www.kaggle.com/datasets/grassknoted/asl-alphabet")
            print("And place the extracted contents into data/raw/")
            return False
        
        try:
            # Requires kaggle.json to be set up in ~/.kaggle/
            subprocess.run(["kaggle", "datasets", "download", "-d", dataset_name, "-p", RAW_DATA_DIR], check=True)
        except subprocess.CalledProcessError:
            print("Failed to download via Kaggle API. Please ensure your Kaggle API token is configured.")
            return False
            
    if os.path.exists(target_zip):
        print(f"Extracting {target_zip}...")
        with zipfile.ZipFile(target_zip, 'r') as zip_ref:
            zip_ref.extractall(RAW_DATA_DIR)
        print("Extraction complete.")
        
        # Cleanup zip to save space
        os.remove(target_zip)
        return True
        
    return False

def validate_dataset():
    """Validates the structure of the extracted dataset."""
    train_dir = os.path.join(RAW_DATA_DIR, "asl_alphabet_train", "asl_alphabet_train")
    if not os.path.exists(train_dir):
        print(f"Validation failed: Could not find training directory at {train_dir}")
        return False
        
    classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    print(f"Found {len(classes)} classes.")
    
    if len(classes) != NUM_CLASSES:
        print(f"Warning: Expected {NUM_CLASSES} classes, but found {len(classes)}.")
        
    print("Validation successful. The dataset is ready for preprocessing.")
    return True

def create_dummy_dataset():
    """Creates a dummy dataset for testing the pipeline if the real one isn't available."""
    print("Creating a minimal dummy dataset for pipeline testing...")
    import numpy as np
    from PIL import Image
    
    # 29 classes: A-Z, space, del, nothing
    classes = [chr(i) for i in range(65, 91)] + ["space", "del", "nothing"]
    train_dir = os.path.join(RAW_DATA_DIR, "asl_alphabet_train", "asl_alphabet_train")
    os.makedirs(train_dir, exist_ok=True)
    
    for cls in classes:
        cls_dir = os.path.join(train_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)
        # Create 5 random images per class
        for i in range(5):
            img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(os.path.join(cls_dir, f"{cls}_{i}.jpg"))
            
    print("Dummy dataset created.")

if __name__ == "__main__":
    print("=== Phase 2: Dataset Preparation ===")
    success = download_dataset()
    if not success:
        create_dummy_dataset()
    validate_dataset()
