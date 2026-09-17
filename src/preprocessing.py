import os
import sys
# Add parent directory to sys.path to allow absolute imports when running as a script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensorflow as tf
from src.config import RAW_DATA_DIR, IMAGE_SIZE, BATCH_SIZE, VAL_SPLIT, TEST_SPLIT

def get_data_generators():
    """
    Loads and splits the dataset into Train, Validation, and Test sets.
    Returns: train_ds, val_ds, test_ds, class_names
    """
    train_dir = os.path.join(RAW_DATA_DIR, "asl_alphabet_train", "asl_alphabet_train")
    
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Dataset directory not found: {train_dir}")
        
    print(f"Loading dataset from {train_dir}...")
    
    # We want a 70/15/15 split.
    # image_dataset_from_directory supports train/val split. 
    # We will split 70% for training and 30% for validation initially.
    # Then we will split that 30% into two equal halves: 15% validation, 15% testing.
    
    combined_val_split = VAL_SPLIT + TEST_SPLIT
    
    # Load Training Data
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=combined_val_split,
        subset="training",
        seed=123,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical'
    )
    
    # Load combined Validation/Test Data
    val_test_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=combined_val_split,
        subset="validation",
        seed=123,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical'
    )
    
    class_names = train_ds.class_names
    print(f"Class names found: {class_names}")
    
    # Determine the number of batches in val_test_ds
    val_test_batches = tf.data.experimental.cardinality(val_test_ds)
    val_batches = val_test_batches // 2
    
    # Split val_test_ds into val_ds and test_ds
    val_ds = val_test_ds.take(val_batches)
    test_ds = val_test_ds.skip(val_batches)
    
    print(f"Dataset split completed:")
    print(f" - Training batches: {tf.data.experimental.cardinality(train_ds)}")
    print(f" - Validation batches: {tf.data.experimental.cardinality(val_ds)}")
    print(f" - Testing batches: {tf.data.experimental.cardinality(test_ds)}")
    
    # Performance optimization: prefetch and cache
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    return train_ds, val_ds, test_ds, class_names

def get_data_augmentation():
    """
    Returns a Sequential model containing data augmentation layers.
    Appropriate augmentations for ASL alphabet:
    - Small rotation
    - Zoom
    - Width/Height shift
    Note: Horizontal flip is NOT appropriate for all ASL signs because
    ASL has 'handedness' and flipping changes the visual representation
    meaningfully. We will omit horizontal_flip to be linguistically valid.
    """
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomTranslation(0.1, 0.1),
    ])
    return data_augmentation

if __name__ == "__main__":
    print("=== Phase 3: Data Preprocessing ===")
    try:
        train_ds, val_ds, test_ds, class_names = get_data_generators()
        print("Preprocessing test successful!")
    except Exception as e:
        print(f"Error during preprocessing: {e}")
