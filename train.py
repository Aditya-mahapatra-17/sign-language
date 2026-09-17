import os
import json
import argparse
import tensorflow as tf
from src.preprocessing import get_data_generators
from src.model import build_baseline_model, build_transfer_learning_model
from src.config import EPOCHS, MODELS_DIR, OUTPUTS_DIR, PLOTS_DIR

def train(use_transfer_learning=True, epochs=10, fast_mode=False):
    """
    Trains the Sign Language Alphabet model.
    Supports both Full Dataset Training and Fast CPU Training mode.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # 1. Load Data
    print("Loading data generators...")
    train_ds, val_ds, test_ds, class_names = get_data_generators()
    
    # Save class names for inference mapping
    class_names_path = os.path.join(MODELS_DIR, 'class_names.json')
    with open(class_names_path, 'w') as f:
        json.dump(class_names, f)
    print(f"Class names successfully saved to {class_names_path}")
    
    if fast_mode:
        print("\n⚡ FAST CPU MODE ACTIVATED: Subsetting dataset for high-speed local training...")
        train_ds = train_ds.take(200) # ~6,400 images
        val_ds = val_ds.take(50)      # ~1,600 images
        epochs = min(epochs, 5)
    
    # 2. Build Model
    if use_transfer_learning:
        print("\n🚀 Building MobileNetV2 Transfer Learning Model (Pretrained on ImageNet)...")
        model = build_transfer_learning_model()
    else:
        print("\n🏗️ Building Baseline CNN Model...")
        model = build_baseline_model()
        
    model.summary()
        
    # 3. Define Callbacks
    model_path = os.path.join(MODELS_DIR, 'sign_language_model.keras')
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            verbose=1
        )
    ]
    
    # 4. Train
    print(f"\n🔥 Starting model training for {epochs} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )
    
    print(f"\n✅ Training complete! Best weights saved to: {model_path}")
    return history, model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Sign Language Model")
    parser.add_argument('--baseline', action='store_true', help="Use baseline CNN instead of MobileNetV2")
    parser.add_argument('--epochs', type=int, default=5, help="Number of training epochs")
    parser.add_argument('--fast', action='store_true', help="Fast CPU mode for quick local convergence")
    args = parser.parse_args()
    
    print("==================================================")
    print("   LOCAL SIGN LANGUAGE MODEL TRAINING PIPELINE    ")
    print("==================================================")
    train(
        use_transfer_learning=not args.baseline,
        epochs=args.epochs,
        fast_mode=args.fast
    )
