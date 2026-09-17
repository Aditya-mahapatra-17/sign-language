import os
import sys
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Ensure absolute imports work when run as script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import EPOCHS, MODELS_DIR, PLOTS_DIR
from src.preprocessing import get_data_generators
from src.model import build_baseline_model

def plot_history(history, save_path):
    """Plots and saves the training and validation accuracy and loss."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 6))
    
    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Training history plot saved to {save_path}")

def train():
    """Main function to execute the training pipeline."""
    print("=== Phase 5: Model Training ===")
    
    # 1. Load Data
    train_ds, val_ds, test_ds, class_names = get_data_generators()
    
    # Take a smaller subset (200 batches = 6400 images) for fast prototyping!
    train_ds = train_ds.take(200)
    val_ds = val_ds.take(40)
    
    num_classes = len(class_names)
    
    # 2. Build Model
    model = build_baseline_model(num_classes=num_classes)
    model.summary()
    
    # 3. Setup Callbacks
    os.makedirs(MODELS_DIR, exist_ok=True)
    best_model_path = os.path.join(MODELS_DIR, 'best_baseline_model.keras')
    
    callbacks = [
        ModelCheckpoint(
            filepath=best_model_path,
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    # 4. Train Model
    print(f"Starting training for up to {EPOCHS} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    
    # 5. Save History Plot
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_path = os.path.join(PLOTS_DIR, 'training_history.png')
    plot_history(history, plot_path)
    
    print("Training complete!")

if __name__ == "__main__":
    train()
