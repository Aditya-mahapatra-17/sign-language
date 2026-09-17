import os
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from src.preprocessing import get_data_generators
from src.config import MODELS_DIR, PLOTS_DIR

def evaluate_model():
    """
    Evaluates the saved model and generates classification metrics.
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    model_path = os.path.join(MODELS_DIR, 'sign_language_model.keras')
    class_names_path = os.path.join(MODELS_DIR, 'class_names.json')
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please train first.")
        return
        
    print("Loading datasets...")
    train_ds, val_ds, test_ds, _ = get_data_generators()
    
    with open(class_names_path, 'r') as f:
        class_names = json.load(f)
        
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    print("Evaluating on test dataset...")
    # Extract true labels and predictions
    y_true = []
    y_pred = []
    
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))
        
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    print("\n=== Classification Report ===")
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    print(report)
    
    with open(os.path.join(PLOTS_DIR, 'classification_report.txt'), 'w') as f:
        f.write(report)
        
    print("\nGenerating Confusion Matrix...")
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(20, 16))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Sign Language Alphabet')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    cm_path = os.path.join(PLOTS_DIR, 'confusion_matrix.png')
    plt.savefig(cm_path, bbox_inches='tight')
    plt.close()
    
    print(f"Evaluation complete. Saved metrics and confusion matrix to {PLOTS_DIR}")

if __name__ == "__main__":
    print("=== Phase 7: Evaluation ===")
    evaluate_model()
