import os
from src.model import build_transfer_learning_model
from src.config import MODELS_DIR

if __name__ == "__main__":
    print("Creating a dummy model for inference development...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    model = build_transfer_learning_model()
    model_path = os.path.join(MODELS_DIR, 'sign_language_model.keras')
    model.save(model_path)
    print(f"Dummy model saved to {model_path}. Use train.py to train a real one.")
