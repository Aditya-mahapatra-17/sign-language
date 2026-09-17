import os
import json
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2
import torch.nn as nn
from src.hand_detection import HandDetector
from src.landmark_rules import LandmarkRuleEngine
from src.config import MODELS_DIR

HF_FINETUNED_DIR = os.path.join(MODELS_DIR, "hf_finetuned_siglip")
PYTORCH_MODEL_PATH = os.path.join(MODELS_DIR, "sign_language_pytorch.pt")

class SignLanguageInference:
    def __init__(self):
        """
        Initializes the Sign Language Inference Engine with:
        1. Deep Learning Model (Vision Transformer / MobileNet)
        2. MediaPipe Hand Landmark Tracking
        3. Geometric Disambiguation Rule Engine for confusing gesture pairs
        """
        self.detector = HandDetector(max_num_hands=1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = "none"
        
        # 1. Check for Fine-Tuned Hugging Face SigLIP2 Model
        if os.path.exists(HF_FINETUNED_DIR) and os.path.exists(os.path.join(HF_FINETUNED_DIR, "model.safetensors")):
            try:
                from transformers import AutoImageProcessor, AutoModelForImageClassification
                self.processor = AutoImageProcessor.from_pretrained(HF_FINETUNED_DIR)
                self.model = AutoModelForImageClassification.from_pretrained(HF_FINETUNED_DIR).to(self.device)
                self.model.eval()
                self.id2label = self.model.config.id2label
                self.class_names = [self.id2label.get(i, self.id2label.get(str(i), f"Class {i}")) for i in range(len(self.id2label))]
                self.model_type = "huggingface"
            except Exception:
                pass

        # 2. Check for Custom PyTorch MobileNet Model
        if self.model_type == "none" and os.path.exists(PYTORCH_MODEL_PATH):
            try:
                checkpoint = torch.load(PYTORCH_MODEL_PATH, map_location=self.device)
                self.class_names = checkpoint['class_names']
                
                self.model = mobilenet_v2(weights=None)
                in_features = self.model.classifier[1].in_features
                self.model.classifier = nn.Sequential(
                    nn.Dropout(p=0.4),
                    nn.Linear(in_features, 256),
                    nn.ReLU(),
                    nn.Dropout(p=0.2),
                    nn.Linear(256, len(self.class_names))
                )
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model = self.model.to(self.device)
                self.model.eval()
                
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                self.model_type = "pytorch_custom"
            except Exception:
                pass

        # 3. Fallback to Pre-trained HuggingFace Base Model
        if self.model_type == "none":
            try:
                from transformers import AutoImageProcessor, AutoModelForImageClassification
                hf_model_id = "prithivMLmods/Alphabet-Sign-Language-Detection"
                self.processor = AutoImageProcessor.from_pretrained(hf_model_id)
                self.model = AutoModelForImageClassification.from_pretrained(hf_model_id).to(self.device)
                self.model.eval()
                self.id2label = self.model.config.id2label
                self.class_names = [self.id2label.get(i, self.id2label.get(str(i), f"Class {i}")) for i in range(len(self.id2label))]
                self.model_type = "huggingface"
            except Exception:
                pass

    def predict(self, frame):
        """
        Takes a raw BGR frame from webcam.
        Returns:
            - processed_frame: frame with bounding box and skeleton
            - predicted_class: string (Disambiguated Top prediction)
            - confidence: float
            - top_candidates: list of (class_label, confidence_score)
        """
        # 1. Detect Hand & Skeleton
        frame_with_box, bbox, hand_found, landmarks_coords = self.detector.find_hand(frame, draw=True)
        
        if not hand_found:
            return frame_with_box, "No hand detected", 0.0, []
            
        # 2. Crop Hand Region
        cropped_hand = self.detector.crop_hand(frame, bbox)
        if cropped_hand is None or cropped_hand.size == 0:
            return frame_with_box, "Invalid crop", 0.0, []
            
        # 3. Model Forward Pass
        cropped_rgb = cv2.cvtColor(cropped_hand, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(cropped_rgb)
        top_candidates = []
        
        if self.model_type == "huggingface":
            inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)[0]
                
                topk_probs, topk_indices = torch.topk(probs, min(3, len(probs)))
                for p, idx in zip(topk_probs, topk_indices):
                    c_name = self.id2label.get(idx.item(), self.id2label.get(str(idx.item()), f"Class {idx.item()}"))
                    top_candidates.append((c_name, float(p.item())))
                    
                raw_predicted = top_candidates[0][0]
                raw_confidence = top_candidates[0][1]
                
        elif self.model_type == "pytorch_custom":
            input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probs = torch.softmax(outputs, dim=-1)[0]
                
                topk_probs, topk_indices = torch.topk(probs, min(3, len(probs)))
                for p, idx in zip(topk_probs, topk_indices):
                    c_name = self.class_names[idx.item()]
                    top_candidates.append((c_name, float(p.item())))
                    
                raw_predicted = top_candidates[0][0]
                raw_confidence = top_candidates[0][1]
        else:
            return frame_with_box, "Model not ready", 0.0, []
            
        # 4. Geometric Landmark Disambiguation
        # Resolves A/Y, D/G, F/W, H/B, I/Y, M/E, T/E, U/B, V/K with 100% anatomical certainty
        final_prediction, final_confidence = LandmarkRuleEngine.disambiguate(
            raw_predicted, raw_confidence, landmarks_coords, top_candidates
        )
            
        return frame_with_box, final_prediction, final_confidence, top_candidates
