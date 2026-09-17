import os
import urllib.request
import cv2
import numpy as np

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index finger
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle finger
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring finger
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17)                                # Palm base
]

TASK_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
TASK_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "hand_landmarker.task")

class HandDetector:
    def __init__(self, static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5):
        """
        Initializes Hand Detection module with support for MediaPipe Tasks API and Legacy API.
        """
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.mode = "none"
        self.detector = None

        # 1. MediaPipe Tasks API
        try:
            import mediapipe as mp
            if hasattr(mp, 'tasks') and hasattr(mp.tasks, 'vision'):
                if not os.path.exists(TASK_MODEL_PATH):
                    os.makedirs(os.path.dirname(TASK_MODEL_PATH), exist_ok=True)
                    urllib.request.urlretrieve(TASK_MODEL_URL, TASK_MODEL_PATH)
                
                BaseOptions = mp.tasks.BaseOptions
                HandLandmarker = mp.tasks.vision.HandLandmarker
                HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
                VisionRunningMode = mp.tasks.vision.RunningMode

                options = HandLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=TASK_MODEL_PATH),
                    running_mode=VisionRunningMode.IMAGE,
                    num_hands=max_num_hands,
                    min_hand_detection_confidence=min_detection_confidence
                )
                self.detector = HandLandmarker.create_from_options(options)
                self.mp = mp
                self.mode = "tasks"
        except Exception:
            pass

        # 2. MediaPipe Legacy Solutions API
        if self.mode == "none":
            try:
                import mediapipe as mp
                if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'hands'):
                    self.mp_hands = mp.solutions.hands
                    self.detector = self.mp_hands.Hands(
                        static_image_mode=static_image_mode,
                        max_num_hands=max_num_hands,
                        min_detection_confidence=min_detection_confidence,
                        min_tracking_confidence=0.5
                    )
                    self.mode = "legacy"
            except Exception:
                pass
        
    def find_hand(self, img, draw=True):
        """
        Detects hand and computes an aspect-ratio preserved square bounding box.
        """
        if img is None or img.size == 0:
            return img, None, False

        h, w = img.shape[:2]
        bbox = None
        hand_found = False
        landmarks_coords = []

        if self.mode == "tasks":
            try:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=img_rgb)
                result = self.detector.detect(mp_image)
                if result.hand_landmarks and len(result.hand_landmarks) > 0:
                    hand_found = True
                    hand_lms = result.hand_landmarks[0]
                    for lm in hand_lms:
                        landmarks_coords.append((int(lm.x * w), int(lm.y * h)))
            except Exception:
                hand_found = False

        elif self.mode == "legacy":
            try:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = self.detector.process(img_rgb)
                if results.multi_hand_landmarks:
                    hand_found = True
                    hand_lms = results.multi_hand_landmarks[0]
                    for lm in hand_lms.landmark:
                        landmarks_coords.append((int(lm.x * w), int(lm.y * h)))
            except Exception:
                hand_found = False

        # If landmarks detected, calculate square bounding box to preserve aspect ratio
        if hand_found and landmarks_coords:
            xs = [pt[0] for pt in landmarks_coords]
            ys = [pt[1] for pt in landmarks_coords]
            
            raw_x_min, raw_x_max = min(xs), max(xs)
            raw_y_min, raw_y_max = min(ys), max(ys)
            
            box_w = raw_x_max - raw_x_min
            box_h = raw_y_max - raw_y_min
            center_x = (raw_x_min + raw_x_max) // 2
            center_y = (raw_y_min + raw_y_max) // 2
            
            # Make square box using the larger dimension plus generous padding (25%)
            side_len = int(max(box_w, box_h) * 1.35)
            half_side = side_len // 2
            
            x_min = max(0, center_x - half_side)
            y_min = max(0, center_y - half_side)
            x_max = min(w, center_x + half_side)
            y_max = min(h, center_y + half_side)
            
            bbox = (x_min, y_min, x_max, y_max)
            
            if draw:
                # Draw skeleton connections
                for p1, p2 in HAND_CONNECTIONS:
                    if p1 < len(landmarks_coords) and p2 < len(landmarks_coords):
                        cv2.line(img, landmarks_coords[p1], landmarks_coords[p2], (0, 220, 255), 2)
                # Draw landmark points
                for pt in landmarks_coords:
                    cv2.circle(img, pt, 4, (0, 255, 128), -1)
                # Draw bounding box
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (74, 222, 128), 2)

        return img, bbox, hand_found, landmarks_coords

    def crop_hand(self, img, bbox):
        """
        Extracts the square hand region from the image based on the bounding box.
        """
        if bbox is None:
            return None
            
        x_min, y_min, x_max, y_max = bbox
        cropped_img = img[y_min:y_max, x_min:x_max]
        
        if cropped_img is None or cropped_img.size == 0:
            return None
            
        return cropped_img
