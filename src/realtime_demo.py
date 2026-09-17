import cv2
import sys
import os

# Ensure absolute imports work when run as script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.inference import ASLPredictor

def main():
    print("Initializing ASL Predictor...")
    predictor = ASLPredictor()
    
    print("Starting webcam... (Press 'q' to quit)")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
        
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        # Flip the frame horizontally for a more natural mirror view
        frame = cv2.flip(frame, 1)
        
        # Convert BGR (OpenCV) to RGB (Model)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # We can either predict the whole frame, or we can define a Region of Interest (ROI).
        # For simplicity, let's predict the whole frame resized by the model.
        # Alternatively, drawing a box and cropping it would be better.
        
        # Let's define a fixed ROI box in the center of the screen
        height, width, _ = frame.shape
        box_size = 224
        x1 = width // 2 - box_size // 2
        y1 = height // 2 - box_size // 2
        x2 = x1 + box_size
        y2 = y1 + box_size
        
        # Extract ROI
        roi = rgb_frame[y1:y2, x1:x2]
        
        # Predict on ROI
        predicted_class, confidence = predictor.predict_smoothed(roi)
        
        # Draw ROI Box on original frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw Prediction
        text = f"{predicted_class} ({confidence:.2f})"
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        cv2.imshow('ASL Real-Time Translation', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
