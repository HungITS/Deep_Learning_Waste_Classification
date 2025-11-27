import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import sys
sys.path.append('..')
from src.model import ResNet50

NUM_CLASSES = 6
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
CLASS_COLORS = {
    'cardboard': (255, 165, 0),
    'glass': (0, 255, 255),
    'metal': (192, 192, 192),
    'paper': (255, 255, 255),
    'plastic': (255, 20, 147),
    'trash': (128, 128, 128)
}

model_path = os.environ.get('MODEL_PATH', 'model/latest.pth')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Loading model from: {model_path}")
model = ResNet50(num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def predict_frame(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    
    image_tensor = transform(pil_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    predicted_class = CLASS_NAMES[predicted.item()]
    confidence_score = confidence.item()
    
    return predicted_class, confidence_score

def draw_prediction(frame, predicted_class, confidence):
    height, width = frame.shape[:2]
    
    color = CLASS_COLORS.get(predicted_class, (255, 255, 255))
    
    cv2.rectangle(frame, (10, 10), (400, 100), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (400, 100), color, 2)
    
    text = f"Class: {predicted_class.upper()}"
    cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    conf_text = f"Confidence: {confidence:.2%}"
    cv2.putText(frame, conf_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    guide_text = "Press 'q' to quit, 's' to save screenshot"
    cv2.putText(frame, guide_text, (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame

if __name__ == "__main__":
    print("Starting camera demo")
    print("Press 'q' to quit, 's' to save screenshot")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Cannot open camera!")
        exit()
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    frame_count = 0
    screenshot_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Cannot read frame from camera!")
            break
        
        if frame_count % 10 == 0:
            try:
                predicted_class, confidence = predict_frame(frame)
            except Exception as e:
                print(f"Prediction error: {e}")
                predicted_class, confidence = "unknown", 0.0
        
        frame = draw_prediction(frame, predicted_class, confidence)
        
        cv2.imshow('Waste Classification - Camera Demo', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            screenshot_name = f"images/screenshot_{screenshot_count:03d}.jpg"
            cv2.imwrite(screenshot_name, frame)
            print(f"Screenshot saved: {screenshot_name}")
            screenshot_count += 1
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    print("Camera demo exited!")