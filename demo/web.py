from flask import Flask, render_template, request, jsonify
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import io
import base64
import sys
sys.path.append('..')
from src.model import ResNet50

app = Flask(__name__, template_folder='../templates')


NUM_CLASSES = 6
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']


model_path = os.environ.get('MODEL_PATH', 'model/latest.pth')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNet50(num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def predict_image(image):
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
    predicted_class = CLASS_NAMES[predicted.item()]
    confidence_score = confidence.item()
    

    top3_prob, top3_idx = torch.topk(probabilities, 3)
    top3_results = []
    for i in range(3):
        class_name = CLASS_NAMES[top3_idx[0][i].item()]
        prob = top3_prob[0][i].item()
        top3_results.append({'class': class_name, 'probability': prob})
    
    return predicted_class, confidence_score, top3_results

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'})
        

        image = Image.open(file.stream).convert('RGB')
        

        predicted_class, confidence, top3 = predict_image(image)
        
        return jsonify({
            'success': True,
            'predicted_class': predicted_class,
            'confidence': f"{confidence:.2%}",
            'top3_predictions': top3
        })
        
    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)