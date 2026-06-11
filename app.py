from flask import Flask, render_template, request
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import base64

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet34()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

model.load_state_dict(torch.load('model_monyet_kera.pth', map_location=device))
model.to(device)
model.eval()

class_names = ['kera', 'monyet']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "Tidak ada file yang diunggah"
    
    file = request.files['file']
    if file.filename == '':
        return "Nama file kosong"
    
    if file:
        try:
            img_bytes = file.read()
            image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            
            encoded_img = base64.b64encode(img_bytes).decode('utf-8')
            
            tensor = transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(tensor)
                _, preds = torch.max(outputs, 1)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence = probabilities[0][preds[0]].item() * 100
            
            hasil_prediksi = class_names[preds[0]].upper()
            
            return render_template('result.html', 
                                 prediction=hasil_prediksi, 
                                 confidence=round(confidence, 2), 
                                 user_image=encoded_img)
                   
        except Exception as e:
            return f"Terjadi kesalahan: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)