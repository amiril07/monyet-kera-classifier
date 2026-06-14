import torch
import cv2
import numpy as np
from torchvision import models, transforms
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.resnet34()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load('model_monyet_kera.pth', map_location=device))
model.eval()

target_layers = [model.layer4[-1]]

image_path = 'dataset/val/monyet/vm9.png'
rgb_img = cv2.imread(image_path)[:, :, ::-1]
rgb_img = np.float32(rgb_img) / 255
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
input_tensor = transform(cv2.imread(image_path)[:, :, ::-1]).unsqueeze(0)

cam = GradCAM(model=model, target_layers=target_layers)
grayscale_cam = cam(input_tensor=input_tensor)[0, :]

rgb_img_resized = cv2.resize(rgb_img, (224, 224))
visualization = show_cam_on_image(rgb_img_resized, grayscale_cam, use_rgb=True)

output_img = Image.fromarray(visualization)
output_img.save('hasil_sorotan_ai.jpg')
print("Selesai! Silakan buka file 'hasil_sorotan_ai.jpg' untuk melihat fokus AI.")