import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import numpy as np
import cv2
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)

model.load_state_dict(torch.load("model.pth", map_location=device))
model = model.to(device)
model.eval()

img_path = "image.jpg"
print("File Exists:", os.path.exists(img_path))

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

image = Image.open(img_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)

gradients = []
activations = []

def backward_hook(module, grad_input, grad_output):
    gradients.append(grad_output[0])

def forward_hook(module, input, output):
    activations.append(output)

target_layer = model.layer4[1].conv2

target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)

output = model(input_tensor)
pred_class = output.argmax(dim=1).item()

probs = torch.softmax(output, dim=1)
confidence = probs[0][pred_class].item()

classes = ["Fake", "Real"]

print(f"Prediction: {classes[pred_class]} ({confidence*100:.2f}%)")

model.zero_grad()
output[0, pred_class].backward()

grads = gradients[0].cpu().data.numpy()[0]
acts = activations[0].cpu().data.numpy()[0]

weights = np.mean(grads, axis=(1, 2))

cam = np.zeros(acts.shape[1:], dtype=np.float32)

for i, w in enumerate(weights):
    cam += w * acts[i]

cam = np.maximum(cam, 0)
cam = cv2.resize(cam, (224, 224))
cam = cam - cam.min()
cam = cam / cam.max()

heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

img = cv2.imread(img_path)
img = cv2.resize(img, (224, 224))

superimposed = heatmap * 0.4 + img
superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)

cv2.imwrite("gradcam_output.jpg", superimposed)

print("Grad-CAM saved as gradcam_output.jpg")