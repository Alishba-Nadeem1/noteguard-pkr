import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torchvision.models as models

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# same transforms (IMPORTANT)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])

# dataset
test_data = datasets.ImageFolder("dataset/test", transform=transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=16)

# ✅ SAME ResNet model
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)

model = model.to(device)

# load weights
model.load_state_dict(torch.load("model.pth", map_location=device))

model.eval()

# testing
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Accuracy: {accuracy:.2f}%")