import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# DEVICE
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# MODEL LOAD
# -----------------------------
from torchvision.models import resnet18

model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)

model.load_state_dict(torch.load("model.pth", map_location=device))
model = model.to(device)
model.eval()

# -----------------------------
# DATASET
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])

test_data = datasets.ImageFolder("dataset/test", transform=transform)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

class_names = test_data.classes

# -----------------------------
# PREDICTIONS
# -----------------------------
y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# -----------------------------
# CLASSIFICATION REPORT
# -----------------------------
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))

# -----------------------------
# CONFUSION MATRIX
# -----------------------------
cm = confusion_matrix(y_true, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=class_names,
            yticklabels=class_names)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.savefig("confusion_matrix.png")
plt.show()

print("Confusion matrix saved as confusion_matrix.png")