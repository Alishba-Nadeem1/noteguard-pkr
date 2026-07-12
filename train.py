import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.models as models

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# dataset load
train_data = datasets.ImageFolder("dataset/train", transform=transform)
test_data = datasets.ImageFolder("dataset/test", transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=16)

print("Classes:", train_data.classes)
print("Mapping:", train_data.class_to_idx)

# Compute class weights to handle imbalance (fake ~2x real in v2 dataset)
class_counts = [0] * len(train_data.classes)
for _, label in train_data.samples:
    class_counts[label] += 1
print("Class counts (train):", dict(zip(train_data.classes, class_counts)))

total = sum(class_counts)
class_weights = [total / (len(class_counts) * c) for c in class_counts]
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
print("Class weights:", dict(zip(train_data.classes, class_weights)))

# ResNet model
model = models.resnet18(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

# unfreeze layer4 for forgery-specific feature learning
for param in model.layer4.parameters():
    param.requires_grad = True

model.fc = nn.Linear(model.fc.in_features, 2)
for param in model.fc.parameters():
    param.requires_grad = True

model = model.to(device)

# weighted loss so majority class (fake, since we have 2x variants)
# doesn't dominate training
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

optimizer = optim.Adam([
    {"params": model.layer4.parameters(), "lr": 1e-5},
    {"params": model.fc.parameters(), "lr": 1e-4},
])


def evaluate_per_class(model, loader, class_names):
    model.eval()
    correct_per_class = {c: 0 for c in class_names}
    total_per_class = {c: 0 for c in class_names}
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            for label, pred in zip(labels, predicted):
                cname = class_names[label.item()]
                total_per_class[cname] += 1
                if label == pred:
                    correct_per_class[cname] += 1
    model.train()
    accs = {c: (correct_per_class[c] / total_per_class[c] if total_per_class[c] else 0)
            for c in class_names}
    return accs


epochs = 10
class_names = train_data.classes

for epoch in range(epochs):
    model.train()
    running_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    accs = evaluate_per_class(model, test_loader, class_names)
    acc_str = "  ".join([f"{c}_acc={accs[c]*100:.1f}%" for c in class_names])
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss:.4f}  {acc_str}")

torch.save(model.state_dict(), "model.pth")
print("Training complete")