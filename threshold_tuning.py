

import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torchvision.models as models
from sklearn.metrics import classification_report, confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

test_data = datasets.ImageFolder("dataset/test", transform=transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=16)

class_names = test_data.classes  # alphabetical -> ['fake', 'real']
fake_idx = class_names.index("fake")
real_idx = class_names.index("real")
print("Classes:", class_names, "  (fake_idx =", fake_idx, ", real_idx =", real_idx, ")")

model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("model.pth", map_location=device))
model = model.to(device)
model.eval()

all_fake_probs = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        all_fake_probs.extend(probs[:, fake_idx].cpu().numpy())
        all_labels.extend(labels.numpy())

print("\nSweeping thresholds (predict 'fake' if fake_probability >= threshold)")
print("Default argmax behavior = threshold 0.50\n")

print(f"{'threshold':>10} {'fake_prec':>10} {'fake_recall':>12} {'fake_f1':>9} {'real_prec':>10} {'real_recall':>12}")

best_threshold = 0.5
best_f1 = 0.0

for threshold in [0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10]:
    preds = [fake_idx if p >= threshold else real_idx for p in all_fake_probs]
    report = classification_report(
        all_labels, preds, target_names=class_names,
        output_dict=True, zero_division=0
    )
    fake_prec = report["fake"]["precision"]
    fake_rec = report["fake"]["recall"]
    fake_f1 = report["fake"]["f1-score"]
    real_prec = report["real"]["precision"]
    real_rec = report["real"]["recall"]

    print(f"{threshold:>10.2f} {fake_prec:>10.3f} {fake_rec:>12.3f} {fake_f1:>9.3f} {real_prec:>10.3f} {real_rec:>12.3f}")

    if fake_f1 > best_f1:
        best_f1 = fake_f1
        best_threshold = threshold

print(f"\nBest threshold by fake-class F1: {best_threshold} (F1={best_f1:.3f})")

print("\n" + "=" * 60)
print(f"DETAILED REPORT AT THRESHOLD = {best_threshold}")
print("=" * 60)
final_preds = [fake_idx if p >= best_threshold else real_idx for p in all_fake_probs]
print(classification_report(all_labels, final_preds, target_names=class_names, digits=4))

cm = confusion_matrix(all_labels, final_preds)
print("CONFUSION MATRIX")
print(f"{'':>12}", "  ".join([f"{c:>10}" for c in class_names]), "  <- predicted")
for i, row in enumerate(cm):
    print(f"{class_names[i]:>12}", "  ".join([f"{v:>10}" for v in row]))

print(f"""
NOTE: To use this threshold in gradcam.py or your Streamlit app, replace:
    pred_class = output.argmax(dim=1).item()
with:
    probs = torch.softmax(output, dim=1)
    fake_prob = probs[0][{fake_idx}].item()
    pred_class = {fake_idx} if fake_prob >= {best_threshold} else {real_idx}
""")