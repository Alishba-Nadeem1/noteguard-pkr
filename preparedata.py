import os
import shutil
import random

real_path = "merged_currency_dataset/data/genuine_flat"
fake_path = "merged_currency_dataset/data/fake_flat"

output = "dataset"

for category, path in [("real", real_path), ("fake", fake_path)]:
    images = os.listdir(path)
    random.shuffle(images)

    split = int(0.8 * len(images))

    train_imgs = images[:split]
    test_imgs = images[split:]

    for img in train_imgs:
        os.makedirs(f"{output}/train/{category}", exist_ok=True)
        shutil.copy(os.path.join(path, img), f"{output}/train/{category}")

    for img in test_imgs:
        os.makedirs(f"{output}/test/{category}", exist_ok=True)
        shutil.copy(os.path.join(path, img), f"{output}/test/{category}")

print("Dataset ready ✅")