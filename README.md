# NoteGuard-PKR

ResNet18-based Pakistani currency authentication system with Grad-CAM explainability. The model classifies currency note images as genuine or counterfeit and highlights the specific regions of the note that influenced its decision.

## Overview

Counterfeit currency detection is typically limited to manual visual inspection. This project explores whether a deep learning model can learn to distinguish genuine Pakistani Rupee (PKR) notes from forged ones, while also providing a visual explanation for each prediction rather than a black-box output.

Since a public dataset of real counterfeit Pakistani currency does not exist, this project uses a synthetic forgery-generation approach: genuine note images are programmatically degraded using transformations that mimic real-world counterfeiting artifacts (blurred security threads, color-shifted ink, print-quality noise, misaligned print layers, and paper-texture irregularities). This approach follows the same methodology used in prior academic work on this problem (e.g. DeepMoney, Ali et al., PeerJ Computer Science 2019), which similarly had to construct its own dataset due to the lack of public counterfeit currency data.

## Features

- Binary classification (genuine vs counterfeit) using a fine-tuned ResNet18
- Grad-CAM heatmap visualization showing which regions of the note drove the prediction
- Tuned decision threshold to balance precision and recall on the counterfeit class
- Streamlit web application for interactive testing
- Class-weighted training to handle dataset imbalance

## Architecture

- Backbone: ResNet18, pretrained on ImageNet
- Fine-tuning strategy: the final residual block (`layer4`) and classification head are unfrozen; earlier layers remain frozen
- Loss: class-weighted cross-entropy
- Optimizer: Adam with differential learning rates (lower rate for the backbone, higher rate for the classification head)
- Input size: 224x224, normalized with mean/std of 0.5 across all channels

## Dataset

The dataset consists of:
- Genuine note images across all standard PKR denominations (10, 20, 50, 100, 500, 1000, 5000), front and back
- Synthetically generated counterfeit variants, produced by applying randomized combinations of the following degradations to genuine images:
  - Localized blur (simulating security thread / watermark reproduction failure)
  - Hue and saturation shift (simulating ink formulation mismatch)
  - Print-artifact noise (moire patterns and halftone-style noise)
  - Color-channel misalignment (simulating printing press registration error)
  - Paper texture noise
  - Partial security-feature occlusion
  - Geometric warping
  - Variable JPEG recompression

The full dataset is not included in this repository due to its size (over 25MB combined across genuine and synthetic images). If you would like access to the dataset, please open an issue or reach out directly and it can be shared.

## Setup

```bash
pip install -r requirements.txt
```

Requirements include: `torch`, `torchvision`, `opencv-python-headless`, `pillow`, `streamlit`, `scikit-learn`.

## Usage

### Prepare the dataset
```bash
python preparedata.py
```
This splits the genuine and counterfeit images into `dataset/train` and `dataset/test` folders.

### Train the model
```bash
python train.py
```
Trains the model and saves weights to `model.pth`.

### Evaluate
```bash
python evaluate_report.py
```
Prints a full classification report (precision, recall, F1-score) and confusion matrix.

### Tune the decision threshold
```bash
python threshold_tuning.py
```
Sweeps decision thresholds on the counterfeit-class probability and reports the best operating point.

### Run the web app
```bash
streamlit run app.py
```
Upload a note image to get a prediction, confidence score, and Grad-CAM heatmap.

## Results

Evaluated on a held-out test set, with a tuned decision threshold:

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| Fake | 0.976 | 0.950 | 0.963 |
| Real | 0.951 | 0.976 | 0.964 |

Overall accuracy: 96.3%

## Explainability

Grad-CAM is applied to the last convolutional block of the ResNet18 backbone to generate a heatmap over the input image, indicating which regions most influenced the model's prediction. This is displayed alongside the prediction in the Streamlit app.

## Limitations

- The model is trained primarily on synthetically-generated forgery patterns. It generalizes well to degradations similar to those in its training distribution, but may not generalize to arbitrary real-world counterfeiting techniques not represented in the synthetic data.
- No public dataset of real counterfeit Pakistani currency was available for training or validation.
- This is a student/portfolio project and is not intended for use as an actual financial authentication tool.

## Future Work

- Incorporate real counterfeit currency samples if a reliable source becomes available
- Explore GAN-based synthetic forgery generation for more realistic degradation patterns
- Expand to additional denominations and note conditions (worn, torn, folded)
- Attention-based (Vision Transformer) variant for comparison against the CNN-based approach

## Disclaimer

This project is for educational and portfolio purposes only. It should not be relied upon for actual currency authentication.
