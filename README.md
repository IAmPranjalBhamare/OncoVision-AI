# Breast Cancer Detection — EDCNN + U-Net + Grad-CAM

Based on: *Enhancing breast cancer segmentation and classification: An Ensemble Deep CNN and U-Net approach on ultrasound images* — Islam et al., 2024

---

## Project structure

```
breast_cancer_project/
├── dataset/
│   ├── train/
│   │   ├── benign/   images/  masks/
│   │   ├── malignant/images/  masks/
│   │   └── normal/   images/
│   ├── val/          (same structure)
│   └── test/         (same structure)
├── models/
│   ├── __init__.py
│   ├── edcnn.py          ← EDCNN hybrid model
│   └── unet.py           ← U-Net segmentation model
├── utils/
│   ├── __init__.py
│   ├── config.py         ← all paths & hyperparameters
│   ├── data_loader.py    ← image + mask loading
│   └── plots.py          ← all visualisation helpers
├── train_unet.py         ← Step 1: train segmentation
├── train_edcnn.py        ← Step 2: train classification
├── evaluate.py           ← Step 3: evaluate on test set
├── gradcam.py            ← Step 4: Grad-CAM XAI
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Run order

### Step 1 — Train U-Net (segmentation)
```bash
python train_unet.py
```
- Trains on `benign` + `malignant` images with their masks
- Saves best model to `models/unet_best.keras`
- Saves training curves and segmentation samples to `results/`

### Step 2 — Train EDCNN (classification)
```bash
python train_edcnn.py
```
- Phase 1: freezes MobileNet + Xception, trains head only (50 epochs)
- Phase 2: unfreezes last 30 layers, fine-tunes (50 epochs)
- Saves best model to `models/edcnn_best.keras`
- Saves accuracy/loss curves, confusion matrix, AUC-ROC

### Step 3 — Evaluate
```bash
python evaluate.py
```
- Loads best EDCNN weights
- Prints full classification report (Precision, Recall, F1)
- Saves confusion matrix and ROC curves to `results/plots/`

### Step 4 — Grad-CAM XAI
```bash
python gradcam.py
```
- Generates heatmap + overlay for 3 samples per class
- Saves to `results/gradcam/gradcam_benign.png` etc.

---

## Key hyperparameters (edit `utils/config.py`)

| Parameter     | Value                         |
|---------------|-------------------------------|
| Image size    | 224 × 224 × 3                 |
| Batch size    | 32 (EDCNN) / 8 (U-Net)        |
| Epochs        | 100 (EDCNN) / 15 (U-Net)      |
| Optimizer     | Adam lr=0.001                 |
| Dropout       | 0.4                           |
| Loss (EDCNN)  | SparseCategoricalCrossentropy |
| Loss (U-Net)  | BCE + Dice                    |
| Classes       | benign, malignant, normal     |

---

## Expected results (from paper)

| Model     | Dataset 1 Accuracy | AUC   |
|-----------|-------------------|-------|
| EDCNN     | **87.82%**        | 0.91  |
| VGG-16    | 78.75%            | —     |
| DenseNet  | 77.50%            | —     |
| AlexNet   | 76.25%            | —     |
