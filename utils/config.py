"""
utils/config.py
Central configuration file for all model and data pipeline settings.
"""
import os

# ── Base Paths ────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR     = os.path.join(BASE_DIR, "dataset")
BM3D_DIR        = os.path.join(BASE_DIR, "dataset_bm3d")
RESULTS_DIR     = os.path.join(BASE_DIR, "results")
GRADCAM_DIR     = os.path.join(RESULTS_DIR, "gradcam")

# ── Dataset Splits ────────────────────────────────────────────────────────────
TRAIN_DIR       = os.path.join(DATASET_DIR, "train")
VAL_DIR         = os.path.join(DATASET_DIR, "val")
TEST_DIR        = os.path.join(DATASET_DIR, "test")

BM3D_TRAIN_DIR  = os.path.join(BM3D_DIR, "train")
BM3D_VAL_DIR    = os.path.join(BM3D_DIR, "val")
BM3D_TEST_DIR   = os.path.join(BM3D_DIR, "test")

# ── Model Weight Paths ────────────────────────────────────────────────────────
MODELS_DIR      = os.path.join(BASE_DIR, "models")
EDCNN_WEIGHTS   = os.path.join(RESULTS_DIR, "edcnn_weights.weights.h5")
UNET_WEIGHTS    = os.path.join(RESULTS_DIR, "unet_weights.weights.h5")

# ── Image Config ──────────────────────────────────────────────────────────────
IMG_HEIGHT      = 224
IMG_WIDTH       = 224
IMG_SIZE        = (IMG_HEIGHT, IMG_WIDTH)
CHANNELS        = 3

# ── Class Config ─────────────────────────────────────────────────────────────
# Order MUST match training label encoding
CLASSES         = ["benign", "malignant", "normal"]
NUM_CLASSES     = len(CLASSES)

# ── Training Hyperparameters ─────────────────────────────────────────────────
BATCH_SIZE      = 32
EPOCHS_PHASE1   = 15
EPOCHS_PHASE2   = 20
LEARNING_RATE   = 1e-4
DROPOUT_RATE    = 0.3
