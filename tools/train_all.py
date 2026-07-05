"""
train_all.py
One-shot training script: trains U-Net segmentation model first,
then trains EDCNN classification model in two phases.

Phase 1: Freeze backbones, train head (fast convergence)
Phase 2: Unfreeze all, fine-tune with low LR

Saves:
    results/edcnn_weights.keras
    results/unet_weights.h5
"""

import os
import sys
# Ensure the project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
)

# Suppress TF info/warning noise
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from utils.config import (
    TRAIN_DIR, VAL_DIR, RESULTS_DIR,
    EDCNN_WEIGHTS, UNET_WEIGHTS,
    BATCH_SIZE, EPOCHS_PHASE1, EPOCHS_PHASE2, LEARNING_RATE,
    IMG_HEIGHT, IMG_WIDTH, CHANNELS, NUM_CLASSES, CLASSES,
    DROPOUT_RATE
)
from utils.data_loader import load_classification_data, load_segmentation_data, apply_bm3d
from models.edcnn import build_edcnn, get_compiled_edcnn, unfreeze_top_layers
from models.unet import build_unet, get_compiled_unet

import cv2

os.makedirs(RESULTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# GPU MEMORY GROWTH (prevents OOM on single GPU)
# ─────────────────────────────────────────────────────────────────────────────
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"[GPU] Using {len(gpus)} GPU(s)")
else:
    print("[CPU] No GPU detected — training on CPU (slower)")

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_cls_data():
    print("\n[Data] Loading classification data...")
    print("  TRAIN:")
    X_train, y_train = load_classification_data(TRAIN_DIR)
    print("  VAL:")
    X_val,   y_val   = load_classification_data(VAL_DIR)
    print(f"\n  Train: {X_train.shape}  labels: {np.bincount(y_train)}")
    print(f"  Val:   {X_val.shape}    labels: {np.bincount(y_val)}")
    return (X_train, y_train), (X_val, y_val)


def load_seg_data():
    print("\n[Data] Loading segmentation data (benign + malignant only)...")
    print("  TRAIN:")
    X_tr, M_tr = load_segmentation_data(TRAIN_DIR, seg_classes=("benign", "malignant"))
    print("  VAL:")
    X_vl, M_vl = load_segmentation_data(VAL_DIR,   seg_classes=("benign", "malignant"))
    print(f"\n  Train imgs: {X_tr.shape}  masks: {M_tr.shape}")
    print(f"  Val   imgs: {X_vl.shape}  masks: {M_vl.shape}")
    return (X_tr, M_tr), (X_vl, M_vl)


# ─────────────────────────────────────────────────────────────────────────────
# AUGMENTATION (in-memory, no albumentations dependency)
# ─────────────────────────────────────────────────────────────────────────────

def augment_cls_batch(X, y, multiplier=3):
    """
    OncoVision AI Clinical Augmentation Pipeline.
    Synchronized with Table 1 parameters: Rotation (6), Shift (0.3), Shear (0.3), Zoom (0.3).
    """
    X_aug, y_aug = [], []
    
    # Parameters from Table 1
    rot_limit = 6
    shift_limit = 0.3
    shear_limit = 0.3
    zoom_limit = 0.3
    brightness_range = (0.8, 1.2)
    
    print(f"[Augment] Enriching dataset by {multiplier}x using Table 1 parameters...")
    
    for img, label in zip(X, y):
        # 1. Original
        X_aug.append(img); y_aug.append(label)
        
        h, w = img.shape[:2]
        
        for _ in range(multiplier):
            aug_img = img.copy()
            
            # 2. Horizontal Flip (clinically valid)
            if np.random.rand() > 0.5:
                aug_img = np.fliplr(aug_img)
            
            # 3. Random Rotation (6 degrees)
            angle = np.random.uniform(-rot_limit, rot_limit)
            M_rot = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            aug_img = cv2.warpAffine(aug_img, M_rot, (w, h), borderMode=cv2.BORDER_REPLICATE)
            
            # 4. Random Shift & Shear & Zoom
            tx = np.random.uniform(-shift_limit, shift_limit) * w
            ty = np.random.uniform(-shift_limit, shift_limit) * h
            shear = np.random.uniform(-shear_limit, shear_limit)
            zoom = np.random.uniform(1 - zoom_limit, 1 + zoom_limit)
            
            M_affine = np.float32([
                [zoom, shear, tx],
                [0, zoom, ty]
            ])
            aug_img = cv2.warpAffine(aug_img, M_affine, (w, h), borderMode=cv2.BORDER_REPLICATE)
            
            # 5. Random Brightness (±20% intensity shift)
            # Since images are Z-score normalized, we apply shift as an additive bias
            brightness_factor = np.random.uniform(*brightness_range)
            aug_img = aug_img * brightness_factor
            
            X_aug.append(aug_img); y_aug.append(label)
            
    return np.array(X_aug, dtype=np.float32), np.array(y_aug, dtype=np.int32)


def augment_seg_batch(X, M):
    """Flip augmentation that mirrors mask too."""
    X_aug, M_aug = [], []
    for img, mask in zip(X, M):
        X_aug.append(img);            M_aug.append(mask)
        X_aug.append(np.fliplr(img)); M_aug.append(np.fliplr(mask))
        X_aug.append(np.flipud(img)); M_aug.append(np.flipud(mask))
    return np.array(X_aug, np.float32), np.array(M_aug, np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 1. TRAIN U-NET
# ─────────────────────────────────────────────────────────────────────────────

def train_unet():
    print("\n" + "="*60)
    print("  PHASE 0: Training U-Net Segmentation Model")
    print("="*60)

    (X_tr, M_tr), (X_vl, M_vl) = load_seg_data()

    print("\n[Augment] Flipping segmentation data...")
    X_tr, M_tr = augment_seg_batch(X_tr, M_tr)
    print(f"  After augmentation: {X_tr.shape}")

    model = get_compiled_unet()
    model.summary(line_length=100)

    callbacks = [
        ModelCheckpoint(
            UNET_WEIGHTS,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1),
        CSVLogger(os.path.join(RESULTS_DIR, 'unet_training_log.csv'))
    ]

    history = model.fit(
        X_tr, M_tr,
        validation_data=(X_vl, M_vl),
        epochs=40,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    print(f"\n[U-Net] Best val_loss: {min(history.history['val_loss']):.4f}")
    print(f"[U-Net] Weights saved to: {UNET_WEIGHTS}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 2. TRAIN EDCNN — PHASE 1 (Frozen backbones)
# ─────────────────────────────────────────────────────────────────────────────

def train_edcnn_phase1(X_train, y_train, X_val, y_val):
    print("\n" + "="*60)
    print("  PHASE 1: Training EDCNN — Frozen Backbones")
    print("="*60)

    model = get_compiled_edcnn(freeze_base=True, learning_rate=LEARNING_RATE)
    model.summary(line_length=100)

    ckpt_phase1 = os.path.join(RESULTS_DIR, "edcnn_phase1_best.weights.h5")

    callbacks = [
        ModelCheckpoint(ckpt_phase1, monitor='val_accuracy', save_best_only=True,
                        save_weights_only=True, mode='max', verbose=1),
        EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True,
                      mode='max', verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1),
        CSVLogger(os.path.join(RESULTS_DIR, 'edcnn_phase1_log.csv'))
    ]

    # Phase 3: Malignant Priority Weights
    # Benign=1.0, Malignant=2.0 (Double priority), Normal=1.0
    class_weights = {0: 1.0, 1: 2.0, 2: 1.0}
    print(f"\n[Phase 1] Using clinical class weights: {class_weights}")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS_PHASE1,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    best_acc = max(history.history['val_accuracy'])
    print(f"\n[Phase 1] Best val_accuracy: {best_acc:.4f}")
    model.load_weights(ckpt_phase1)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 3. TRAIN EDCNN — PHASE 2 (Fine-tune all layers)
# ─────────────────────────────────────────────────────────────────────────────

def train_edcnn_phase2(model, X_train, y_train, X_val, y_val):
    print("\n" + "="*60)
    print("  PHASE 2: Fine-Tuning EDCNN — Unfreezing Backbones")
    print("="*60)

    # Unfreeze top layers of each backbone
    unfreeze_top_layers(model, num_layers=30)

    # Phase 3 Enhancement: Use Focal Loss to handle hard Malignant cases
    from models.edcnn import focal_loss
    
    # Recompile with lower LR and clinical Focal Loss
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE * 0.1),
        loss=focal_loss(gamma=2.0, alpha=0.5), # alpha=0.5 helps balance precision/recall
        metrics=["accuracy"]
    )

    callbacks = [
        ModelCheckpoint(
            EDCNN_WEIGHTS,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True,
                      mode='max', verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-7, verbose=1),
        CSVLogger(os.path.join(RESULTS_DIR, 'edcnn_phase2_log.csv'))
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS_PHASE2,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    best_acc = max(history.history['val_accuracy'])
    print(f"\n[Phase 2] Best val_accuracy: {best_acc:.4f}")
    print(f"[EDCNN] Final weights saved to: {EDCNN_WEIGHTS}")
    model.load_weights(EDCNN_WEIGHTS)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("================================================================")
    print("  ONCOVISION AI — FULL MODEL TRAINING PIPELINE")
    print("================================================================\n")
    print(f"  Classes    : {CLASSES}")
    print(f"  Image Size : {IMG_HEIGHT}x{IMG_WIDTH}x{CHANNELS}")
    print(f"  Results    : {RESULTS_DIR}")

    # ── 0. Train U-Net ────────────────────────────────────────────────────────
    train_unet()

    # ── 1. Load classification data + augment ─────────────────────────────────
    (X_train, y_train), (X_val, y_val) = load_cls_data()

    print("\n[Augment] Applying flip augmentation to classification data...")
    X_train, y_train = augment_cls_batch(X_train, y_train)
    print(f"  After augmentation: {X_train.shape}")

    # ── 2. Train EDCNN Phase 1 ────────────────────────────────────────────────
    model = train_edcnn_phase1(X_train, y_train, X_val, y_val)

    # ── 3. Train EDCNN Phase 2 ────────────────────────────────────────────────
    model = train_edcnn_phase2(model, X_train, y_train, X_val, y_val)

    print("\n" + "="*60)
    print("  TRAINING COMPLETE!")
    print("="*60)
    print(f"  EDCNN weights : {EDCNN_WEIGHTS}")
    print(f"  U-Net weights : {UNET_WEIGHTS}")
    print("\n  Run: python app.py  to start the diagnostic portal.\n")


if __name__ == "__main__":
    main()
