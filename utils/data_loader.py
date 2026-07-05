"""
utils/data_loader.py
Loads images + masks for EDCNN classification and U-Net segmentation.
Applies BM3D denoising and Z-score normalization to match training pipeline.
"""

import os
import numpy as np
import cv2
from utils.config import (
    TRAIN_DIR, VAL_DIR, TEST_DIR,
    BM3D_TRAIN_DIR, BM3D_VAL_DIR, BM3D_TEST_DIR,
    CLASSES, NUM_CLASSES,
    IMG_HEIGHT, IMG_WIDTH, CHANNELS,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def apply_bm3d(img: np.ndarray) -> np.ndarray:
    """Apply BM3D denoising filter on BGR/RGB images."""
    try:
        import bm3d
        img_float = img.astype(np.float32) / 255.0
        denoised = bm3d.bm3d(img_float, sigma_psd=0.05)
        return np.clip(denoised * 255.0, 0, 255).astype(np.uint8)
    except Exception:
        # Fallback: use cv2 fast non-local means if bm3d fails
        print("[Warning] BM3D failed, falling back to cv2.fastNlMeans.")
        return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)


def _zscore_normalize(img: np.ndarray) -> np.ndarray:
    """Z-score normalise a float32 image array."""
    mean = img.mean()
    std  = img.std() + 1e-8
    return (img - mean) / std


def _load_image(path: str, mask_path: str = None, dim_factor: float = None,
                apply_denoising: bool = False) -> np.ndarray:
    """Read, resize, convert to RGB float32, apply mask dimming if given, and z-score normalise.
    
    Args:
        apply_denoising: Set True only at inference time. Training skips BM3D for speed.
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")

    if apply_denoising:
        img = apply_bm3d(img)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img.astype(np.float32)

    # Optional background dimming
    if dim_factor is not None and mask_path and os.path.exists(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            mask = cv2.resize(mask, (IMG_WIDTH, IMG_HEIGHT))
            mask = (mask > 127).astype(np.float32)[..., np.newaxis]
            dim_matrix = np.where(mask > 0.5, 1.0, dim_factor)
            img = img * dim_matrix

    img = _zscore_normalize(img)
    return img


def _load_mask(path: str) -> np.ndarray:
    """Read, resize, and binarise a mask to shape (H, W, 1)."""
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.float32)
    mask = cv2.resize(mask, (IMG_WIDTH, IMG_HEIGHT))
    mask = (mask > 127).astype(np.float32)
    return mask[..., np.newaxis]


# ── Classification dataset ────────────────────────────────────────────────────

def load_classification_data(split_dir: str, return_paths: bool = False, dim_factor: float = None,
                             apply_denoising: bool = False):
    """Load images from dataset split directory with optional background dimming."""
    X, y, paths = [], [], []
    for class_idx, class_name in enumerate(CLASSES):
        images_dir = os.path.join(split_dir, class_name, "images")
        masks_dir  = os.path.join(split_dir, class_name, "masks")
        if not os.path.isdir(images_dir):
            print(f"[WARN] Directory not found, skipping: {images_dir}")
            continue
        files = sorted([
            f for f in os.listdir(images_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
        ])
        print(f"  [{class_name}] {len(files)} images found")
        for fname in files:
            fpath = os.path.join(images_dir, fname)
            try:
                mask_path = None
                if dim_factor is not None:
                    mask_path = os.path.join(masks_dir, fname)
                    if not os.path.exists(mask_path):
                        stem = os.path.splitext(fname)[0]
                        candidates = [f for f in os.listdir(masks_dir) if stem in f] if os.path.exists(masks_dir) else []
                        mask_path = os.path.join(masks_dir, candidates[0]) if candidates else None

                img = _load_image(fpath, mask_path, dim_factor, apply_denoising=apply_denoising)
                X.append(img)
                y.append(class_idx)
                paths.append(fpath)
            except Exception as e:
                print(f"  [SKIP] {fpath}: {e}")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    return (X, y, paths) if return_paths else (X, y)


def get_all_classification_data(dim_factor: float = None):
    """Returns train / val / test splits as (X, y) tuples."""
    print("Loading TRAIN set ...")
    X_train, y_train = load_classification_data(TRAIN_DIR, dim_factor=dim_factor)
    print("Loading VAL set ...")
    X_val,   y_val   = load_classification_data(VAL_DIR, dim_factor=dim_factor)
    print("Loading TEST set ...")
    X_test,  y_test  = load_classification_data(TEST_DIR, dim_factor=dim_factor)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# ── BM3D dataset loader ───────────────────────────────────────────────────────

def load_bm3d_data(bm3d_split_dir: str, return_paths: bool = False):
    """Load BM3D-denoised images from a mirrored split directory."""
    if not os.path.isdir(bm3d_split_dir):
        raise FileNotFoundError(f"BM3D directory not found: {bm3d_split_dir}")

    X, y, paths = [], [], []
    for class_idx, class_name in enumerate(CLASSES):
        bm3d_images_dir = os.path.join(bm3d_split_dir, class_name, "images")
        if not os.path.isdir(bm3d_images_dir):
            print(f"[WARN] BM3D dir not found, skipping: {bm3d_images_dir}")
            continue
        files = sorted([
            f for f in os.listdir(bm3d_images_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
        ])
        print(f"  [{class_name}] {len(files)} BM3D images found")
        for fname in files:
            fpath = os.path.join(bm3d_images_dir, fname)
            try:
                img = _load_image(fpath)
                X.append(img)
                y.append(class_idx)
                paths.append(fpath)
            except Exception as e:
                print(f"  [SKIP] {fpath}: {e}")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    return (X, y, paths) if return_paths else (X, y)


# ── Segmentation dataset ──────────────────────────────────────────────────────

def load_segmentation_data(split_dir: str, seg_classes=("benign", "malignant"),
                           apply_denoising: bool = False):
    """Returns (X_imgs, X_masks) for U-Net training."""
    X_imgs, X_masks = [], []
    for class_name in seg_classes:
        images_dir = os.path.join(split_dir, class_name, "images")
        masks_dir  = os.path.join(split_dir, class_name, "masks")
        if not os.path.isdir(images_dir) or not os.path.isdir(masks_dir):
            print(f"[WARN] Skipping segmentation for: {class_name}")
            continue
        img_files = sorted([
            f for f in os.listdir(images_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
        ])
        print(f"  [{class_name}] segmentation: {len(img_files)} image-mask pairs")
        for fname in img_files:
            img_path  = os.path.join(images_dir, fname)
            mask_path = os.path.join(masks_dir,  fname)
            if not os.path.exists(mask_path):
                stem       = os.path.splitext(fname)[0]
                candidates = [f for f in os.listdir(masks_dir) if stem in f]
                if not candidates:
                    continue
                mask_path = os.path.join(masks_dir, candidates[0])
            try:
                img  = _load_image(img_path, apply_denoising=apply_denoising)
                mask = _load_mask(mask_path)
                X_imgs.append(img)
                X_masks.append(mask)
            except Exception as e:
                print(f"  [SKIP] {img_path}: {e}")

    X_imgs  = np.array(X_imgs,  dtype=np.float32)
    X_masks = np.array(X_masks, dtype=np.float32)
    return X_imgs, X_masks


def get_all_segmentation_data():
    """Returns train / val segmentation arrays."""
    print("Loading segmentation TRAIN set ...")
    X_tr, M_tr = load_segmentation_data(TRAIN_DIR)
    print("Loading segmentation VAL set ...")
    X_vl, M_vl = load_segmentation_data(VAL_DIR)
    return (X_tr, M_tr), (X_vl, M_vl)
