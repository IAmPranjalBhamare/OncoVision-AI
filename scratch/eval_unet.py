import os
import sys
sys.path.insert(0, os.getcwd())
import numpy as np
import tensorflow as tf
from utils.data_loader import load_segmentation_data
from utils.config import TEST_DIR, IMG_SIZE

def evaluate_unet():
    print("[Load] Model...")
    model = tf.keras.models.load_model("results/unet_best.keras", compile=False)
    
    # Load data
    print(f"[Data] Loading from {TEST_DIR}...")
    X_test, y_test = load_segmentation_data(TEST_DIR)
    
    print(f"[Eval] Testing on {len(X_test)} samples...")
    # Standard Dice / IoU metrics
    y_pred = model.predict(X_test, verbose=1)
    y_pred = (y_pred > 0.5).astype(np.float32)
    y_true = (y_test > 0.5).astype(np.float32)
    
    intersection = np.sum(y_true * y_pred)
    dice = (2. * intersection) / (np.sum(y_true) + np.sum(y_pred) + 1e-8)
    iou = intersection / (np.sum(y_true) + np.sum(y_pred) - intersection + 1e-8)
    
    print("\n" + "="*30)
    print(f"U-NET SEGMENTATION METRICS")
    print(f"Dice Coefficient: {dice:.4f}")
    print(f"Mean IoU (Jaccard): {iou:.4f}")
    print("="*30)

if __name__ == "__main__":
    evaluate_unet()
