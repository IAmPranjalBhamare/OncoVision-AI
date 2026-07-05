import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
sys.path.insert(0, os.getcwd())
from utils.data_loader import load_classification_data, load_segmentation_data
from utils.config import TEST_DIR, IMG_SIZE, CLASSES

def final_eval():
    print("\n" + "="*30)
    print("  PHASE 3 FINAL CLINICAL EVALUATION")
    print("="*30)
    
    # 1. Classification Evaluation
    print("\n[EDCNN] Loading model and test data...")
    edcnn = tf.keras.models.load_model("results/edcnn_best.keras", compile=False)
    X_test, y_test = load_classification_data(TEST_DIR)
    
    print("[EDCNN] Predicting...")
    y_pred_probs = edcnn.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    report = classification_report(y_test, y_pred, target_names=CLASSES)
    print("\nClassification Report:")
    print(report)
    
    with open("results/classification_report_phase3.txt", "w") as f:
        f.write(report)
        
    # 2. Segmentation Evaluation
    print("\n[U-Net] Loading model and test data...")
    unet = tf.keras.models.load_model("results/unet_best.keras", compile=False)
    X_seg, Y_seg = load_segmentation_data(TEST_DIR)
    
    print("[U-Net] Predicting...")
    m_pred = unet.predict(X_seg, verbose=0)
    m_pred = (m_pred > 0.5).astype(np.float32)
    m_true = (Y_seg > 0.5).astype(np.float32)
    
    intersection = np.sum(m_true * m_pred)
    dice = (2. * intersection) / (np.sum(m_true) + np.sum(m_pred) + 1e-8)
    iou = intersection / (np.sum(m_true) + np.sum(m_pred) - intersection + 1e-8)
    
    print(f"\nU-Net Metrics:")
    print(f"  Dice Coefficient: {dice:.4f}")
    print(f"  Mean IoU:         {iou:.4f}")
    
    with open("results/segmentation_report_phase3.txt", "w") as f:
        f.write(f"Dice: {dice:.4f}\nIoU: {iou:.4f}")

if __name__ == "__main__":
    final_eval()
