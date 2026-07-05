import os
import sys
import tensorflow as tf
sys.path.insert(0, os.getcwd())
from models.edcnn import build_edcnn
from models.unet import build_unet
from utils.config import RESULTS_DIR

def convert_to_keras():
    print("[Conversion] Converting Phase 3 weights to full models...")
    
    # 1. EDCNN
    print("  Loading EDCNN weights...")
    edcnn = build_edcnn()
    edcnn.load_weights("results/edcnn_weights.weights.h5")
    edcnn.save("results/edcnn_best.keras")
    print("  Saved results/edcnn_best.keras")
    
    # 2. U-Net
    print("  Loading U-Net weights...")
    unet = build_unet()
    unet.load_weights("results/unet_weights.weights.h5")
    unet.save("results/unet_best.keras")
    print("  Saved results/unet_best.keras")

if __name__ == "__main__":
    convert_to_keras()
