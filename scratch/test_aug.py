import os
import sys
import numpy as np
import cv2

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.train_all import augment_cls_batch

def test_augmentation():
    print("Testing Clinical Augmentation Pipeline...")
    
    # Create a dummy image (224, 224, 3)
    dummy_img = np.zeros((224, 224, 3), dtype=np.float32)
    cv2.rectangle(dummy_img, (50, 50), (150, 150), (1, 1, 1), -1) # White box
    
    X = np.array([dummy_img])
    y = np.array([1]) # Malignant
    
    try:
        X_aug, y_aug = augment_cls_batch(X, y, multiplier=3)
        print(f"Success! Augmented batch shape: {X_aug.shape}")
        print(f"Labels shape: {y_aug.shape}")
        
        # Verify diversity
        for i in range(len(X_aug)):
            print(f" Image {i} mean: {X_aug[i].mean():.4f}")
            
        if len(X_aug) == 4: # 1 original + 3 augmented
            print("Verified: Correct number of copies generated.")
        else:
            print(f"Warning: Expected 4 copies, got {len(X_aug)}")
            
    except Exception as e:
        print(f"Error during augmentation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_augmentation()
