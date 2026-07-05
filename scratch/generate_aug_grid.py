
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Paths
base_dir = r"d:\project\dataset\train\benign\images"
save_path = r"C:\Users\pranj\.gemini\antigravity\brain\6f063695-ef3f-46a5-9fa4-33fb46367548\augmentation_visualization.png"

# Files to load
files = {
    "Original": "benign (10).png",
    "Brightness": "benign (10)_bright.png",
    "H-Flip": "benign (10)_hflip.png",
    "Rotation": "benign (10)_rot.png",
    "V-Flip": "benign (10)_vflip.png",
    "Zoom": "benign (10)_zoom.png"
}

# Create plot
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle("Data Augmentation Techniques in OncoVision AI", fontsize=20, fontweight='bold', color='#0984e3')

for i, (label, filename) in enumerate(files.items()):
    ax = axes[i // 3, i % 3]
    fpath = os.path.join(base_dir, filename)
    img = cv2.imread(fpath)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
    else:
        ax.text(0.5, 0.5, f"Missing: {filename}", ha='center', va='center')
    ax.set_title(label, fontsize=14, fontweight='semibold')
    ax.axis('off')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Visualization saved to: {save_path}")
