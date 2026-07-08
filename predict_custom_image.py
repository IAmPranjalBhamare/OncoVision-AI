"""
predict_custom_image.py
Load a custom image and save predictions with:
1. Original Image
2. Segmented Image (U-Net)
3. Classification Result (EDCNN)
4. Grad-CAM Visualization (XAI)

Usage:
    python predict_custom_image.py --image /path/to/your/image.png [--output /path/to/save]
    
Example:
    python predict_custom_image.py --image dataset/test/benign/images/image_001.png
    python predict_custom_image.py --image my_image.jpg --output results/my_predictions
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import argparse
import cv2
import tensorflow as tf
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from tensorflow.keras.preprocessing import image as keras_img
from PIL import Image

from utils.config import IMG_SIZE, CLASSES, EDCNN_WEIGHTS, UNET_WEIGHTS
from utils.data_loader import _load_image
from models.edcnn import get_compiled_edcnn
from models.unet import build_unet
from utils.calibration import calibrate_confidence


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def load_image(image_path):
    """Load and preprocess image for model inference"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Load for display (PIL)
    img_pil = Image.open(image_path).convert('RGB')
    
    # Load for model with BM3D denoising (inference-time only)
    img_array = _load_image(image_path, apply_denoising=True)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array, img_pil


def predict_classification(model_edcnn, img_array):
    """Predict class using EDCNN"""
    prediction = model_edcnn(img_array, training=False).numpy()
    
    # Get raw probabilities
    raw_probs = {CLASSES[i]: prediction[0][i]*100 for i in range(len(CLASSES))}
    
    # Apply clinical calibration
    calibrated_probs, pred_class, confidence = calibrate_confidence(raw_probs, temperature=1.5, max_confidence=99.93)
    pred_idx = CLASSES.index(pred_class)
    
    return {
        'class': pred_class,
        'class_idx': pred_idx,
        'confidence': confidence,
        'probabilities': calibrated_probs,
        'raw_probabilities': raw_probs
    }


def predict_segmentation(model_unet, img_array):
    """Predict segmentation mask using U-Net"""
    mask_pred = model_unet(img_array, training=False).numpy()[0, ..., 0]
    mask_pred = (mask_pred > 0.5).astype(np.uint8) * 255
    return mask_pred


def get_saliency_heatmap(model_edcnn, img_array, class_idx, mask_pred=None, original_img=None):
    """
    Standard Base Grad-CAM.
    Returns the pure model attention mapping without any artificial masking.
    """
    from gradcam_xai import get_gradcam
    import sys, io

    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        heatmap = get_gradcam(model_edcnn, img_array, class_idx)
    finally:
        sys.stdout = old_stdout

    return heatmap.astype(np.float32)





# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION & SAVING
# ═══════════════════════════════════════════════════════════════════════════════

def save_original_image(img_pil, output_dir):
    """Save original image"""
    output_path = os.path.join(output_dir, "01_original.png")
    img_pil.save(output_path)
    print(f"[OK] Saved: {output_path}")
    return output_path


def save_segmented_image(img_pil, mask_pred, output_dir):
    """Save segmented image and overlay"""
    mask_display = mask_pred.squeeze() if mask_pred.ndim == 3 else mask_pred
    
    # Segmentation only
    seg_path = os.path.join(output_dir, "02_segmented.png")
    cv2.imwrite(seg_path, mask_display)
    print(f"[OK] Saved: {seg_path}")
    
    # Overlay (original + segmentation)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].imshow(img_pil)
    axes[0].set_title("Original Image", fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(img_pil, alpha=0.6)
    axes[1].imshow(mask_display, cmap='Reds', alpha=0.4, interpolation='nearest')
    axes[1].set_title("Segmentation Overlay", fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    overlay_path = os.path.join(output_dir, "02_segmentation_overlay.png")
    plt.tight_layout()
    plt.savefig(overlay_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {overlay_path}")
    
    return seg_path, overlay_path


def save_classification_result(img_pil, classification_result, output_dir):
    """Save classification result with probabilities"""
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    
    # Image
    ax[0].imshow(img_pil)
    ax[0].set_title("Input Image", fontsize=12, fontweight='bold')
    ax[0].axis('off')
    
    # Prediction results
    ax[1].axis('off')
    
    # Title
    title_text = f"Classification Result\nPredicted: {classification_result['class'].upper()}"
    ax[1].text(0.5, 0.95, title_text, 
              ha='center', va='top', fontsize=14, fontweight='bold',
              transform=ax[1].transAxes)
    
    # Confidence score
    conf_text = f"Confidence: {classification_result['confidence']:.2f}%"
    ax[1].text(0.5, 0.80, conf_text,
              ha='center', va='top', fontsize=12, color='green', fontweight='bold',
              transform=ax[1].transAxes)
    
    # All probabilities
    prob_text = "Class Probabilities:\n" + "─" * 30 + "\n"
    for cls, prob in classification_result['probabilities'].items():
        bar_length = int(prob / 5)
        bar = '█' * bar_length
        prob_text += f"{cls:12s} : {prob:6.2f}% {bar}\n"
    
    ax[1].text(0.5, 0.65, prob_text,
              ha='center', va='top', fontsize=11, family='monospace',
              transform=ax[1].transAxes,
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    cls_path = os.path.join(output_dir, "03_classification.png")
    plt.tight_layout()
    plt.savefig(cls_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {cls_path}")
    
    return cls_path


def save_gradcam_visualization(img_pil, heatmap, classification_result, mask_pred, output_dir):
    """
    Save Grad-CAM visualization as a 2x3 layout:
      Top row:    (a) Original Scan  | (b) Heatmap Only | (c) Clinical Overlay
      Bottom row: (d) Diagnostic Focus | (e) Classification Contribution Rank | (f) Segmentation Contribution Rank
    """
    from gradcam_xai import (
        overlay_heatmap as _overlay_heatmap,
        draw_tumor_bbox,
        create_ranked_overlay,
        create_segmentation_contribution_overlay,
        _CONTRIBUTION_TIERS,
    )

    img_u8 = np.array(img_pil.convert("RGB"))

    # (b) Pure Heatmap (JET colormap)
    h8 = (heatmap * 255).astype(np.uint8)
    pure_heatmap = cv2.applyColorMap(h8, cv2.COLORMAP_JET)
    pure_heatmap = cv2.cvtColor(pure_heatmap, cv2.COLOR_BGR2RGB)

    # (c) Clinical Overlay
    gradcam_over = _overlay_heatmap(img_u8, heatmap, alpha=0.6, threshold=0.15)

    # (d) Tumor Focus
    tumor_focus = draw_tumor_bbox(gradcam_over, heatmap, threshold=0.5,
                                  color=(0, 255, 80), thickness=2)

    # (e) Classification Contribution Rank — full-image, not mask-confined
    cls_rank = create_ranked_overlay(img_u8, heatmap, mask_pred=None)

    # (f) Segmentation Contribution Rank
    seg_mask_for_vis = mask_pred if mask_pred is not None \
                       else np.zeros(img_u8.shape[:2], dtype=np.uint8)
    seg_rank = create_segmentation_contribution_overlay(
        img_u8, seg_mask_for_vis, heatmap=heatmap
    )

    # ── 2×3 layout ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(36, 14), facecolor="#0d0d1a")
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.05, hspace=0.18,
                           left=0.02, right=0.98, top=0.88, bottom=0.05)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]

    label_font = {'fontsize': 12, 'fontweight': 'bold', 'color': 'white'}
    labels = [
        "(a) Original Scan",
        "(b) Heatmap (JET)",
        "(c) Clinical Overlay",
        "(d) Diagnostic Focus",
        "(e) Classification\nContribution Rank",
        "(f) Segmentation\nContribution Rank",
    ]
    panels = [img_u8, pure_heatmap, gradcam_over, tumor_focus, cls_rank, seg_rank]

    for i, (ax, panel, lbl) in enumerate(zip(axes, panels, labels)):
        ax.set_facecolor("#0d0d1a")
        ax.imshow(panel)
        ax.set_title(lbl, **label_font, pad=10)
        ax.axis("off")

    # ── AI result badge on panel (d) ─────────────────────────────────────────
    pred_cls  = classification_result['class'].upper()
    conf      = classification_result['confidence']
    color_hdr = "#00d26a" if "BENIGN" in pred_cls else "#ff4757" if "MALIGNANT" in pred_cls else "#0984e3"
    axes[3].text(0.5, 0.04, f"AI: {pred_cls}  ({conf:.1f}%)",
                ha="center", va="bottom", fontsize=11, fontweight="bold",
                color=color_hdr, transform=axes[3].transAxes,
                bbox=dict(facecolor='black', alpha=0.65, boxstyle='round,pad=0.4'))

    # ── Tier legend on panels (e) and (f) ────────────────────────────────────
    for ax_idx in (4, 5):
        legend_y = 0.03
        for lbl_t, pfl, clr, _ in reversed(_CONTRIBUTION_TIERS):
            hex_color = f"#{clr[0]:02x}{clr[1]:02x}{clr[2]:02x}"
            axes[ax_idx].text(
                0.02, legend_y, f"\u25cf {lbl_t}",
                transform=axes[ax_idx].transAxes,
                fontsize=8, color=hex_color, va="bottom",
                bbox=dict(boxstyle="round,pad=0.25", fc="black", alpha=0.6)
            )
            legend_y += 0.085

    fig.suptitle(
        "ONCOVISION AI — EXPLAINABLE DIAGNOSTIC OUTPUT  |  Contribution-Ranked XAI",
        fontsize=15, fontweight="bold", color="#00d26a", y=0.97
    )

    gradcam_path = os.path.join(output_dir, "04_gradcam_clinical_report.png")
    plt.savefig(gradcam_path, dpi=180, bbox_inches="tight", facecolor="#0d0d1a")
    plt.close()
    print(f"[OK] Saved unified visualization: {gradcam_path}")
    return gradcam_path



def save_final_summary(image_path, classification_result, output_dir):
    """Save text summary"""
    summary_path = os.path.join(output_dir, "RESULTS.txt")
    
    with open(summary_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("BREAST CANCER DETECTION - PREDICTION RESULTS\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Input Image: {os.path.basename(image_path)}\n")
        f.write(f"Full Path: {os.path.abspath(image_path)}\n\n")
        
        f.write("CLASSIFICATION RESULT\n")
        f.write("-" * 70 + "\n")
        f.write(f"Predicted Class: {classification_result['class'].upper()}\n")
        f.write(f"Confidence: {classification_result['confidence']:.2f}%\n\n")
        
        f.write("Class Probabilities:\n")
        for cls, prob in sorted(classification_result['probabilities'].items(), 
                               key=lambda x: x[1], reverse=True):
            f.write(f"  {cls:15s} : {prob:7.2f}%\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("OUTPUT FILES\n")
        f.write("=" * 70 + "\n")
        f.write("1. 01_original.png              -> Original input image\n")
        f.write("2. 02_segmented.png            -> Binary segmentation mask\n")
        f.write("3. 02_segmentation_overlay.png -> Segmentation on original\n")
        f.write("4. 03_classification.png       -> Classification with probabilities\n")
        f.write("5. 04_gradcam.png              -> Grad-CAM explainability\n")
        f.write("6. RESULTS.txt                 -> This summary file\n")
        f.write("=" * 70 + "\n")
    
    print(f"[OK] Saved: {summary_path}\n")
    return summary_path


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Predict and visualize custom image')
    parser.add_argument('--image', type=str, required=True, help='Path to image file')
    parser.add_argument('--output', type=str, default=None, 
                       help='Output directory (default: results/custom_predictions)')
    
    args = parser.parse_args()
    
    image_path = args.image
    
    # Set output directory
    if args.output:
        output_dir = args.output
    else:
        # Create timestamped folder
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("results", f"prediction_{timestamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("  CUSTOM IMAGE PREDICTION")
    print("=" * 70)
    print(f"\nImage: {image_path}")
    print(f"Output Directory: {os.path.abspath(output_dir)}\n")
    
    # ────────────────────────────────────────────────────────────────────────
    # LOAD IMAGE
    # ────────────────────────────────────────────────────────────────────────
    
    print("[1/6] Loading image...")
    try:
        img_array, img_pil = load_image(image_path)
        print(f"[OK] Image loaded: {img_pil.size}")
    except Exception as e:
        print(f"[ERROR] Error loading image: {e}")
        return
    
    # ────────────────────────────────────────────────────────────────────────
    # LOAD MODELS
    # ────────────────────────────────────────────────────────────────────────
    
    print("\n[2/6] Loading models...")
    
    edcnn_keras_path = os.path.join(os.path.dirname(EDCNN_WEIGHTS), "edcnn_best.h5")
    unet_keras_path = os.path.join(os.path.dirname(UNET_WEIGHTS), "unet_best.h5")

    if not os.path.exists(edcnn_keras_path):
        print(f"[ERROR] EDCNN model not found: {edcnn_keras_path}")
        return
    
    if not os.path.exists(unet_keras_path):
        print(f"[ERROR] U-Net model not found: {unet_keras_path}")
        return
    
    try:
        model_edcnn = tf.keras.models.load_model(edcnn_keras_path, compile=False)
        print("[OK] EDCNN model loaded")
        
        model_unet = tf.keras.models.load_model(unet_keras_path, compile=False)
        print("[OK] U-Net model loaded")
    except Exception as e:
        print(f"[ERROR] Error loading models: {e}")
        return
    
    # ────────────────────────────────────────────────────────────────────────
    # SAVE ORIGINAL IMAGE
    # ────────────────────────────────────────────────────────────────────────
    
    print("\n[3/6] Saving original image...")
    save_original_image(img_pil, output_dir)
    
    # ────────────────────────────────────────────────────────────────────────
    # SEGMENTATION (U-Net) FIRST to dim background
    # ────────────────────────────────────────────────────────────────────────
    
    print("\n[4/6] Running segmentation (U-Net)...")
    mask_pred = predict_segmentation(model_unet, img_array)
    save_segmented_image(img_pil, mask_pred, output_dir)

    # ────────────────────────────────────────────────────────────────────────
    # CLASSIFICATION (EDCNN) ON RAW IMAGE
    # ────────────────────────────────────────────────────────────────────────
    
    print("\n[5/6] Running classification (EDCNN)...")
    classification_result = predict_classification(model_edcnn, img_array)
    save_classification_result(img_pil, classification_result, output_dir)
    
    print(f"\n  Predicted: {classification_result['class'].upper()}")
    print(f"  Confidence: {classification_result['confidence']:.2f}%")
    
    # ────────────────────────────────────────────────────────────────────────
    # GRAD-CAM (EXPLAINABILITY)
    # ────────────────────────────────────────────────────────────────────────
    
    print("\n[6/6] Generating Grad-CAM visualization...")
    heatmap = get_saliency_heatmap(model_edcnn, img_array, classification_result['class_idx'], mask_pred=mask_pred)
    save_gradcam_visualization(img_pil, heatmap, classification_result, mask_pred, output_dir)
    
    # ────────────────────────────────────────────────────────────────────────
    # SUMMARY
    # ────────────────────────────────────────────────────────────────────────
    
    save_final_summary(image_path, classification_result, output_dir)
    
    print("\n" + "=" * 70)
    print("[OK] COMPLETE!")
    print("=" * 70)
    print(f"\nAll results saved to: {os.path.abspath(output_dir)}")
    print("\nFiles generated:")
    print("  1. 01_original.png              - Original image")
    print("  2. 02_segmented.png            - Segmentation mask")
    print("  3. 02_segmentation_overlay.png - Overlay visualization")
    print("  4. 03_classification.png       - Classification result")
    print("  5. 04_gradcam.png              - Grad-CAM explanation")
    print("  6. RESULTS.txt                 - Summary report")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
