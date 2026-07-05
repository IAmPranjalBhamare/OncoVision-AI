"""
generate_20_report.py
---------------------
Generates a comprehensive diagnostic report for 20 sampled test images.

Each sample produces one full-page figure with 5 panels:
  1. Original Ultrasound Image
  2. U-Net Predicted Segmentation Mask
  3. Segmentation Overlay (mask on image)
  4. Grad-CAM Heatmap (Jet colormap, research-paper style)
  5. Grad-CAM Overlay (heatmap blended on original)

Also generates a final summary validation page showing all 20 predictions
vs. ground truth with a batch accuracy table.

Usage:
    python generate_20_report.py
"""

import os
import sys
import random
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from PIL import Image

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.config import CLASSES, EDCNN_WEIGHTS, UNET_WEIGHTS
from models.edcnn import get_compiled_edcnn
from models.unet import build_unet
from predict_custom_image import (
    load_image, predict_classification,
    predict_segmentation, get_saliency_heatmap
)
from gradcam_xai import overlay_heatmap as _overlay_heatmap, draw_tumor_bbox
from utils.data_loader import _load_image as _load_image_dimmed


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def overlay_seg_mask(img_pil, mask, color=(255, 80, 0), alpha=0.45):
    """Draw predicted segmentation as a semi-transparent red-orange overlay."""
    img_u8 = np.array(img_pil.convert("RGB"))
    if mask.ndim == 3:
        mask = mask.squeeze()
    mask_r = cv2.resize(mask.astype(np.float32),
                        (img_u8.shape[1], img_u8.shape[0]),
                        interpolation=cv2.INTER_NEAREST)
    mask_bin = mask_r > 127
    colored = np.zeros_like(img_u8)
    colored[mask_bin] = color
    result = img_u8.copy().astype(np.float32)
    result[mask_bin] = (1 - alpha) * result[mask_bin] + alpha * colored[mask_bin]
    return np.clip(result, 0, 255).astype(np.uint8)


def make_gradcam_overlay(img_pil, heatmap, alpha=0.45):
    """Classic research-paper style Grad-CAM overlay (full-field Jet blend)."""
    img_u8 = np.array(img_pil.convert("RGB"))
    heatmap_r = cv2.resize(heatmap.astype(np.float32),
                           (img_u8.shape[1], img_u8.shape[0]),
                           interpolation=cv2.INTER_LINEAR)
    h8 = (heatmap_r * 255).astype(np.uint8)
    colored = cv2.cvtColor(cv2.applyColorMap(h8, cv2.COLORMAP_JET),
                           cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_u8, 1 - alpha, colored, alpha, 0)


def make_class_panel(ax, classification_result, true_cls):
    """Render a text panel showing classification probabilities."""
    ax.set_facecolor("#1a1a2e")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    pred_cls = classification_result["class"]
    conf     = classification_result["confidence"]
    probs    = classification_result["probabilities"]
    correct  = pred_cls == true_cls

    status_color = "#00d26a" if correct else "#ff4757"
    status_text  = "[OK] CORRECT" if correct else "[!!] MISMATCH"

    # Header
    ax.text(0.5, 0.92, "Classification Result",
            ha="center", va="top", fontsize=10, fontweight="bold",
            color="white", transform=ax.transAxes)

    # Predicted class badge
    ax.text(0.5, 0.78, pred_cls.upper(),
            ha="center", va="top", fontsize=13, fontweight="bold",
            color=status_color, transform=ax.transAxes)
    ax.text(0.5, 0.66, f"{conf:.1f}% confidence",
            ha="center", va="top", fontsize=9, color="#dfe6e9",
            transform=ax.transAxes)

    # Ground truth
    ax.text(0.5, 0.55, f"Ground Truth: {true_cls.upper()}",
            ha="center", va="top", fontsize=9, color="#b2bec3",
            transform=ax.transAxes)

    # Validation badge
    ax.text(0.5, 0.44, status_text,
            ha="center", va="top", fontsize=11, fontweight="bold",
            color=status_color, transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#2d3436",
                      edgecolor=status_color, linewidth=1.5))

    # Probability bars
    bar_y = 0.30
    for cls, p in sorted(probs.items(), key=lambda x: -x[1]):
        bar_w = p / 100 * 0.7
        bar_color = "#e17055" if cls == pred_cls else "#74b9ff"
        ax.barh(bar_y, bar_w, height=0.06, left=0.15, color=bar_color,
                transform=ax.transAxes)
        ax.text(0.14, bar_y, f"{cls[:3].upper()}",
                ha="right", va="center", fontsize=7, color="white",
                transform=ax.transAxes)
        ax.text(0.15 + bar_w + 0.01, bar_y, f"{p:.1f}%",
                ha="left", va="center", fontsize=7, color="white",
                transform=ax.transAxes)
        bar_y -= 0.09


# ─────────────────────────────────────────────────────────────────────────────
# PER-SAMPLE PAGE
# ─────────────────────────────────────────────────────────────────────────────

def generate_sample_page(idx, img_path, true_cls,
                         model_edcnn, model_unet, out_dir):
    """
    Generate and save a single unified diagnostic figure.

    Instead of redundant multi-panel layouts, we display a single composite image
    ("Tumor Focus") which contains:
      1. The original grayscale ultrasound
      2. The Grad-CAM heatmap overlay (showing tumor location)
      3. A neon-green bounding box around the primary tumor zone

    IMPORTANT: classification and Grad-CAM use a background-dimmed image
    (dim_factor=0.3) matching the training distribution, while panel (a)
    always shows the original undimmed image for clinical readability.
    """
    # Load original for display
    img_array_raw, img_pil = load_image(img_path)

    # ── Step 1: U-Net segmentation (on raw image) ─────────────────────────────
    pred_mask = predict_segmentation(model_unet, img_array_raw)

    # ── Step 2: Classification on raw image ─────────────────────────────
    # We use the raw image because normal images were not dimmed during training,
    # so applying U-Net mask dimming to normal images causes false positives.
    cls_result = predict_classification(model_edcnn, img_array_raw)

    # ── Step 3: Grad-CAM on raw image with soft mask boost ────────────────
    heatmap = get_saliency_heatmap(
        model_edcnn, img_array_raw,
        cls_result["class_idx"], mask_pred=pred_mask
    )

    # ── Step 5: Build overlay panels ─────────────────────────────────────────
    seg_overlay  = overlay_seg_mask(img_pil, pred_mask)
    img_u8       = np.array(img_pil.convert("RGB"))
    gradcam_over = _overlay_heatmap(img_u8, heatmap, alpha=0.55, threshold=0.25)
    tumor_focus  = draw_tumor_bbox(gradcam_over, heatmap,
                                   threshold=0.55, color=(0, 255, 80), thickness=2)

    correct   = cls_result["class"] == true_cls
    hdr_color = "#00b894" if correct else "#d63031"
    status    = "[OK] CORRECT" if correct else "[!!] MISMATCH"

    # ── Figure: 1x4 Layout ──────────────────────────────────────────
    fig = plt.figure(figsize=(20, 5), facecolor="#0d0d1a")
    gs  = gridspec.GridSpec(1, 4, figure=fig, wspace=0.04,
                            left=0.02, right=0.98, top=0.82, bottom=0.05)
    
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    
    labels = [
        "1. Original Image",
        "2. Segmented",
        "3. Grad-CAM",
        "4. Validation & Prediction"
    ]
    
    axes[0].imshow(np.array(img_pil.convert("RGB")))
    axes[1].imshow(seg_overlay)
    axes[2].imshow(tumor_focus)
    
    for i in range(3):
        axes[i].set_facecolor("#0d0d1a")
        axes[i].set_title(labels[i], fontsize=11, fontweight="bold",
                          color="white", pad=8)
        axes[i].axis("off")

    # Panel 4: Validation text
    ax_txt = axes[3]
    ax_txt.set_facecolor("#1a1a2e")
    ax_txt.set_title(labels[3], fontsize=11, fontweight="bold",
                     color="white", pad=8)
    ax_txt.axis("off")
    ax_txt.set_xlim(0, 1)
    ax_txt.set_ylim(0, 1)
    
    ax_txt.text(0.5, 0.85, f"True Label",
                ha="center", va="top", fontsize=11, color="#b2bec3", transform=ax_txt.transAxes)
    ax_txt.text(0.5, 0.75, f"{true_cls.upper()}",
                ha="center", va="top", fontsize=14, fontweight="bold", color="white", transform=ax_txt.transAxes)
                
    ax_txt.text(0.5, 0.55, f"Predicted",
                ha="center", va="top", fontsize=11, color="#b2bec3", transform=ax_txt.transAxes)
    ax_txt.text(0.5, 0.45, f"{cls_result['class'].upper()}",
                ha="center", va="top", fontsize=15, fontweight="bold", color=hdr_color, transform=ax_txt.transAxes)
    ax_txt.text(0.5, 0.35, f"Confidence: {cls_result['confidence']:.1f}%",
                ha="center", va="top", fontsize=10, color="white", transform=ax_txt.transAxes)
                
    ax_txt.text(0.5, 0.15, status,
                ha="center", va="top", fontsize=12, fontweight="bold", color=hdr_color,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#2d3436", edgecolor=hdr_color),
                transform=ax_txt.transAxes)

    # Clean figure header
    fig.suptitle(
        f"Sample {idx+1:02d} | File: {os.path.basename(img_path)}",
        fontsize=13, fontweight="bold", color="white", y=0.98
    )

    fname    = f"sample_{idx+1:02d}_{true_cls}_{'OK' if correct else 'FAIL'}.png"
    out_path = os.path.join(out_dir, fname)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0d0d1a")
    plt.close()

    return {
        "idx":      idx + 1,
        "filename": os.path.basename(img_path),
        "true_cls": true_cls,
        "pred_cls": cls_result["class"],
        "conf":     cls_result["confidence"],
        "correct":  correct,
    }


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION SUMMARY PAGE
# ─────────────────────────────────────────────────────────────────────────────

def generate_summary_page(results, out_dir):
    """
    Build a research-style validation summary page showing:
     - Accuracy per class
     - Overall accuracy
     - Table of all 20 predictions
    """
    total    = len(results)
    correct  = sum(1 for r in results if r["correct"])
    accuracy = correct / total * 100

    # Per-class stats
    class_stats = {}
    for cls in CLASSES:
        clr = [r for r in results if r["true_cls"] == cls]
        if clr:
            correct_c = sum(1 for r in clr if r["correct"])
            class_stats[cls] = {
                "n": len(clr),
                "correct": correct_c,
                "acc": correct_c / len(clr) * 100
            }

    # ── Figure ──
    fig = plt.figure(figsize=(18, 14), facecolor="#0d0d1a")
    fig.suptitle("Validation Summary — 20-Sample Diagnostic Report",
                 fontsize=18, fontweight="bold", color="white", y=0.97)

    gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.45,
                           top=0.92, bottom=0.04, left=0.04, right=0.96)

    # ── Top: overall + per-class accuracy bars ──
    ax_bar = fig.add_subplot(gs[0])
    ax_bar.set_facecolor("#1a1a2e")

    classes_list = list(class_stats.keys())
    accs         = [class_stats[c]["acc"] for c in classes_list]
    bar_colors   = ["#00b894" if a >= 80 else "#e17055" for a in accs]

    bars = ax_bar.bar(classes_list, accs, color=bar_colors,
                      edgecolor="white", linewidth=0.8, width=0.4)
    for bar, a in zip(bars, accs):
        ax_bar.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1.5,
                    f"{a:.1f}%", ha="center", va="bottom",
                    fontsize=12, fontweight="bold", color="white")

    ax_bar.axhline(accuracy, color="#fdcb6e", linewidth=2,
                   linestyle="--", label=f"Overall: {accuracy:.1f}%")
    ax_bar.set_ylim(0, 115)
    ax_bar.set_ylabel("Accuracy (%)", color="white", fontsize=12)
    ax_bar.set_title("Per-Class & Overall Accuracy", color="white",
                     fontsize=13, fontweight="bold")
    ax_bar.tick_params(colors="white")
    ax_bar.spines[["top", "right"]].set_visible(False)
    for spine in ["bottom", "left"]:
        ax_bar.spines[spine].set_color("#636e72")
    ax_bar.legend(fontsize=11, facecolor="#2d3436",
                  labelcolor="white", edgecolor="#636e72")

    # ── Middle: overall stats text ──
    ax_txt = fig.add_subplot(gs[1])
    ax_txt.set_facecolor("#1a1a2e")
    ax_txt.axis("off")

    stats_lines = [
        f"Total Samples Evaluated : {total}",
        f"Correctly Classified    : {correct}",
        f"Misclassified           : {total - correct}",
        f"Overall Accuracy        : {accuracy:.1f}%",
    ]
    for i, cls in enumerate(classes_list):
        s = class_stats[cls]
        stats_lines.append(
            f"  {cls.capitalize():12s}: {s['correct']}/{s['n']} correct  ({s['acc']:.1f}%)"
        )

    ax_txt.text(0.02, 0.90, "\n".join(stats_lines),
                transform=ax_txt.transAxes, fontsize=12,
                color="white", va="top", family="monospace",
                bbox=dict(boxstyle="round,pad=0.6",
                          facecolor="#2d3436", edgecolor="#636e72"))

    # ── Bottom: prediction table ──
    ax_tbl = fig.add_subplot(gs[2])
    ax_tbl.set_facecolor("#0d0d1a")
    ax_tbl.axis("off")

    col_labels = ["#", "Filename", "True Class", "Predicted", "Confidence", "Result"]
    table_data = []
    for r in results:
        table_data.append([
            str(r["idx"]),
            r["filename"][:28],
            r["true_cls"].upper(),
            r["pred_cls"].upper(),
            f"{r['conf']:.2f}%",
            "PASS" if r["correct"] else "FAIL"
        ])

    tbl = ax_tbl.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="center", loc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.45)

    # Style header
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#2c3e50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    # Style rows
    for i, r in enumerate(results, start=1):
        row_color = "#1e3a2f" if r["correct"] else "#3a1e1e"
        res_color = "#00b894" if r["correct"] else "#d63031"
        for j in range(len(col_labels)):
            tbl[i, j].set_facecolor(row_color)
            tbl[i, j].set_text_props(color="white")
        tbl[i, 5].set_text_props(color=res_color, fontweight="bold")

    ax_tbl.set_title("Prediction Table — All 20 Samples",
                     color="white", fontsize=12, fontweight="bold", pad=12)

    out_path = os.path.join(out_dir, "00_validation_summary.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0d0d1a")
    plt.close()
    print(f"\n  [Saved] Summary -> {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  20-Sample Comprehensive Diagnostic Report")
    print("=" * 60)

    # Load models
    print("\n[1/3] Loading models...")
    model_edcnn = get_compiled_edcnn(freeze_base=False)
    model_edcnn.load_weights(EDCNN_WEIGHTS)

    model_unet = build_unet()
    model_unet.load_weights(UNET_WEIGHTS)
    print("  [OK] Models loaded")

    # Gather images
    print("\n[2/3] Collecting test images...")
    all_files = []
    for cls in CLASSES:
        img_dir = os.path.join("dataset", "test", cls, "images")
        if os.path.exists(img_dir):
            for f in sorted(os.listdir(img_dir)):
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    all_files.append((os.path.join(img_dir, f), cls))

    if len(all_files) < 20:
        print(f"  WARNING: Only {len(all_files)} images found. Using all.")

    # Balanced sampling: ~7 per class (where available)
    random.seed(99)
    sampled = []
    per_class_target = max(1, 20 // len(CLASSES))
    for cls in CLASSES:
        cls_files = [(p, c) for p, c in all_files if c == cls]
        take = min(per_class_target, len(cls_files))
        sampled.extend(random.sample(cls_files, take))

    # Top-up to exactly 20 if needed
    remaining = [x for x in all_files if x not in sampled]
    random.shuffle(remaining)
    while len(sampled) < 20 and remaining:
        sampled.append(remaining.pop())

    sampled = sampled[:20]
    random.shuffle(sampled)

    print(f"  [OK] Selected {len(sampled)} samples:")
    for cls in CLASSES:
        n = sum(1 for _, c in sampled if c == cls)
        print(f"      {cls}: {n}")

    # Output directory
    out_dir = os.path.join("results", "20_sample_report")
    os.makedirs(out_dir, exist_ok=True)

    # Process each sample
    print(f"\n[3/3] Generating {len(sampled)} diagnostic pages...")
    results = []
    for idx, (img_path, true_cls) in enumerate(sampled):
        print(f"  [{idx+1:02d}/{len(sampled)}] {os.path.basename(img_path)} ({true_cls})")
        try:
            r = generate_sample_page(
                idx, img_path, true_cls,
                model_edcnn, model_unet, out_dir
            )
            results.append(r)
            status = "CORRECT" if r["correct"] else "MISMATCH"
            print(f"        -> Pred: {r['pred_cls'].upper()} {r['conf']:.1f}%  {status}")
        except Exception as e:
            print(f"        -> ERROR: {e}")

    # Validation summary
    generate_summary_page(results, out_dir)

    # Print terminal summary
    total    = len(results)
    correct  = sum(1 for r in results if r["correct"])
    accuracy = correct / total * 100

    print(f"\n{'='*60}")
    print(f"  VALIDATION COMPLETE")
    print(f"  Samples : {total}")
    print(f"  Correct : {correct}")
    print(f"  Accuracy: {accuracy:.1f}%")
    print(f"{'='*60}")
    print(f"\n  All outputs saved to: {os.path.abspath(out_dir)}")
    print(f"  Files:")
    print(f"    00_validation_summary.png  -  Summary + table")
    print(f"    sample_XX_*.png            -  20 per-sample pages")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
