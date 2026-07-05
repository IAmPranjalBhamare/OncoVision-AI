"""
gradcam_xai.py
Grad-CAM explainability — dual-backbone (DenseNet + MobileNet).

Improvements:
  - Combines Grad-CAM from BOTH backbones via weighted sum.
  - 98th-percentile contrast clipping before normalization.
  - Reduced Gaussian blur (7x7) for tighter focus.
  - Contribution-ordered tier overlay: regions ranked by activation percentile.
  - Segmentation-guided contribution map for U-Net explainability.
  - enhance_heatmap() helper with CLAHE-like sharpening.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import cv2
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.config import (
    CLASSES, EDCNN_WEIGHTS, GRADCAM_DIR,
    TEST_DIR,
)
from utils.data_loader import load_classification_data
from models.edcnn import get_compiled_edcnn


# ──────────────────────────────────────────────────────────────────────────────
# ALIGNMENT ACCURACY
# ──────────────────────────────────────────────────────────────────────────────

def compute_alignment_score(img_gray, heatmap, threshold=0.5):
    """
    Composite 0-100 score measuring how well the heatmap
    aligns with real tissue in the image.
    """
    img_f = img_gray.astype(np.float32)
    if img_f.max() > 1.0:
        img_f = img_f / 255.0
    if img_f.ndim == 3:
        img_f = np.mean(img_f, axis=-1)

    if heatmap.shape != img_f.shape:
        heatmap = cv2.resize(heatmap, (img_f.shape[1], img_f.shape[0]))

    hot_mask  = (heatmap >= threshold).astype(np.float32)
    cold_mask = (heatmap <  threshold).astype(np.float32)
    tissue    = (img_f   >  0.05).astype(np.float32)

    hot_pixels = hot_mask.sum()

    overlap_ratio = float((hot_mask * tissue).sum() / (hot_pixels + 1e-8))
    concentration = float(np.clip(1.0 - hot_pixels / float(img_f.size), 0, 1))
    hot_mean      = float((img_f * hot_mask).sum()  / (hot_pixels          + 1e-8))
    cold_mean     = float((img_f * cold_mask).sum() / (cold_mask.sum()     + 1e-8))
    contrast      = float(np.clip(abs(hot_mean - cold_mean) * 3, 0, 1))

    score = (0.5 * overlap_ratio + 0.3 * concentration + 0.2 * contrast) * 100
    return {
        "overlap_ratio"     : round(overlap_ratio * 100, 1),
        "concentration"     : round(concentration * 100, 1),
        "weighted_contrast" : round(contrast       * 100, 1),
        "score"             : round(score,           1),
    }


def print_alignment_report(class_name, sample_scores):
    print(f"\n  +-- Alignment Report [{class_name.upper()}] --------------------")
    for i, s in enumerate(sample_scores):
        print(f"  |  Sample {i+1}: score={s['score']:5.1f}/100  "
              f"overlap={s['overlap_ratio']:5.1f}%  "
              f"concentration={s['concentration']:5.1f}%  "
              f"contrast={s['weighted_contrast']:5.1f}%")
    avg = np.mean([s['score'] for s in sample_scores])
    print(f"  |  Mean alignment score: {avg:.1f}/100")
    print(f"  +------------------------------------------------------")


# ──────────────────────────────────────────────────────────────────────────────
# HEATMAP ENHANCEMENT
# ──────────────────────────────────────────────────────────────────────────────

def sigmoid_contrast_boost(x, k=12, mid=0.5):
    """
    Apply a sigmoid boost to expand the high-intensity 'hot' regions
    and sharpen the fall-off to the background.
    """
    return 1 / (1 + np.exp(-k * (x - mid)))


def keep_largest_hotspot(heatmap, threshold=0.15):
    """
    Removes scattered noisy activations by keeping only the largest 
    connected component (dominant lesion area).
    """
    # Create binary mask of interesting regions
    binary = (heatmap > threshold).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    if num_labels <= 1: # Only background
        return heatmap
        
    # Find the largest area (excluding background at label 0)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    
    # Create a mask where only that component survives
    mask = (labels == largest_label).astype(np.float32)
    
    # Apply Gaussian blur to the mask to avoid "cookie cutter" edges
    mask = cv2.GaussianBlur(mask, (15, 15), 0)
    
    return heatmap * mask


def apply_clinical_refinement(heatmap, mask_pred=None, original_img=None):
    """
    Radiological Post-Processing:
    Gently emphasizes high-activation (tumor) regions while keeping
    the full-image gradient — no zeroing of background.
    """
    if mask_pred is not None and mask_pred.max() > 0:
        mask_norm = (mask_pred / 255.0).astype(np.float32)
        if mask_norm.ndim == 3: mask_norm = mask_norm.squeeze()

        if mask_norm.shape != heatmap.shape:
            mask_norm = cv2.resize(mask_norm, (heatmap.shape[1], heatmap.shape[0]),
                                   interpolation=cv2.INTER_LINEAR)

        # Mild emphasis inside the segmented region (×1.0 outside, ×1.4 inside)
        soft_mask = cv2.GaussianBlur(mask_norm, (31, 31), 0)
        emphasis = 1.0 + 0.4 * soft_mask
        heatmap = heatmap * emphasis

        # Renormalize to [0, 1] while preserving the floor already set in get_gradcam
        h_max = heatmap.max()
        if h_max > 0:
            heatmap = heatmap / h_max

    return heatmap.astype(np.float32)


def enhance_heatmap(heatmap, percentile=None):
    """
    Sharpen heatmap contrast according to Adaptive XAI Analytics.
    Optimized for 'Soft Bullseye' effect: focused core with radiant halo.
    """
    h_min, h_max = heatmap.min(), heatmap.max()
    if h_max - h_min < 1e-8:
        return heatmap

    # 1. Normalize and Apply Wide-Context Sigmoid Boost
    hnorm = (heatmap - h_min) / (h_max - h_min)
    
    # mid=0.15 allows the low-level signal to spread across the whole image.
    # k=4.5 provides a very soft, natural gradient.
    hnorm = sigmoid_contrast_boost(hnorm, k=4.5, mid=0.15)
    
    # 2. Power-Law Stretch (Gamma) for intense center
    # Slightly increased gamma to ensure the center still pops despite the wide context
    hnorm = np.power(hnorm, 1.5)
    
    # 3. Aggressive "Radiant" Smoothing
    # We use a very large Gaussian kernel to get that 'liquid' medical look
    h_vibrant = cv2.GaussianBlur(hnorm, (41, 41), 0)
    
    # Enhance the edges via moderate Bilateral filtering
    h_vibrant_u8 = (h_vibrant * 255).astype(np.uint8)
    h_smooth = cv2.bilateralFilter(h_vibrant_u8, d=15, sigmaColor=75, sigmaSpace=75)
    
    heatmap = h_smooth.astype(np.float32) / 255.0

    return heatmap.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# CONTRIBUTION-ORDERED HEATMAP (RANKED TIERS)
# ──────────────────────────────────────────────────────────────────────────────

# Tier definitions updated for Adaptive Intensity Thresholding
_CONTRIBUTION_TIERS = [
    ("Tier 1 · Critical Signal",   80, (220,  38,  38), 0.88),   # Upper 20% of mapped signal
    ("Tier 2 · High Active",       55, (249, 115,  22), 0.72),   # Mid-high mapped signal
    ("Tier 3 · Moderate Focus",    40, (234, 179,   8), 0.50),   # Above Otsu Threshold (mapped >= 0.4)
    ("Background Noise",            0, ( 14, 165, 233), 0.10),   # Below Otsu Threshold
]


def generate_contribution_map(heatmap: np.ndarray) -> np.ndarray:
    """
    Convert a [0,1] heatmap to an Adaptive XAI analytic map [0,1].
    Uses Otsu's method to establish true background noise, scaling 
    signal intensities to provide a mathematically robust semantic overlay.
    """
    h_min, h_max = heatmap.min(), heatmap.max()
    if h_max - h_min < 1e-8: return np.zeros_like(heatmap)
    
    hnorm = (heatmap - h_min) / (h_max - h_min)
    h_u8 = (hnorm * 255).astype(np.uint8)
    otsu_val, _ = cv2.threshold(h_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t_bg = otsu_val / 255.0
    
    # Scale background into [0.0, 0.4)
    # Scale active foreground into [0.4, 1.0]
    ranks = np.zeros_like(hnorm)
    bg_mask = hnorm <= t_bg
    fg_mask = hnorm > t_bg
    
    if t_bg > 0:
        ranks[bg_mask] = 0.4 * (hnorm[bg_mask] / t_bg)
    if t_bg < 1.0:
        ranks[fg_mask] = 0.4 + 0.6 * ((hnorm[fg_mask] - t_bg) / (1.0 - t_bg))
        
    return ranks


def create_ranked_overlay(
    img_u8: np.ndarray,
    heatmap: np.ndarray,
    mask_pred: np.ndarray = None,
    n_blur: int = 5,
) -> np.ndarray:
    """
    Overlay a contribution-ordered tier heatmap on top of the image.

    Tier coloring (most-to-least contributive):
      Tier 1 · Critical  (≥90th pct) → Crimson
      Tier 2 · High      (70–90th)   → Orange
      Tier 3 · Moderate  (40–70th)   → Yellow
      Background         (<40th)     → Faint blue wash

    Args:
        img_u8   : uint8 RGB image (H, W, 3)
        heatmap  : float32 heatmap [0, 1] (H, W)
        mask_pred: optional uint8 binary mask (H, W) — if provided, ranks are
                   computed only within the segmented region for clinical accuracy
        n_blur   : kernel size for softening tier edges

    Returns:
        uint8 RGB image with contribution-tier overlay
    """
    H, W = img_u8.shape[:2]

    # Resize heatmap if needed
    if heatmap.shape != (H, W):
        heatmap = cv2.resize(heatmap, (W, H), interpolation=cv2.INTER_LINEAR)

    # Optional: restrict ranking to segmented region
    if mask_pred is not None and mask_pred.max() > 0:
        seg_mask = (mask_pred.squeeze() > 127).astype(np.float32)
        if seg_mask.shape != (H, W):
            seg_mask = cv2.resize(seg_mask, (W, H), interpolation=cv2.INTER_NEAREST)
        seg_soft = cv2.GaussianBlur(seg_mask, (21, 21), 0)
        heatmap_for_rank = heatmap * seg_soft
    else:
        heatmap_for_rank = heatmap

    # Build rank map: each pixel gets a percentile rank among non-zero pixels
    rank_map = generate_contribution_map(heatmap_for_rank)

    # Tier boundaries as floats
    tier_bounds = [(lbl, pfl / 100.0, clr, a)
                   for lbl, pfl, clr, a in _CONTRIBUTION_TIERS]

    # Start from a darkened version of the original (so overlays stand out)
    base = (img_u8.astype(np.float32) * 0.45).clip(0, 255)
    result = base.copy()

    # Paint tiers from background up so higher tiers overwrite lower ones
    for i in range(len(tier_bounds) - 1, -1, -1):
        lbl, lo, clr, alpha_mult = tier_bounds[i]
        hi = tier_bounds[i - 1][1] if i > 0 else 1.001   # top tier goes to 1

        tier_mask = ((rank_map >= lo) & (rank_map < hi)).astype(np.float32)

        # Smooth tier boundary to avoid hard pixel edges
        if n_blur > 1 and tier_mask.sum() > 0:
            tier_mask = cv2.GaussianBlur(tier_mask, (n_blur, n_blur), 0)

        # Alpha scales with how deep into the tier the pixel is
        tier_range = hi - lo if (hi - lo) > 1e-6 else 1.0
        depth = np.clip((rank_map - lo) / tier_range, 0.0, 1.0)
        intensity_alpha = (tier_mask * alpha_mult * (0.4 + 0.6 * depth))[..., np.newaxis]

        color_layer = np.full((H, W, 3), clr, dtype=np.float32)
        result = result * (1.0 - intensity_alpha) + color_layer * intensity_alpha

    return np.clip(result, 0, 255).astype(np.uint8)


def create_segmentation_contribution_overlay(
    img_u8: np.ndarray,
    mask_pred: np.ndarray,
    heatmap: np.ndarray = None,
) -> np.ndarray:
    """
    Creates a segmentation-focused contribution map showing which parts of the
    *segmented region* contributed most to the prediction.

    Logic:
      - If heatmap is provided: restrict ranking to the mask region, rank by
        Grad-CAM activation within the tumor area (= U-Net contribution proxy).
      - Fallback: use distance transform (center-of-mass = highest rank).

    Returns:
        uint8 RGB image with segmentation contribution overlay
    """
    H, W = img_u8.shape[:2]
    mask_u8 = mask_pred.squeeze()
    if mask_u8.shape != (H, W):
        mask_u8 = cv2.resize(mask_u8, (W, H), interpolation=cv2.INTER_NEAREST)

    binary = (mask_u8 > 127).astype(np.uint8)

    if binary.sum() == 0:
        # No segmentation — return image with a faint blue tint
        tint = img_u8.astype(np.float32) * 0.7
        tint[..., 2] = np.clip(tint[..., 2] + 60, 0, 255)
        return tint.astype(np.uint8)

    if heatmap is not None:
        # Use Grad-CAM activation restricted to segmented region
        seg_float = binary.astype(np.float32)
        seg_hm = cv2.resize(heatmap, (W, H), interpolation=cv2.INTER_LINEAR) \
                 if heatmap.shape != (H, W) else heatmap
        contribution = seg_hm * seg_float
    else:
        # Fallback: distance transform (pixels near center = higher contribution)
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        contribution = dist.astype(np.float32)

    contribution = cv2.GaussianBlur(contribution, (9, 9), 0)
    if contribution.max() > 0:
        contribution = contribution / contribution.max()

    # Use segmentation mask to restrict the ranked overlay region
    return create_ranked_overlay(img_u8, contribution, mask_pred=mask_u8)


# ──────────────────────────────────────────────────────────────────────────────
# DUAL-BACKBONE GRAD-CAM
# ──────────────────────────────────────────────────────────────────────────────

def get_gradcam(model, img_array, class_idx, n_samples=5, noise_spread=0.12):
    """
    Dual-backbone SmoothGrad-CAM++ combining DenseNet and MobileNet.
    Uses SmoothGrad (noise averaging) and Pos-Only Gradients (Grad-CAM++ style).
    Returns heatmap in [0, 1] matching input (H, W).
    """
    img_tensor = tf.cast(img_array, tf.float32)
    H, W = img_array.shape[1], img_array.shape[2]

    # Find sub-models
    primary_sm = None
    mobilenet_sm    = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            if "densenet" in layer.name.lower() or "xception" in layer.name.lower(): 
                primary_sm = layer
            if "mobilenet" in layer.name.lower(): 
                mobilenet_sm = layer

    # Build the dense head score function
    dense_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.Dense)]

    combined_heatmaps = []

    for i in range(n_samples):
        # Apply Gaussian noise for SmoothGrad (unless only 1 sample)
        if n_samples > 1:
            stdev = noise_spread * (tf.reduce_max(img_tensor) - tf.reduce_min(img_tensor))
            noise = tf.random.normal(tf.shape(img_tensor), mean=0.0, stddev=stdev)
            noisy_img_tensor = img_tensor + noise
        else:
            noisy_img_tensor = img_tensor

        cams = []

        # ── Primary Backbone CAM ──────────────────────────────────────────────────────
        if primary_sm is not None and mobilenet_sm is not None:
            last_conv_dense = None
            for layer in reversed(primary_sm.layers):
                if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D, tf.keras.layers.SeparableConv2D)):
                    last_conv_dense = layer
                    break

            if last_conv_dense is not None:
                mini_dense = tf.keras.Model(
                    inputs  = primary_sm.input,
                    outputs = [last_conv_dense.output, primary_sm.output],
                    name    = "mini_dense"
                )
                with tf.GradientTape() as tape:
                    conv_d, dense_out = mini_dense(noisy_img_tensor, training=False)
                    tape.watch(conv_d)
                    mob_out  = mobilenet_sm(noisy_img_tensor, training=False)
                    dp       = tf.reduce_mean(dense_out, axis=[1, 2])
                    mp       = tf.reduce_mean(mob_out, axis=[1, 2])
                    merged   = tf.concat([mp, dp], axis=-1)
                    x_score = merged
                    for j, dl in enumerate(dense_layers):
                        w, b = dl.kernel, dl.bias
                        x_score = tf.nn.relu(tf.linalg.matmul(x_score, w) + b) \
                                  if j < len(dense_layers) - 1 \
                                  else tf.linalg.matmul(x_score, w) + b
                    score_d = x_score[:, class_idx]

                grads_d = tape.gradient(score_d, conv_d)
                if grads_d is not None:
                    maps_d = conv_d[0].numpy()
                    grads_d_val = grads_d[0].numpy()
                    
                    # Grad-CAM++ Element: Only positive gradients strongly influence the map
                    pos_grads_d = np.maximum(grads_d_val, 0)
                    weights_d = np.mean(pos_grads_d, axis=(0, 1))
                    
                    cam_d = np.sum(maps_d * weights_d, axis=-1)
                    cam_d = np.maximum(cam_d, 0) # Strictly positive activations
                    
                    if cam_d.max() > 1e-8:
                        if i == 0:
                            print(f"  [SmoothGrad-CAM++ DenseNet] layer={last_conv_dense.name} max={cam_d.max():.6f}")
                        cams.append(("densenet", cam_d, 0.55))

        # ── MobileNetV2 CAM ───────────────────────────────────────────────────────
        if mobilenet_sm is not None and primary_sm is not None:
            last_conv_mob = None
            for layer in reversed(mobilenet_sm.layers):
                if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D, tf.keras.layers.SeparableConv2D)):
                    last_conv_mob = layer
                    break

            if last_conv_mob is not None:
                mini_mob = tf.keras.Model(
                    inputs  = mobilenet_sm.input,
                    outputs = [last_conv_mob.output, mobilenet_sm.output],
                    name    = "mini_mob"
                )
                with tf.GradientTape() as tape:
                    conv_m, mob_out2 = mini_mob(noisy_img_tensor, training=False)
                    tape.watch(conv_m)
                    dense_out2 = primary_sm(noisy_img_tensor, training=False)
                    dp2       = tf.reduce_mean(dense_out2,  axis=[1, 2])
                    mp2       = tf.reduce_mean(mob_out2,  axis=[1, 2])
                    merged2   = tf.concat([mp2, dp2], axis=-1)
                    x_score2 = merged2
                    for j, dl in enumerate(dense_layers):
                        w, b = dl.kernel, dl.bias
                        x_score2 = tf.nn.relu(tf.linalg.matmul(x_score2, w) + b) \
                                   if j < len(dense_layers) - 1 \
                                   else tf.linalg.matmul(x_score2, w) + b
                    score_m = x_score2[:, class_idx]

                grads_m = tape.gradient(score_m, conv_m)
                if grads_m is not None:
                    maps_m = conv_m[0].numpy()
                    grads_m_val = grads_m[0].numpy()
                    
                    # Grad-CAM++ Element: Pos-only gradients
                    pos_grads_m = np.maximum(grads_m_val, 0)
                    weights_m = np.mean(pos_grads_m, axis=(0, 1))
                    
                    cam_m = np.sum(maps_m * weights_m, axis=-1)
                    cam_m = np.maximum(cam_m, 0)
                    
                    if cam_m.max() > 1e-8:
                        if i == 0:
                            print(f"  [SmoothGrad-CAM++ MobileNet] layer={last_conv_mob.name} max={cam_m.max():.6f}")
                        cams.append(("mobilenet", cam_m, 0.45))

        # ── Fallback: single DenseNet activation ──────────────────────────────
        if not cams and primary_sm is not None:
            last_conv = None
            for layer in reversed(primary_sm.layers):
                if isinstance(layer, (tf.keras.layers.Conv2D,
                                       tf.keras.layers.DepthwiseConv2D)):
                    last_conv = layer
                    break
            if last_conv is not None:
                mini = tf.keras.Model(inputs=primary_sm.input, outputs=last_conv.output)
                maps = mini(noisy_img_tensor, training=False)[0].numpy()
                cam_fallback = np.mean(np.abs(maps), axis=-1)
                if i == 0:
                    print(f"  [SmoothGrad-Fallback] activation only, max={cam_fallback.max():.6f}")
                cams.append(("fallback", cam_fallback, 1.0))

        # ── Combine CAMs for this sample ──────────────────────────────────────────
        combined_sample = np.zeros((H, W), dtype=np.float32)
        for name, cam, weight in cams:
            cam_resized = cv2.resize(cam.astype(np.float32), (W, H),
                                     interpolation=cv2.INTER_LINEAR)
            c_min, c_max = cam_resized.min(), cam_resized.max()
            if c_max - c_min > 1e-10:
                cam_resized = (cam_resized - c_min) / (c_max - c_min)
            combined_sample += weight * cam_resized
            
        combined_heatmaps.append(combined_sample)

    # ── Average across all SmoothGrad samples ─────────────────────────────────
    combined = np.mean(combined_heatmaps, axis=0)

    # ── Final Enhancement ─────────────────────────────────────────────────────
    combined = enhance_heatmap(combined)

    # ── Final normalization ───────────────────────────────────────────────────
    h_min, h_max = combined.min(), combined.max()
    if h_max - h_min > 1e-10:
        combined = (combined - h_min) / (h_max - h_min)
    else:
        combined = np.zeros_like(combined)

    return combined.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# OVERLAY
# ──────────────────────────────────────────────────────────────────────────────

def overlay_heatmap(img_u8, heatmap, alpha=0.50, threshold=0.0):
    """
    Experimental Wide-Context Overlay: 
    Vibrant JET colormap with density-aware alpha blending.
    Maintains image-wide context (no threshold cut) while boosting malignant core.
    """
    h8      = (heatmap * 255).astype(np.uint8)
    colored = cv2.applyColorMap(h8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    if colored.shape[:2] != img_u8.shape[:2]:
        colored = cv2.resize(colored, (img_u8.shape[1], img_u8.shape[0]))
        heatmap = cv2.resize(heatmap, (img_u8.shape[1], img_u8.shape[0]))

    # Density-aware alpha for image-wide coverage
    # No floor, but very low threshold (0.0) to preserve all context
    heat_alpha = (np.power(heatmap, 0.90)) * alpha
    heat_alpha = np.clip(heat_alpha, 0, 1.0)
    
    # threshold=0.0 means we do NOT mask out any cold regions (image-wide coverage)
    if threshold > 0:
        heat_alpha[heatmap < threshold] = 0.0
    
    heat_alpha = heat_alpha[..., np.newaxis].astype(np.float32)

    # Blend original and colored heatmap
    result = (img_u8.astype(np.float32) * (1.0 - heat_alpha) +
              colored.astype(np.float32) * heat_alpha)
    return np.clip(result, 0, 255).astype(np.uint8)


def draw_tumor_bbox(img_u8, heatmap, threshold=0.55, color=(0, 255, 80), thickness=2):
    """
    Draw a bounding box around the hottest heatmap region.
    """
    result = img_u8.copy()
    heatmap_r = heatmap
    if heatmap_r.shape[:2] != img_u8.shape[:2]:
        heatmap_r = cv2.resize(heatmap_r, (img_u8.shape[1], img_u8.shape[0]))

    hot_bin = (heatmap_r > threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(hot_bin, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        all_pts = np.concatenate(contours)
        x, y, w, h = cv2.boundingRect(all_pts)
        cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)
        cv2.putText(result, "Tumor Focus",
                    (x, max(y - 6, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
                    cv2.LINE_AA)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# MAIN PLOTTING
# ──────────────────────────────────────────────────────────────────────────────

def save_gradcam_plots(model_edcnn, model_unet, X_test, y_test, n_per_class=6):
    os.makedirs(GRADCAM_DIR, exist_ok=True)
    all_class_scores = {}

    for class_idx, class_name in enumerate(CLASSES):
        print(f"\n{'='*50}")
        print(f"[Class: {class_name.upper()}]")

        scored = []
        for i, y in enumerate(y_test):
            if y == class_idx:
                img_4d = X_test[i][np.newaxis, ...].astype(np.float32)
                pred   = model_edcnn(tf.cast(img_4d, tf.float32), training=False).numpy()[0]
                scored.append((pred[class_idx], i))
        scored.sort(reverse=True)
        indices = [idx for _, idx in scored[:n_per_class]]

        n    = len(indices)
        # 6 columns: original, heatmap, overlay, bbox, cls-contribution, seg-contribution
        fig, axes = plt.subplots(n, 6, figsize=(30, 5.5 * n), facecolor="#0d0d1a")
        if n == 1:
            axes = [axes]

        col_titles = [
            "(a) Original Input",
            "(b) Clinical Heatmap (JET)",
            "(c) Clinical Overlay",
            "(d) Diagnostic Focus",
            "(e) Classification\nContribution Rank",
            "(f) Segmentation\nContribution Rank",
        ]

        sample_scores = []

        for row, img_idx in enumerate(indices):
            img_norm = X_test[img_idx]
            img_4d   = img_norm[np.newaxis, ...]

            pred = model_edcnn(tf.cast(img_4d, tf.float32), training=False).numpy()[0]
            conf = pred[class_idx]

            print(f"\n  Sample {row+1}: idx={img_idx}  conf={conf:.4f}")

            # ── Grad-CAM heatmap ────────────────────────────────────────────
            heatmap = get_gradcam(model_edcnn, img_4d, class_idx)

            # ── Segmentation mask for steering & contribution
            mask_pred = None
            if model_unet is not None:
                mask_raw  = model_unet.predict(tf.cast(img_4d, tf.float32), verbose=0)[0, ..., 0]
                mask_pred = (mask_raw > 0.5).astype(np.uint8) * 255
                heatmap   = apply_clinical_refinement(heatmap, mask_pred, original_img=img_norm)

            # ── Build display image from normalized array ─────────────────
            img_disp = img_norm.copy()
            lo, hi   = img_disp.min(), img_disp.max()
            img_disp = (img_disp - lo) / (hi - lo + 1e-8)

            img_u8 = (img_disp * 255).astype(np.uint8)
            if img_u8.ndim == 2:
                img_u8 = np.stack([img_u8]*3, axis=-1)
            elif img_u8.shape[-1] == 1:
                img_u8 = np.concatenate([img_u8]*3, axis=-1)

            # ── Standard panels ───────────────────────────────────────────
            overlay       = overlay_heatmap(img_u8, heatmap, alpha=0.6, threshold=0.15)
            tumor_focused = draw_tumor_bbox(overlay, heatmap, threshold=0.55)

            # ── Contribution-ranked panels ────────────────────────────────
            cls_rank_img  = create_ranked_overlay(img_u8, heatmap, mask_pred=mask_pred)
            seg_rank_img  = create_segmentation_contribution_overlay(
                img_u8, mask_pred if mask_pred is not None
                else np.zeros(img_u8.shape[:2], dtype=np.uint8),
                heatmap=heatmap
            )

            align = compute_alignment_score(img_disp, heatmap, threshold=0.5)
            sample_scores.append(align)

            panels = [img_disp, heatmap, overlay, tumor_focused, cls_rank_img, seg_rank_img]
            cmaps  = ["gray", "jet", None, None, None, None]

            for col in range(6):
                ax = axes[row][col]
                ax.set_facecolor("#0d0d1a")
                if row == 0:
                    ax.set_title(col_titles[col], fontsize=11, fontweight="bold",
                                 color="white", pad=10)

                if cmaps[col] == "jet":
                    im = ax.imshow(panels[col], cmap="jet", vmin=0, vmax=1,
                                   interpolation="bilinear")
                    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    cbar.ax.tick_params(colors="white", labelsize=7)
                    cbar.set_label("Activation", color="white", fontsize=8)
                elif cmaps[col] == "gray":
                    ax.imshow(panels[col], cmap="gray")
                else:
                    ax.imshow(panels[col])

                ax.axis("off")

                if col == 0:
                    ax.text(-0.05, 0.5, f"Sample {row+1}\nconf={conf:.1%}",
                            transform=ax.transAxes, fontsize=10, va="center",
                            ha="right", color="white")

                if col == 2:
                    ax.text(0.02, 0.02,
                            f"align={align['score']}/100\n"
                            f"overlap={align['overlap_ratio']}%",
                            transform=ax.transAxes, fontsize=8, color="white",
                            bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.6),
                            va="bottom")

                # ── Tier legend on classification contribution panel ──────
                if col == 4 and row == 0:
                    legend_y = 0.01
                    for lbl, pfl, clr, _ in _CONTRIBUTION_TIERS:
                        ax.text(
                            0.02, legend_y, f"● {lbl}",
                            transform=ax.transAxes, fontsize=6.5,
                            color=f"#{clr[0]:02x}{clr[1]:02x}{clr[2]:02x}",
                            va="bottom",
                            bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.55)
                        )
                        legend_y += 0.09

        print_alignment_report(class_name, sample_scores)
        all_class_scores[class_name] = sample_scores

        avg_score = np.mean([s['score'] for s in sample_scores])
        fig.suptitle(
            f"Grad-CAM Explainability — {class_name.capitalize()}"
            f"  (Dual-Backbone)  |  Mean Alignment: {avg_score:.1f}/100",
            fontsize=13, fontweight="bold", color="white", y=1.005
        )
        plt.tight_layout()
        path = os.path.join(GRADCAM_DIR, f"gradcam_{class_name}.png")
        plt.savefig(path, dpi=200, bbox_inches="tight", facecolor="#0d0d1a")
        plt.close()
        print(f"\n  [Saved] {path}")


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Grad-CAM XAI — Dual-Backbone (Xception + MobileNet)")
    print("=" * 60)

    # Roots
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    edcnn_keras_path = os.path.join(results_dir, "edcnn_best.keras")
    unet_keras_path  = os.path.join(results_dir, "unet_best.keras")

    print(f"\n[Load] EDCNN from {edcnn_keras_path}")
    model_edcnn = tf.keras.models.load_model(edcnn_keras_path, compile=False)
    
    print(f"[Load] U-Net from {unet_keras_path}")
    model_unet = None
    if os.path.exists(unet_keras_path):
        model_unet = tf.keras.models.load_model(unet_keras_path, compile=False)

    print(f"\n[Data] Loading test set from: {TEST_DIR}")
    X_test, y_test = load_classification_data(TEST_DIR, dim_factor=0.3)
    print(f"  Total: {len(X_test)} samples")

    save_gradcam_plots(model_edcnn, model_unet, X_test, y_test, n_per_class=6)
    print("Done! Check results/gradcam/")


if __name__ == "__main__":
    main()
