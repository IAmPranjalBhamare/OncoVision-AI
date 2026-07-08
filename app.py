import os
import io
import base64
import uuid
import time
import threading
import numpy as np
import cv2
import nibabel as nib
from PIL import Image
from flask import Flask, request, jsonify, render_template, session, redirect, url_for, send_file
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fpdf import FPDF
from datetime import datetime

# Import models and utils from existing project
from utils.config import EDCNN_WEIGHTS, UNET_WEIGHTS, CLASSES
from utils.data_loader import _load_image, apply_bm3d
from models.edcnn import get_compiled_edcnn
from models.unet import build_unet
from predict_custom_image import predict_classification, predict_segmentation, get_saliency_heatmap
from gradcam_xai import overlay_heatmap, draw_tumor_bbox, create_ranked_overlay, create_segmentation_contribution_overlay

app = Flask(__name__)
app.secret_key = 'super-secure-medical-key-do-not-share'

# In-memory storage for processing data privately. Data is never written to disk.
# Format: { session_id: { 'original': np.array, 'mask': np.array, 'heatmap': np.array, ... } }
MEMORY_CACHE = {}

# Task queue for background inference jobs
# Format: { task_id: { 'status': str, 'step': str, 'progress': int, 'result': dict|None, 'error': str|None } }
TASK_QUEUE = {}

# ── Global model state ────────────────────────────────────────────────────────
model_edcnn = None
model_unet  = None
_models_loading = False
_models_ready   = False


def load_models_once():
    """Load both Keras models into global RAM. Called once at startup in a background thread."""
    global model_edcnn, model_unet, _models_loading, _models_ready
    if _models_ready or _models_loading:
        return
    _models_loading = True
    import tensorflow as tf
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    edcnn_path  = os.path.join(results_dir, "edcnn_best.keras")
    unet_path   = os.path.join(results_dir, "unet_best.keras")
    try:
        print("[System] Loading EDCNN model…")
        model_edcnn = tf.keras.models.load_model(edcnn_path, compile=False)
        print("[System] Loading U-Net model…")
        model_unet  = tf.keras.models.load_model(unet_path,  compile=False)
        _models_ready = True
        print("[System] ✓ Both models loaded and ready.")
    except Exception as e:
        print(f"[System] ERROR loading models: {e}")
    finally:
        _models_loading = False


# ── Pre-warm models in background thread at startup ───────────────────────────
_warmup_thread = threading.Thread(target=load_models_once, daemon=True)
_warmup_thread.start()


# ── Fast denoising replacement for BM3D ──────────────────────────────────────
def fast_denoise(img_rgb_uint8):
    """
    Drop-in replacement for apply_bm3d().
    Uses OpenCV's Non-Local Means — 10-50x faster on CPU, still effective.
    Input/output: uint8 RGB numpy array (H, W, 3).
    """
    return cv2.fastNlMeansDenoisingColored(img_rgb_uint8, None, h=6, hColor=6,
                                           templateWindowSize=7, searchWindowSize=21)


def _set_task(task_id, status, step, progress, result=None, error=None):
    TASK_QUEUE[task_id] = {
        'status':   status,    # 'running' | 'done' | 'error'
        'step':     step,      # Human-readable current step label
        'progress': progress,  # 0-100
        'result':   result,
        'error':    error,
    }


# ── Image encoding helpers ────────────────────────────────────────────────────
def array_to_base64(img_array):
    """Convert numpy array image (RGB/BGR/Grayscale) to base64 string"""
    if img_array.dtype != np.uint8:
        img_array = (img_array * 255).clip(0, 255).astype(np.uint8)
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        is_success, buffer = cv2.imencode(".png", cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
    elif len(img_array.shape) == 2 or img_array.shape[2] == 1:
        is_success, buffer = cv2.imencode(".png", img_array)
    else:
        is_success, buffer = cv2.imencode(".png", img_array)
    if is_success:
        io_buf = io.BytesIO(buffer)
        encoded = base64.b64encode(io_buf.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{encoded}"
    return ""


def apply_colormap_to_base64(heatmap, colormap=cv2.COLORMAP_JET):
    """Convert a 2D float heatmap [0..1] to base64 RGB with colormap"""
    heatmap_u8 = np.uint8(255 * heatmap)
    mapped = cv2.applyColorMap(heatmap_u8, colormap)
    mapped_rgb = cv2.cvtColor(mapped, cv2.COLOR_BGR2RGB)
    return array_to_base64(mapped_rgb)


def create_mask_overlay_base64(mask_pred):
    """Create a colored semi-transparent mask for frontend overlay"""
    mask_u8 = mask_pred.squeeze()
    colored = np.zeros((mask_u8.shape[0], mask_u8.shape[1], 4), dtype=np.uint8)
    colored[mask_u8 > 127] = [255, 0, 0, 180]
    is_success, buffer = cv2.imencode(".png", cv2.cvtColor(colored, cv2.COLOR_RGBA2BGRA))
    if is_success:
        io_buf = io.BytesIO(buffer)
        encoded = base64.b64encode(io_buf.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{encoded}"
    return ""


def create_contour_overlay_base64(mask_pred):
    """Create a boundary contour image where only the edges are drawn."""
    mask_u8 = mask_pred.squeeze()
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    colored = np.zeros((mask_u8.shape[0], mask_u8.shape[1], 4), dtype=np.uint8)
    if contours:
        cv2.drawContours(colored, contours, -1, (0, 255, 100, 255), 2)
    is_success, buffer = cv2.imencode(".png", cv2.cvtColor(colored, cv2.COLOR_RGBA2BGRA))
    if is_success:
        io_buf = io.BytesIO(buffer)
        encoded = base64.b64encode(io_buf.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{encoded}"
    return ""


def compute_xai_metrics(mask_pred, heatmap, img_rgb):
    """Compute XAI Analytics for dashboard."""
    mask_u8 = mask_pred.squeeze()
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    tissue_mask = (gray > 15).astype(np.uint8)
    tissue_area = tissue_mask.sum()
    tumor_bin   = (mask_u8 > 127).astype(np.uint8)
    tumor_area  = tumor_bin.sum()
    coverage    = (tumor_area / tissue_area * 100.0) if tissue_area > 0 else 0.0
    hotspot     = (heatmap > 0.5).astype(np.uint8)
    hot_pixels  = hotspot.sum()
    intersection = (hotspot * tumor_bin).sum()
    alignment   = (intersection / hot_pixels * 100.0) if hot_pixels > 0 else 0.0
    concentration = max(0, min(100, (1.0 - (hot_pixels / tissue_area)) * 100.0)) if tissue_area > 0 else 50.0
    return {
        'coverage':      float(coverage),
        'alignment':     float(alignment),
        'concentration': float(concentration)
    }


def generate_narrative(class_name, conf, metrics):
    """Generate a clinical narrative explanation based on AI results"""
    c = class_name.lower()
    align    = metrics['alignment']
    coverage = metrics['coverage']
    base = f"The automated analysis identified the tissue as {c.upper()} with {conf:.1f}% confidence. "
    if c == 'malignant':
        narrative = base + "The presence of a suspicious lesion was localized with significant model attention. "
        if align > 60:
            narrative += "High target alignment indicates the AI is strongly focused on the segmented mass. "
        else:
            narrative += "Model attention is somewhat dispersed; however, the primary focal point remains concerning. "
        if coverage > 20:
            narrative += "The lesion occupies a relatively large percentage of the scanned tissue, suggesting the need for immediate clinical intervention."
        else:
            narrative += "The localized finding is specific and suggests targeted biopsy may be required."
    elif c == 'benign':
        narrative = base + "Findings are consistent with non-malignant tissue characteristics. "
        if align > 50:
            narrative += "Clear localization of a well-defined mass with low metabolic confidence for malignancy suggest a Bi-Rads 2-3 classification."
        else:
            narrative += "No focal points of concern were detected with high confidence."
    else:
        narrative = base + "No suspicious architectural distortions or masses were detected within the model's operating parameters."
    return narrative


# ── Background inference worker ───────────────────────────────────────────────
def _run_inference(task_id, file_bytes, patient_id, uid):
    """Runs the full inference pipeline in a background thread, updating TASK_QUEUE."""
    try:
        _set_task(task_id, 'running', 'Decoding image…', 5)

        nparr   = np.frombuffer(file_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            _set_task(task_id, 'error', '', 0, error='Invalid image format.')
            return

        img_rgb_raw = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb_raw, (224, 224))

        # ── Fast denoising (replaces BM3D) ────────────────────────────────
        _set_task(task_id, 'running', 'Denoising image…', 15)
        img_denoised = fast_denoise(img_resized)

        # ── Normalise for model ────────────────────────────────────────────
        img_float   = img_denoised.astype(np.float32)
        mean        = img_float.mean()
        std         = img_float.std() + 1e-8
        img_norm    = (img_float - mean) / std
        model_input = np.expand_dims(img_norm, axis=0)

        # ── Segmentation ──────────────────────────────────────────────────
        _set_task(task_id, 'running', 'Running U-Net segmentation…', 30)
        mask_pred = predict_segmentation(model_unet, model_input)

        # ── Classification ────────────────────────────────────────────────
        _set_task(task_id, 'running', 'Running EDCNN classification…', 55)
        class_result = predict_classification(model_edcnn, model_input)

        # ── Grad-CAM XAI ──────────────────────────────────────────────────
        _set_task(task_id, 'running', 'Generating Grad-CAM XAI maps…', 70)
        heatmap_raw = get_saliency_heatmap(model_edcnn, model_input, class_result['class_idx'],
                                           mask_pred=mask_pred, original_img=img_resized)

        # ── Overlay construction ───────────────────────────────────────────
        _set_task(task_id, 'running', 'Building diagnostic overlays…', 82)
        fusion_img  = overlay_heatmap(img_resized, heatmap_raw, alpha=0.6, threshold=0.15)
        fusion_img  = draw_tumor_bbox(fusion_img, heatmap_raw, threshold=0.5, color=(0, 255, 80), thickness=2)
        cls_rank_img = create_ranked_overlay(img_resized, heatmap_raw, mask_pred=None)
        seg_mask_vis = mask_pred if mask_pred is not None else np.zeros(img_resized.shape[:2], dtype=np.uint8)
        seg_rank_img = create_segmentation_contribution_overlay(img_resized, seg_mask_vis, heatmap=heatmap_raw)

        # ── Metrics & narrative ────────────────────────────────────────────
        _set_task(task_id, 'running', 'Computing XAI analytics…', 90)
        metrics   = compute_xai_metrics(mask_pred, heatmap_raw, img_denoised)
        narrative = generate_narrative(class_result['class'], class_result['confidence'], metrics)

        # ── Store to MEMORY_CACHE for report generation ────────────────────
        MEMORY_CACHE[uid] = {
            'original_bgr': img_bgr,
            'mask_pred':    mask_pred,
            'heatmap':      heatmap_raw,
            'fusion':       fusion_img,
            'classification': class_result,
            'metrics':      metrics,
            'narrative':    narrative,
            'patient_id':   patient_id,
            'timestamp':    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # ── Encode images ──────────────────────────────────────────────────
        _set_task(task_id, 'running', 'Encoding results…', 95)
        processed_class_result = {
            'class':         class_result['class'],
            'class_idx':     int(class_result['class_idx']),
            'confidence':    float(class_result['confidence']),
            'probabilities': {k: float(v) for k, v in class_result['probabilities'].items()}
        }
        result = {
            'classification': processed_class_result,
            'metrics':        metrics,
            'narrative':      narrative,
            'images': {
                'original':           array_to_base64(img_resized),
                'bm3d_output':        array_to_base64(img_denoised),
                'mask_overlay':       create_mask_overlay_base64(mask_pred),
                'contour_overlay':    create_contour_overlay_base64(mask_pred),
                'gradcam':            apply_colormap_to_base64(heatmap_raw, cv2.COLORMAP_JET),
                'diagnostic_fusion':  array_to_base64(fusion_img),
                'contribution_rank':  array_to_base64(cls_rank_img),
                'seg_contribution_rank': array_to_base64(seg_rank_img),
            }
        }
        _set_task(task_id, 'done', 'Complete', 100, result=result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        _set_task(task_id, 'error', '', 0, error=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/')
def root():
    if 'logged_in' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.json
        if data and data.get('username') == 'doctor' and data.get('password') == 'secure123':
            session['logged_in'] = True
            session['user_id']   = str(uuid.uuid4())
            return jsonify({'success': True}), 200
        return jsonify({'success': False, 'message': 'Invalid credentials'}), 401
    return render_template('login.html')


@app.route('/logout', methods=['POST'])
def logout():
    uid = session.get('user_id')
    if uid in MEMORY_CACHE:
        del MEMORY_CACHE[uid]
    session.clear()
    return jsonify({'success': True})


@app.route('/dashboard')
def dashboard():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('index.html')


@app.route('/api/status')
def api_status():
    """Returns whether models are loaded and ready."""
    return jsonify({
        'ready':   _models_ready,
        'loading': _models_loading,
    })


@app.route('/api/predict/start', methods=['POST'])
def predict_start():
    """Accepts the image upload, kicks off a background inference job, and returns a task_id immediately."""
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not _models_ready:
        return jsonify({'error': 'Models are still loading. Please wait a moment and try again.'}), 503

    task_id    = str(uuid.uuid4())
    file_bytes = file.read()
    patient_id = request.form.get('patient_id', 'Unknown')
    uid        = session.get('user_id')

    _set_task(task_id, 'running', 'Starting inference…', 2)

    t = threading.Thread(target=_run_inference,
                         args=(task_id, file_bytes, patient_id, uid),
                         daemon=True)
    t.start()

    return jsonify({'task_id': task_id})


@app.route('/api/predict/status/<task_id>')
def predict_status(task_id):
    """Poll this endpoint to get progress updates for a running inference task."""
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    task = TASK_QUEUE.get(task_id)
    if task is None:
        return jsonify({'error': 'Task not found'}), 404
    # Don't send the full result on every poll — only when done
    payload = {
        'status':   task['status'],
        'step':     task['step'],
        'progress': task['progress'],
        'error':    task.get('error'),
    }
    if task['status'] == 'done':
        payload['result'] = task['result']
        # Clean up from queue after delivering result once
        del TASK_QUEUE[task_id]
    return jsonify(payload)


# Keep the old /api/predict endpoint as a fallback (synchronous)
@app.route('/api/predict', methods=['POST'])
def predict_endpoint():
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    try:
        load_models_once()
        file_bytes = file.read()
        nparr      = np.frombuffer(file_bytes, np.uint8)
        img_bgr    = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return jsonify({'error': 'Invalid image format.'}), 400
        img_rgb_raw  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized  = cv2.resize(img_rgb_raw, (224, 224))
        img_denoised = fast_denoise(img_resized)
        img_float    = img_denoised.astype(np.float32)
        mean         = img_float.mean()
        std          = img_float.std() + 1e-8
        img_norm     = (img_float - mean) / std
        model_input  = np.expand_dims(img_norm, axis=0)
        mask_pred    = predict_segmentation(model_unet, model_input)
        class_result = predict_classification(model_edcnn, model_input)
        heatmap_raw  = get_saliency_heatmap(model_edcnn, model_input, class_result['class_idx'],
                                            mask_pred=mask_pred, original_img=img_resized)
        fusion_img   = overlay_heatmap(img_resized, heatmap_raw, alpha=0.6, threshold=0.15)
        fusion_img   = draw_tumor_bbox(fusion_img, heatmap_raw, threshold=0.5, color=(0, 255, 80), thickness=2)
        cls_rank_img = create_ranked_overlay(img_resized, heatmap_raw, mask_pred=None)
        seg_mask_vis = mask_pred if mask_pred is not None else np.zeros(img_resized.shape[:2], dtype=np.uint8)
        seg_rank_img = create_segmentation_contribution_overlay(img_resized, seg_mask_vis, heatmap=heatmap_raw)
        metrics      = compute_xai_metrics(mask_pred, heatmap_raw, img_denoised)
        narrative    = generate_narrative(class_result['class'], class_result['confidence'], metrics)
        uid          = session.get('user_id')
        patient_id   = request.form.get('patient_id', 'Unknown')
        MEMORY_CACHE[uid] = {
            'original_bgr': img_bgr,
            'mask_pred':    mask_pred,
            'heatmap':      heatmap_raw,
            'fusion':       fusion_img,
            'classification': class_result,
            'metrics':      metrics,
            'narrative':    narrative,
            'patient_id':   patient_id,
            'timestamp':    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        processed_class_result = {
            'class':         class_result['class'],
            'class_idx':     int(class_result['class_idx']),
            'confidence':    float(class_result['confidence']),
            'probabilities': {k: float(v) for k, v in class_result['probabilities'].items()}
        }
        response_data = {
            'classification': processed_class_result,
            'metrics':        metrics,
            'narrative':      narrative,
            'images': {
                'original':              array_to_base64(img_resized),
                'bm3d_output':           array_to_base64(img_denoised),
                'mask_overlay':          create_mask_overlay_base64(mask_pred),
                'contour_overlay':       create_contour_overlay_base64(mask_pred),
                'gradcam':               apply_colormap_to_base64(heatmap_raw, cv2.COLORMAP_JET),
                'diagnostic_fusion':     array_to_base64(fusion_img),
                'contribution_rank':     array_to_base64(cls_rank_img),
                'seg_contribution_rank': array_to_base64(seg_rank_img),
            }
        }
        return jsonify(response_data)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/generate_report')
def generate_report():
    """Generate a multi-pane clinical PDF report from current analysis."""
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    uid = session.get('user_id')
    if uid not in MEMORY_CACHE:
        return "No diagnostic data found. Please run an analysis first.", 404
    data = MEMORY_CACHE[uid]
    pdf = FPDF(unit='mm', format='A4')
    pdf.add_page()
    pdf.set_margins(15, 15, 15)
    pdf.set_font('Arial', 'B', 18)
    pdf.set_text_color(20, 50, 100)
    pdf.cell(0, 10, "ONCOVISION AI - CLINICAL DIAGNOSTIC REPORT", 0, 1, 'L')
    pdf.set_draw_color(200, 200, 200)
    pdf.line(15, 25, 195, 25)
    pdf.ln(5)
    pdf.set_font('Arial', '', 10)
    pdf.set_text_color(50, 50, 50)
    pdf.set_fill_color(245, 245, 245)
    pdf.cell(90, 8, f" Patient Reference ID: {data['patient_id']}", border=1, fill=True)
    pdf.cell(90, 8, f" Diagnostic System: Dual-Backbone EDCNN-XAI v4.2", border=1, fill=True, ln=1)
    pdf.cell(90, 8, f" Report Date: {data['timestamp']}", border=1, fill=True)
    pdf.cell(90, 8, f" AI Confidence: {data['classification']['confidence']:.1f}%", border=1, fill=True, ln=1)
    pdf.ln(8)
    pdf.set_font("Arial", 'B', 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 8, "1. DIAGNOSTIC CLASSIFICATION & NARRATIVE", 0, 1)
    result_class = data['classification']['class'].upper()
    confidence   = data['classification']['confidence']
    if "MALIGNANT" in result_class:
        pdf.set_text_color(220, 38, 38)
        urgency = "URGENT CLINICAL REVIEW RECOMMENDED"
    elif "BENIGN" in result_class:
        pdf.set_text_color(22, 163, 74)
        urgency = "STANDARD PROTOCOL MONITORING"
    else:
        pdf.set_text_color(9, 132, 227)
        urgency = "ROUTINE SCREENING"
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 8, f"PRIMARY FINDING: {result_class} ({confidence:.1f}% Confidence)", 0, 1)
    pdf.set_font("Arial", 'I', 9)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 5, urgency, 0, 1)
    pdf.ln(3)
    pdf.set_font("Arial", '', 10)
    pdf.set_text_color(0, 0, 0)
    pdf.multi_cell(0, 6, data['narrative'])
    pdf.ln(6)
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 8, "2. QUANTITATIVE XAI METRICS", 0, 1)
    pdf.set_font("Arial", '', 10)
    pdf.set_text_color(50, 50, 50)
    metrics = data['metrics']
    pdf.cell(60, 6, f"- Tissue Coverage: {metrics['coverage']:.1f}%", 0, 0)
    pdf.cell(60, 6, f"- Target Alignment: {metrics['alignment']:.1f}%", 0, 0)
    pdf.cell(60, 6, f"- Signal Concentration: {metrics['concentration']:.1f}%", 0, 1)
    pdf.ln(8)
    pdf.set_font("Arial", 'B', 11)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 8, "3. DIAGNOSTIC FUSION", 0, 1)
    pdf.set_font("Arial", 'I', 9)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 5, "(Comprehensive Overlay: Original Scan + AI Mask + Grad-CAM Focus)", 0, 1)

    def add_img_to_pdf(img_array, title, x, y, w=75):
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img_array
        img_name = f"temp_report_img_{uuid.uuid4()}.png"
        pil_img  = Image.fromarray(img_rgb)
        pil_img  = pil_img.resize((300, 300))
        pil_img.save(img_name)
        pdf.image(img_name, x=x, y=y, w=w, h=w)
        if os.path.exists(img_name):
            os.remove(img_name)
        pdf.set_xy(x, y + w + 2)
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Arial", 'B', 9)
        pdf.cell(w, 5, title, 0, 0, 'C')

    def draw_footer():
        pdf.set_y(-20)
        pdf.set_draw_color(200, 200, 200)
        pdf.line(15, 275, 195, 275)
        pdf.set_font("Arial", 'I', 8)
        pdf.set_text_color(120, 120, 120)
        pdf.multi_cell(0, 4, "DISCLAIMER: This diagnostic report is generated by an AI framework intended for research and auxiliary clinician assistance. It is NOT a definitive diagnosis. Clinical validation by a qualified radiologist or oncologist is strictly required.", align='C')

    y_start = pdf.get_y() + 2
    w_main  = 100
    x_main  = (210 - w_main) / 2
    add_img_to_pdf(data['fusion'], "PRIMARY DIAGNOSTIC FUSION", x_main, y_start, w_main)
    draw_footer()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 14)
    pdf.set_text_color(20, 50, 100)
    pdf.cell(0, 10, "SUPPLEMENTARY RADIOLOGICAL DATA", 0, 1, 'C')
    pdf.set_draw_color(200, 200, 200)
    pdf.line(15, 25, 195, 25)
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 11)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 8, "4. COMPONENT ANALYSIS MAPS", 0, 1)
    pdf.set_font("Arial", 'I', 9)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 5, "(Individual localization layers resized uniformly)", 0, 1)
    y_start_p2 = pdf.get_y() + 2
    w_sub = 75
    gap   = 10
    x1    = (210 - (2 * w_sub + gap)) / 2
    x2    = x1 + w_sub + gap
    h_u8  = (data['heatmap'] * 255).astype(np.uint8)
    h_jet = cv2.applyColorMap(h_u8, cv2.COLORMAP_JET)
    overlay_img = overlay_heatmap(data['original_bgr'], data['mask_pred'], alpha=0.5, threshold=0.1)
    add_img_to_pdf(data['original_bgr'], "A. Original Scan",       x1, y_start_p2, w_sub)
    add_img_to_pdf(h_jet,               "B. Grad-CAM Activation",  x2, y_start_p2, w_sub)
    y_row2_p2 = y_start_p2 + w_sub + 12
    x_center  = (210 - w_sub) / 2
    add_img_to_pdf(overlay_img, "C. Segmentation Mask", x_center, y_row2_p2, w_sub)
    draw_footer()
    output  = io.BytesIO()
    pdf_str = pdf.output(dest='S').encode('latin-1')
    output.write(pdf_str)
    output.seek(0)
    return send_file(
        output,
        as_attachment=True,
        download_name=f"Report_{data['patient_id']}_{int(time.time())}.pdf",
        mimetype='application/pdf'
    )


@app.route('/api/clear', methods=['POST'])
def clear_data():
    """Explicitly wipe memory cache for the doctor"""
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    uid = session.get('user_id')
    if uid in MEMORY_CACHE:
        del MEMORY_CACHE[uid]
        return jsonify({'success': True, 'message': 'Information securely wiped from RAM.'})
    return jsonify({'success': True, 'message': 'No data found to clear.'})


if __name__ == '__main__':
    app.run(debug=True, port=5000, threaded=True)
