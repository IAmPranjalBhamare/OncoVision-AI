import os
import io
import base64
import uuid
import time
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

# Global variables for models
model_edcnn = None
model_unet = None

def load_models_once():
    """Load models at startup into memory for fast inference"""
    global model_edcnn, model_unet
    import tensorflow as tf
    import os
    # Root dir is where app.py is
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    edcnn_keras_path = os.path.join(results_dir, "edcnn_best.keras")
    unet_keras_path = os.path.join(results_dir, "unet_best.keras")

    if model_edcnn is None:
        print("[System] Loading EDCNN model...")
        try:
            model_edcnn = tf.keras.models.load_model(edcnn_keras_path, compile=False)
        except Exception as e:
            print(f"[Warning] Could not load EDCNN weights: {e}")
            
    if model_unet is None:
        print("[System] Loading U-Net model...")
        try:
            model_unet = tf.keras.models.load_model(unet_keras_path, compile=False)
        except Exception as e:
            print(f"[Warning] Could not load U-Net weights: {e}")
    print("[System] Models loaded successfully and ready.")

# Initialize models
# Models are now lazily loaded on the first request to prevent startup timeouts
# load_models_once()


def array_to_base64(img_array):
    """Convert numpy array image (RGB/BGR/Grayscale) to base64 string"""
    # Normalize if needed
    if img_array.dtype != np.uint8:
        img_array = (img_array * 255).clip(0, 255).astype(np.uint8)
    
    # Convert OpenCV BGR to RGB if needed (assuming incoming is RGB for display)
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        # We usually expect RGB PIL arrays, lets just encode as PNG
        is_success, buffer = cv2.imencode(".png", cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
    elif len(img_array.shape) == 2 or img_array.shape[2] == 1:
        # Grayscale
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
    # Create an RGBA image where mask > 0 is red and opaque, else transparent
    colored = np.zeros((mask_u8.shape[0], mask_u8.shape[1], 4), dtype=np.uint8)
    colored[mask_u8 > 127] = [255, 0, 0, 180]  # Red, semi-transparent
    
    # Encode as PNG to keep alpha channel
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
    tumor_bin = (mask_u8 > 127).astype(np.uint8)
    tumor_area = tumor_bin.sum()
    
    coverage = (tumor_area / tissue_area * 100.0) if tissue_area > 0 else 0.0
    
    hotspot = (heatmap > 0.5).astype(np.uint8)
    hot_pixels = hotspot.sum()
    intersection = (hotspot * tumor_bin).sum()
    
    alignment = (intersection / hot_pixels * 100.0) if hot_pixels > 0 else 0.0
    concentration = max(0, min(100, (1.0 - (hot_pixels / tissue_area)) * 100.0)) if tissue_area > 0 else 50.0
    
    return {
        'coverage': float(coverage),
        'alignment': float(alignment),
        'concentration': float(concentration)
    }

def generate_narrative(class_name, conf, metrics):
    """Generate a clinical narrative explanation based on AI results"""
    c = class_name.lower()
    align = metrics['alignment']
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
        # Simple secure hardcoded auth for medical demo
        data = request.json
        if data and data.get('username') == 'doctor' and data.get('password') == 'secure123':
            session['logged_in'] = True
            session['user_id'] = str(uuid.uuid4())
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
        # Lazily load models if not already in memory
        load_models_once()
        
        # Load image strictly into RAM, do not save to disk
        file_bytes = file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Ensure it's read correctly
        if img_bgr is None:
            return jsonify({'error': 'Invalid image format.'}), 400
            
        # 1. Translate to RGB and heavily compress dimensions FIRST
        img_rgb_raw = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb_raw, (224, 224))
        
        # 2. Excecute BM3D Denoising exactly on the smaller footprint
        print("[Predict] Applying BM3D Denoising Filter...")
        img_resized_bm3d = apply_bm3d(img_resized)
        
        # Preprocess using same logic as data_loader._load_image (Z-score normalize)
        img_float = img_resized_bm3d.astype(np.float32)
        mean = img_float.mean()
        std = img_float.std() + 1e-8
        img_norm = (img_float - mean) / std
        model_input = np.expand_dims(img_norm, axis=0)
        
        print("[Predict] Running Segmentation...")
        mask_pred = predict_segmentation(model_unet, model_input)
        
        print("[Predict] Running Classification...")
        class_result = predict_classification(model_edcnn, model_input)
        
        print("[Predict] Running Targeted Grad-CAM (Radiological Steering)...")
        # Generate the heatmap steered by the original image's density (hypoechoic regions)
        heatmap_raw = get_saliency_heatmap(model_edcnn, model_input, class_result['class_idx'], 
                                           mask_pred=mask_pred, original_img=img_resized)
        
        # Create Diagnostic Fusion: original image + heatmap overlay with bounding box
        fusion_img = overlay_heatmap(img_resized, heatmap_raw, alpha=0.6, threshold=0.15)
        fusion_img = draw_tumor_bbox(fusion_img, heatmap_raw, threshold=0.5, color=(0, 255, 80), thickness=2)

        # ── Contribution-ranked images ──────────────────────────────────────
        # (e) Classification: rank ALL pixels across the full image by Grad-CAM activation
        cls_rank_img = create_ranked_overlay(img_resized, heatmap_raw, mask_pred=None)

        # (f) Segmentation: rank pixels within the segmented tumor region
        seg_mask_vis = mask_pred if mask_pred is not None \
                       else np.zeros(img_resized.shape[:2], dtype=np.uint8)
        seg_rank_img = create_segmentation_contribution_overlay(
            img_resized, seg_mask_vis, heatmap=heatmap_raw
        )
        
        # Explainability Metrics (using raw heatmap)
        metrics = compute_xai_metrics(mask_pred, heatmap_raw, img_resized_bm3d)
        
        narrative = generate_narrative(class_result['class'], class_result['confidence'], metrics)
        
        uid = session.get('user_id')
        patient_id = request.form.get('patient_id', 'Unknown')
        
        # Store comprehensive data for reporting
        MEMORY_CACHE[uid] = {
            'original_bgr': img_bgr,
            'mask_pred': mask_pred,
            'heatmap': heatmap_raw,
            'fusion': fusion_img,
            'classification': class_result,
            'metrics': metrics,
            'narrative': narrative,
            'patient_id': patient_id,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        processed_class_result = {
            'class': class_result['class'],
            'class_idx': int(class_result['class_idx']),
            'confidence': float(class_result['confidence']),
            'probabilities': {k: float(v) for k, v in class_result['probabilities'].items()}
        }
        
        response_data = {
            'classification': processed_class_result,
            'metrics': metrics,
            'narrative': narrative,
            'images': {
                'original': array_to_base64(img_resized),
                'bm3d_output': array_to_base64(img_resized_bm3d),
                'mask_overlay': create_mask_overlay_base64(mask_pred),
                'contour_overlay': create_contour_overlay_base64(mask_pred),
                'gradcam': apply_colormap_to_base64(heatmap_raw, cv2.COLORMAP_JET),
                'diagnostic_fusion': array_to_base64(fusion_img),
                'contribution_rank': array_to_base64(cls_rank_img),
                'seg_contribution_rank': array_to_base64(seg_rank_img),
            }
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error processing image: {e}")
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
    
    # Initialize PDF
    pdf = FPDF(unit='mm', format='A4')
    pdf.add_page()
    pdf.set_margins(15, 15, 15)
    
    # ── Header ──
    pdf.set_font('Arial', 'B', 18)
    pdf.set_text_color(20, 50, 100) # Dark Blue
    pdf.cell(0, 10, "ONCOVISION AI - CLINICAL DIAGNOSTIC REPORT", 0, 1, 'L')
    
    # Line separator
    pdf.set_draw_color(200, 200, 200)
    pdf.line(15, 25, 195, 25)
    pdf.ln(5)
    
    # ── Patient Meta Box ──
    pdf.set_font('Arial', '', 10)
    pdf.set_text_color(50, 50, 50)
    pdf.set_fill_color(245, 245, 245)
    
    pdf.cell(90, 8, f" Patient Reference ID: {data['patient_id']}", border=1, fill=True)
    pdf.cell(90, 8, f" Diagnostic System: Dual-Backbone EDCNN-XAI v4.2", border=1, fill=True, ln=1)
    
    pdf.cell(90, 8, f" Report Date: {data['timestamp']}", border=1, fill=True)
    pdf.cell(90, 8, f" AI Confidence: {data['classification']['confidence']:.1f}%", border=1, fill=True, ln=1)
    
    pdf.ln(8)
    
    # ── 1. Diagnostic Findings ──
    pdf.set_font("Arial", 'B', 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 8, "1. DIAGNOSTIC CLASSIFICATION & NARRATIVE", 0, 1)
    
    result_class = data['classification']['class'].upper()
    confidence = data['classification']['confidence']
    
    if "MALIGNANT" in result_class:
        pdf.set_text_color(220, 38, 38) # Red
        urgency = "URGENT CLINICAL REVIEW RECOMMENDED"
    elif "BENIGN" in result_class:
        pdf.set_text_color(22, 163, 74) # Green
        urgency = "STANDARD PROTOCOL MONITORING"
    else:
        pdf.set_text_color(9, 132, 227) # Blue
        urgency = "ROUTINE SCREENING"

    pdf.set_font("Arial", 'B', 14)    
    pdf.cell(0, 8, f"PRIMARY FINDING: {result_class} ({confidence:.1f}% Confidence)", 0, 1)
    pdf.set_font("Arial", 'I', 9)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 5, urgency, 0, 1)
    
    pdf.ln(3)
    
    # Narrative
    pdf.set_font("Arial", '', 10)
    pdf.set_text_color(0, 0, 0)
    pdf.multi_cell(0, 6, data['narrative'])
    pdf.ln(6)
    
    # Quantitative Metrics
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 8, "2. QUANTITATIVE XAI METRICS", 0, 1)
    pdf.set_font("Arial", '', 10)
    pdf.set_text_color(50, 50, 50)
    
    metrics = data['metrics']
    pdf.cell(60, 6, f"- Tissue Coverage: {metrics['coverage']:.1f}%", 0, 0)
    pdf.cell(60, 6, f"- Target Alignment: {metrics['alignment']:.1f}%", 0, 0)
    pdf.cell(60, 6, f"- Signal Concentration: {metrics['concentration']:.1f}%", 0, 1)
    
    pdf.ln(8)

    # ── Image Grid (Page 1) ──
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
        pil_img = Image.fromarray(img_rgb)
        pil_img = pil_img.resize((300, 300))
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

    # Primary Image (Centered on Page 1)
    y_start = pdf.get_y() + 2
    w_main = 100
    x_main = (210 - w_main) / 2
    add_img_to_pdf(data['fusion'], "PRIMARY DIAGNOSTIC FUSION", x_main, y_start, w_main)
    
    draw_footer()
    
    # ── PAGE 2 ──
    pdf.add_page()
    
    pdf.set_font('Arial', 'B', 14)
    pdf.set_text_color(20, 50, 100) # Dark Blue
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
    gap = 10
    x1 = (210 - (2 * w_sub + gap)) / 2
    x2 = x1 + w_sub + gap
    
    h_u8 = (data['heatmap'] * 255).astype(np.uint8)
    h_jet = cv2.applyColorMap(h_u8, cv2.COLORMAP_JET)
    overlay = overlay_heatmap(data['original_bgr'], data['mask_pred'], alpha=0.5, threshold=0.1)
    
    # Row 1 (Page 2)
    add_img_to_pdf(data['original_bgr'], "A. Original Scan", x1, y_start_p2, w_sub)
    add_img_to_pdf(h_jet, "B. Grad-CAM Activation", x2, y_start_p2, w_sub)
    
    # Row 2 (Centered on Page 2)
    y_row2_p2 = y_start_p2 + w_sub + 12
    x_center = (210 - w_sub) / 2
    add_img_to_pdf(overlay, "C. Segmentation Mask", x_center, y_row2_p2, w_sub)
    
    draw_footer()

    # Output as Byte stream
    output = io.BytesIO()
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
    # Threaded=False prevents TensorFlow threading crashes in dev serve
    app.run(debug=True, port=5000, threaded=False)
