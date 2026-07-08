// main.js v5 — Advanced CAD Diagnostics + Progressive Inference Polling

document.addEventListener('DOMContentLoaded', () => {

    // ─── DOM References ─────────────────────────────────────────────────────
    const dropZone     = document.getElementById('dropZone');
    const fileInput    = document.getElementById('fileInput');
    const patientIdIn  = document.getElementById('patientIdInput');
    const uploadView   = document.getElementById('uploadView');
    const loadingState = document.getElementById('loadingState');
    const resultsView  = document.getElementById('resultsView');

    let isCompareMode  = false;
    let zoomLevel      = 1.0;

    // ─── Drag & Drop ────────────────────────────────────────────────────────
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
    });
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) handleFile(e.target.files[0]);
    });

    // ─── Upload & Predict ───────────────────────────────────────────────────
    async function handleFile(file) {
        if (!file.type.match('image.*')) {
            alert('Please upload an image file (PNG/JPG)');
            return;
        }
        const patientId = patientIdIn.value.trim() || "Unknown";
        dropZone.classList.add('hidden');
        loadingState.classList.remove('hidden');

        // Update loading UI with step text
        const loadingTitle = loadingState.querySelector('h3');
        const loadingMsg   = loadingState.querySelector('p');

        // ── Add a progress bar to the loading state if not already there ──
        if (!document.getElementById('inferenceProgress')) {
            const progressWrap = document.createElement('div');
            progressWrap.style.cssText = 'width:260px;margin:1.2rem auto 0;background:rgba(255,255,255,0.08);border-radius:8px;overflow:hidden;height:6px;';
            const progressBar = document.createElement('div');
            progressBar.id = 'inferenceProgress';
            progressBar.style.cssText = 'height:6px;width:0%;background:linear-gradient(90deg,#0984e3,#00d26a);border-radius:8px;transition:width 0.4s ease;';
            progressWrap.appendChild(progressBar);
            loadingState.appendChild(progressWrap);
        }
        const progressBar = document.getElementById('inferenceProgress');

        function setStep(stepText, pct) {
            if (loadingTitle) loadingTitle.textContent = stepText;
            if (progressBar)  progressBar.style.width  = pct + '%';
        }

        // ── Check if models are ready first ───────────────────────────────
        setStep('Checking AI system status…', 5);
        try {
            const statusRes  = await fetch('/api/status');
            const statusData = await statusRes.json();
            if (!statusData.ready) {
                setStep('Models warming up — please wait…', 8);
                if (loadingMsg) loadingMsg.textContent = 'The AI models are loading for the first time. This takes ~60s once.';
                // Poll until models are ready
                await new Promise((resolve, reject) => {
                    const pollId = setInterval(async () => {
                        try {
                            const r = await fetch('/api/status');
                            const d = await r.json();
                            if (d.ready) { clearInterval(pollId); resolve(); }
                        } catch { clearInterval(pollId); reject(new Error('Status check failed')); }
                    }, 3000);
                });
            }
        } catch (e) {
            // If status endpoint fails, proceed anyway (old server fallback)
        }

        // ── Start async inference job ──────────────────────────────────────
        setStep('Uploading scan…', 10);
        if (loadingMsg) loadingMsg.textContent = 'Sending image to AI pipeline…';

        const formData = new FormData();
        formData.append('file', file);
        formData.append('patient_id', patientId);

        try {
            // Try the new async endpoint first
            const startRes = await fetch('/api/predict/start', { method: 'POST', body: formData });

            if (startRes.status === 503) {
                // Models still loading — wait and retry once
                setStep('Models warming up…', 12);
                await new Promise(r => setTimeout(r, 5000));
                const retryRes = await fetch('/api/predict/start', { method: 'POST', body: formData });
                if (!retryRes.ok) throw new Error('Server not ready. Please try again in a moment.');
                const retryData = await retryRes.json();
                if (retryData.error) throw new Error(retryData.error);
                await pollTaskUntilDone(retryData.task_id, setStep);
                return;
            }

            if (!startRes.ok) {
                // Fall back to synchronous endpoint
                setStep('Running AI analysis…', 20);
                if (loadingMsg) loadingMsg.textContent = 'Processing — this may take up to 60 seconds…';
                const syncRes = await fetch('/api/predict', { method: 'POST', body: formData });
                if (!syncRes.ok) throw new Error('Server error during processing. Please try again.');
                const data = await syncRes.json();
                if (data.error) throw new Error(data.error);
                populateResults(data);
                uploadView.classList.add('hidden');
                resultsView.classList.remove('hidden');
                return;
            }

            const startData = await startRes.json();
            if (startData.error) throw new Error(startData.error);

            await pollTaskUntilDone(startData.task_id, setStep);

        } catch (error) {
            alert(error.message);
            loadingState.classList.add('hidden');
            dropZone.classList.remove('hidden');
        }
    }

    // ── Poll /api/predict/status/<task_id> until done ─────────────────────
    async function pollTaskUntilDone(task_id, setStep) {
        const loadingMsg = loadingState.querySelector('p');

        return new Promise((resolve, reject) => {
            const pollId = setInterval(async () => {
                try {
                    const res  = await fetch(`/api/predict/status/${task_id}`);
                    if (!res.ok) { clearInterval(pollId); reject(new Error('Polling failed')); return; }
                    const data = await res.json();

                    if (data.error && data.status !== 'running') {
                        clearInterval(pollId);
                        reject(new Error(data.error || 'Unknown server error'));
                        return;
                    }

                    // Update progress UI
                    setStep(data.step || 'Processing…', data.progress || 0);
                    if (loadingMsg) loadingMsg.textContent = getStepSubtitle(data.step);

                    if (data.status === 'done') {
                        clearInterval(pollId);
                        setStep('Rendering results…', 100);
                        populateResults(data.result);
                        uploadView.classList.add('hidden');
                        resultsView.classList.remove('hidden');
                        resolve();
                    } else if (data.status === 'error') {
                        clearInterval(pollId);
                        reject(new Error(data.error || 'Inference failed'));
                    }
                } catch (e) {
                    clearInterval(pollId);
                    reject(e);
                }
            }, 800);
        });
    }

    function getStepSubtitle(step) {
        const map = {
            'Starting inference…':            'Initialising AI pipeline…',
            'Decoding image…':                'Reading and validating the uploaded scan…',
            'Denoising image…':               'Applying noise reduction filter…',
            'Running U-Net segmentation…':    'Detecting and outlining tumor regions…',
            'Running EDCNN classification…':  'Classifying tissue — Benign / Malignant / Normal…',
            'Generating Grad-CAM XAI maps…':  'Computing gradient-weighted attention heatmaps…',
            'Building diagnostic overlays…':  'Compositing final diagnostic fusion images…',
            'Computing XAI analytics…':       'Calculating coverage, alignment, and signal metrics…',
            'Encoding results…':              'Preparing images for display…',
        };
        return map[step] || 'Running AI pipeline…';
    }

    // ─── Comparison Mode ────────────────────────────────────────────────────
    window.toggleCompareMode = function () {
        // Mutually exclusive with Triple View
        if (document.getElementById('tripleVisualizer').classList.contains('hidden') === false) {
            toggleTripleView();
        }

        isCompareMode = !isCompareMode;
        const panel      = document.getElementById('visualizerPanel');
        const compareViz = document.getElementById('compareVisualizer');
        const btn        = document.getElementById('btnToggleCompare');
        if (isCompareMode) {
            panel.classList.add('compare-active');
            compareViz.classList.remove('hidden');
            btn.classList.add('active');
        } else {
            panel.classList.remove('compare-active');
            compareViz.classList.add('hidden');
            btn.classList.remove('active');
        }
    };

    window.toggleTripleView = function () {
        const tripleViz = document.getElementById('tripleVisualizer');
        const mainViz   = document.getElementById('viewerContainer');
        const btn       = document.getElementById('btnToggleTriple');
        
        const isHidden  = tripleViz.classList.contains('hidden');
        
        if (isHidden) {
            // Turning ON
            // Mutually exclusive with Compare Mode
            if (isCompareMode) toggleCompareMode();

            tripleViz.classList.remove('hidden');
            mainViz.classList.add('hidden');
            btn.classList.add('active');
            
            // Sync images
            document.getElementById('tripleImgOriginal').src  = document.getElementById('imgOriginal').src;
            
            // Pane 2: Segmentation Overlay
            document.getElementById('tripleImgOriginal2').src = document.getElementById('imgOriginal').src;
            document.getElementById('tripleImgMask').src      = document.getElementById('imgMask').src;
            
            // Pane 3: Diagnostic Fusion
            // Note: tripleImgFusion is populated in populateResults from data.images.diagnostic_fusion
        } else {
            // Turning OFF
            tripleViz.classList.add('hidden');
            mainViz.classList.remove('hidden');
            btn.classList.remove('active');
        }
    };

    window.handleCompareFile = async function (input) {
        const file = input.files[0];
        if (!file) return;
        const placeholder = document.getElementById('comparePlaceholder');
        placeholder.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Analyzing history...';

        const formData = new FormData();
        formData.append('file', file);
        formData.append('patient_id', "HISTORICAL_REF");

        try {
            const res  = await fetch('/api/predict', { method: 'POST', body: formData });
            const data = await res.json();

            placeholder.classList.add('hidden');
            document.getElementById('imgOriginalCompare').src = data.images.original;
            document.getElementById('imgBm3dCompare').src     = data.images.bm3d_output;
            document.getElementById('imgOriginalCompare').classList.remove('hidden');
            document.getElementById('imgOriginalContext1Compare').src = data.images.original;
            document.getElementById('imgMaskCompare').src             = data.images.mask_overlay;
            document.getElementById('imgOriginalContext3Compare').src = data.images.original;
            document.getElementById('imgContourCompare').src          = data.images.contour_overlay;
            document.getElementById('imgOriginalContext2Compare').src = data.images.original;
            document.getElementById('imgGradcamCompare').src          = data.images.gradcam;
            syncCompareView();
        } catch (e) {
            alert("Error analyzing historical scan: " + e.message);
            placeholder.innerHTML = '<button class="btn btn-outline-primary" onclick="document.getElementById(\'compareInput\').click()">Retry historical scan</button>';
        }
    };

    function syncCompareView() {
        const activeBtn = document.querySelector('#viewerContainer .tool-btn.active');
        const activeView = activeBtn ? activeBtn.getAttribute('data-view') : 'original';
        switchCompareView(activeView);
    }

    // ─── Results Population ─────────────────────────────────────────────────
    function populateResults(data) {
        const cls  = data.classification.class.toLowerCase();
        const conf = data.classification.confidence;

        const primaryEl = document.getElementById('primaryPrediction');
        primaryEl.innerText   = cls.toUpperCase();
        primaryEl.className   = `prediction-large ${cls}`;
        document.getElementById('confidenceText').innerText = `Confidence: ${conf.toFixed(1)}%`;

        const barColor = cls === 'malignant' ? '#ff4757' : (cls === 'benign' ? '#00d26a' : '#0984e3');
        const fill     = document.getElementById('confidenceFill');
        fill.style.width      = '0%';
        fill.style.background = barColor;
        setTimeout(() => { fill.style.transition = 'width 1s ease'; fill.style.width = conf.toFixed(1) + '%'; }, 50);

        // Class Probability Bars
        const probs      = data.classification.probabilities;
        const probList   = document.getElementById('probList');
        probList.innerHTML = '';
        const sortedProbs = Object.entries(probs).sort((a, b) => b[1] - a[1]);
        sortedProbs.forEach(([c_name, p_val]) => {
            const pStr = p_val.toFixed(1) + '%';
            const isTop = c_name.toLowerCase() === cls;
            probList.innerHTML += `
                <div class="prob-row">
                    <div style="width:35%;font-weight:${isTop ? '700' : '400'}" class="prob-label">${c_name}</div>
                    <div class="prob-bar-bg" style="width:45%">
                        <div class="prob-bar-fill" style="width:${pStr}; background:${isTop ? barColor : '#3a3a5c'};"></div>
                    </div>
                    <div style="width:20%;text-align:right;" class="prob-val">${pStr}</div>
                </div>`;
        });

        // XAI Metrics
        if (data.metrics) {
            const { coverage, alignment, concentration } = data.metrics;
            setMetric('metricCoverage',      coverage.toFixed(1) + '%',      coverage > 40 ? 'alert' : coverage > 15 ? 'warn' : 'good');
            setMetric('metricAlignment',     alignment.toFixed(1) + '%',     alignment > 70 ? 'good' : alignment > 40 ? 'warn' : 'alert');
            setMetric('metricConcentration', concentration.toFixed(1) + '%', concentration > 70 ? 'good' : concentration > 40 ? 'warn' : 'alert');
        }

        // Load images into viewer layers
        document.getElementById('imgOriginal').src         = data.images.original;
        document.getElementById('imgBm3d').src             = data.images.bm3d_output;
        document.getElementById('imgOriginalContext1').src = data.images.original;
        document.getElementById('imgMask').src             = data.images.mask_overlay;
        document.getElementById('imgOriginalContext3').src = data.images.original;
        document.getElementById('imgContour').src          = data.images.contour_overlay;
        document.getElementById('imgOriginalContext2').src = data.images.original;
        document.getElementById('imgGradcam').src          = data.images.gradcam;

        document.getElementById('tripleImgOriginal').src  = data.images.original;
        document.getElementById('tripleImgOriginal2').src = data.images.original;
        document.getElementById('tripleImgMask').src      = data.images.mask_overlay;
        document.getElementById('tripleImgFusion').src    = data.images.diagnostic_fusion;

        // ── Contribution Rank panels ──────────────────────────────────────────
        if (data.images.contribution_rank) {
            document.getElementById('imgClsRank').src = data.images.contribution_rank;
            document.getElementById('xaiRankSection').classList.remove('hidden');
        }
        if (data.images.seg_contribution_rank) {
            document.getElementById('imgSegRank').src = data.images.seg_contribution_rank;
        }

        switchView('original');
        // Apply current CAD filter state to newly loaded images
        applyFilters();
    }

    function setMetric(id, valText, level) {
        const el   = document.getElementById(id);
        el.innerText  = valText;
        el.className  = `metric-${level}`;
    }

    // ─── Visualizer Toolbar ─────────────────────────────────────────────────
    const mainToolBtns = document.querySelectorAll('#viewerContainer .tool-btn');
    const compToolBtns = document.querySelectorAll('#compareVisualizer .tool-btn');

    mainToolBtns.forEach(btn => {
        btn.addEventListener('click', (e) => {
            const view = e.target.getAttribute('data-view');
            switchView(view);
            if (isCompareMode) switchCompareView(view);
        });
    });
    compToolBtns.forEach(btn => {
        btn.addEventListener('click', (e) => {
            switchCompareView(e.target.getAttribute('data-view'));
        });
    });

    function switchView(view) {
        mainToolBtns.forEach(b => b.classList.remove('active'));
        const targetBtn = document.querySelector(`#viewerContainer .tool-btn[data-view="${view}"]`);
        if (targetBtn) targetBtn.classList.add('active');

        document.getElementById('imgOriginal').classList.add('hidden');
        document.getElementById('imgBm3d').classList.add('hidden');
        document.getElementById('overlaySegmentation').classList.add('hidden');
        document.getElementById('overlayContour').classList.add('hidden');
        document.getElementById('overlayGradcam').classList.add('hidden');

        if (view === 'original')     document.getElementById('imgOriginal').classList.remove('hidden');
        if (view === 'bm3d')         document.getElementById('imgBm3d').classList.remove('hidden');
        if (view === 'segmentation') document.getElementById('overlaySegmentation').classList.remove('hidden');
        if (view === 'contour')      document.getElementById('overlayContour').classList.remove('hidden');
        if (view === 'gradcam')      document.getElementById('overlayGradcam').classList.remove('hidden');

        // Re-apply filters after view switch
        applyFilters();
    }

    function switchCompareView(view) {
        compToolBtns.forEach(b => b.classList.remove('active'));
        const activeBtn = document.querySelector(`#compareVisualizer .tool-btn[data-view="${view}"]`);
        if (activeBtn) activeBtn.classList.add('active');

        document.getElementById('imgOriginalCompare').classList.add('hidden');
        document.getElementById('imgBm3dCompare').classList.add('hidden');
        document.getElementById('overlaySegmentationCompare').classList.add('hidden');
        document.getElementById('overlayContourCompare').classList.add('hidden');
        document.getElementById('overlayGradcamCompare').classList.add('hidden');

        if (view === 'original')     document.getElementById('imgOriginalCompare').classList.remove('hidden');
        if (view === 'bm3d')         document.getElementById('imgBm3dCompare').classList.remove('hidden');
        if (view === 'segmentation') document.getElementById('overlaySegmentationCompare').classList.remove('hidden');
        if (view === 'contour')      document.getElementById('overlayContourCompare').classList.remove('hidden');
        if (view === 'gradcam')      document.getElementById('overlayGradcamCompare').classList.remove('hidden');
    }

    // ─── Zoom & Pan ─────────────────────────────────────────────────────────
    window.zoom = function (delta) {
        zoomLevel = Math.min(Math.max(zoomLevel + delta, 0.5), 5.0);
        applyTransform();
    };
    window.resetZoom = function () {
        zoomLevel = 1.0;
        applyTransform();
    };

    function applyTransform() {
        const containers = [document.getElementById('container1'), document.getElementById('container2')];
        containers.forEach(c => {
            if (!c) return;
            c.querySelectorAll('.viewer-img').forEach(img => {
                const sx = cadState.flipH ? -zoomLevel : zoomLevel;
                const sy = cadState.flipV ? -zoomLevel : zoomLevel;
                img.style.transform = `scale(${sx}, ${sy})`;
            });
        });
    }

    // ─── Opacity Sliders ────────────────────────────────────────────────────
    document.getElementById('sliderMask').addEventListener('input', (e) => {
        const val = e.target.value / 100;
        document.getElementById('imgMask').style.opacity = val;
        const cmp = document.getElementById('imgMaskCompare');
        if (cmp) cmp.style.opacity = val;
    });
    document.getElementById('sliderContour').addEventListener('input', (e) => {
        const val = e.target.value / 100;
        document.getElementById('imgContour').style.opacity = val;
        const cmp = document.getElementById('imgContourCompare');
        if (cmp) cmp.style.opacity = val;
    });
    document.getElementById('sliderGradcam').addEventListener('input', (e) => {
        const val = e.target.value / 100;
        document.getElementById('imgGradcam').style.opacity = val;
        const cmp = document.getElementById('imgGradcamCompare');
        if (cmp) cmp.style.opacity = val;
    });


    // ═══════════════════════════════════════════════════════════════════════
    // ─── CAD IMAGE FILTER ENGINE ────────────────────────────────────────────
    // ═══════════════════════════════════════════════════════════════════════

    const cadState = {
        brightness:  100,
        contrast:    100,
        saturation:  100,
        hue:         0,
        opacity:     100,
        invert:      false,
        grayscale:   false,
        sharpen:     false,
        edge:        false,
        flipH:       false,
        flipV:       false,
    };

    /** Build the CSS filter string from current CAD state */
    function buildFilterString() {
        let f = '';
        f += `brightness(${cadState.brightness}%) `;
        f += `contrast(${cadState.contrast}%) `;
        f += `saturate(${cadState.saturation}%) `;
        if (cadState.hue !== 0) f += `hue-rotate(${cadState.hue}deg) `;
        if (cadState.invert)    f += `invert(100%) `;
        if (cadState.grayscale) f += `grayscale(100%) `;
        if (cadState.edge)      f += `url(#cad-edge) `;
        else if (cadState.sharpen) f += `url(#cad-sharpen) `;
        return f.trim();
    }

    /** Apply filter + opacity + flip transform to all viewer images in container1 */
    function applyFilters() {
        const filterStr = buildFilterString();
        const opacityVal = cadState.opacity / 100;
        // Target all image layers in the main viewer
        const targets = document.querySelectorAll('#container1 .viewer-img');
        targets.forEach(img => {
            img.style.filter  = filterStr;
            img.style.opacity = opacityVal;
        });
        // Sync transform (flips + zoom)
        applyTransform();
    }

    // ── Slider: Brightness
    document.getElementById('cadBrightness').addEventListener('input', (e) => {
        cadState.brightness = parseInt(e.target.value);
        document.getElementById('cadBrightnessVal').textContent = e.target.value + '%';
        applyFilters();
    });

    // ── Slider: Contrast
    document.getElementById('cadContrast').addEventListener('input', (e) => {
        cadState.contrast = parseInt(e.target.value);
        document.getElementById('cadContrastVal').textContent = e.target.value + '%';
        applyFilters();
    });

    // ── Slider: Saturation
    document.getElementById('cadSaturation').addEventListener('input', (e) => {
        cadState.saturation = parseInt(e.target.value);
        document.getElementById('cadSaturationVal').textContent = e.target.value + '%';
        applyFilters();
    });

    // ── Slider: Hue / False Color
    document.getElementById('cadHue').addEventListener('input', (e) => {
        cadState.hue = parseInt(e.target.value);
        document.getElementById('cadHueVal').textContent = e.target.value + '°';
        applyFilters();
    });

    // ── Slider: Global Image Opacity
    document.getElementById('cadOpacity').addEventListener('input', (e) => {
        cadState.opacity = parseInt(e.target.value);
        document.getElementById('cadOpacityVal').textContent = e.target.value + '%';
        applyFilters();
    });

    // ── Toggle: Invert
    window.toggleCadInvert = function () {
        cadState.invert = !cadState.invert;
        document.getElementById('cadInvertBtn').classList.toggle('active', cadState.invert);
        applyFilters();
    };

    // ── Toggle: Grayscale
    window.toggleCadGrayscale = function () {
        cadState.grayscale = !cadState.grayscale;
        document.getElementById('cadGrayscaleBtn').classList.toggle('active', cadState.grayscale);
        applyFilters();
    };

    // ── Toggle: Sharpen (mild kernel)
    window.toggleCadSharpen = function () {
        cadState.sharpen = !cadState.sharpen;
        if (cadState.sharpen) cadState.edge = false;  // mutually exclusive with edge
        document.getElementById('cadSharpenBtn').classList.toggle('active', cadState.sharpen);
        document.getElementById('cadEdgeBtn').classList.remove('active');
        applyFilters();
    };

    // ── Toggle: Edge Boost (strong Laplacian kernel)
    window.toggleCadEdge = function () {
        cadState.edge = !cadState.edge;
        if (cadState.edge) cadState.sharpen = false;  // mutually exclusive with sharpen
        document.getElementById('cadEdgeBtn').classList.toggle('active', cadState.edge);
        document.getElementById('cadSharpenBtn').classList.remove('active');
        applyFilters();
    };

    // ── Toggle: Flip Horizontal
    window.toggleCadFlipH = function () {
        cadState.flipH = !cadState.flipH;
        document.getElementById('cadFlipHBtn').classList.toggle('active', cadState.flipH);
        applyTransform();
    };

    // ── Toggle: Flip Vertical
    window.toggleCadFlipV = function () {
        cadState.flipV = !cadState.flipV;
        document.getElementById('cadFlipVBtn').classList.toggle('active', cadState.flipV);
        applyTransform();
    };

    // ── Reset All CAD Filters
    window.resetCadFilters = function () {
        cadState.brightness = 100;
        cadState.contrast   = 100;
        cadState.saturation = 100;
        cadState.hue        = 0;
        cadState.opacity    = 100;
        cadState.invert     = false;
        cadState.grayscale  = false;
        cadState.sharpen    = false;
        cadState.edge       = false;
        cadState.flipH      = false;
        cadState.flipV      = false;

        document.getElementById('cadBrightness').value = 100;
        document.getElementById('cadContrast').value   = 100;
        document.getElementById('cadSaturation').value = 100;
        document.getElementById('cadHue').value        = 0;
        document.getElementById('cadOpacity').value    = 100;
        document.getElementById('cadBrightnessVal').textContent = '100%';
        document.getElementById('cadContrastVal').textContent   = '100%';
        document.getElementById('cadSaturationVal').textContent = '100%';
        document.getElementById('cadHueVal').textContent        = '0°';
        document.getElementById('cadOpacityVal').textContent    = '100%';

        ['cadInvertBtn','cadGrayscaleBtn','cadSharpenBtn','cadEdgeBtn','cadFlipHBtn','cadFlipVBtn'].forEach(id => {
            const el = document.getElementById(id);
            if (el) el.classList.remove('active');
        });

        applyFilters();
    };


    // ═══════════════════════════════════════════════════════════════════════
    // ─── CALIPER MEASUREMENT TOOL ───────────────────────────────────────────
    // ═══════════════════════════════════════════════════════════════════════

    let isMeasuring   = false;
    let measureStart  = null;
    let isDrawing     = false;

    window.toggleMeasure = function () {
        isMeasuring = !isMeasuring;
        const canvas  = document.getElementById('measureCanvas');
        const btn     = document.getElementById('cadMeasureBtn');
        canvas.classList.toggle('active', isMeasuring);
        btn.classList.toggle('active', isMeasuring);
        if (!isMeasuring) {
            clearMeasure();
        }
    };

    window.clearMeasureCanvas = function () {
        clearMeasure();
    };

    function clearMeasure() {
        const canvas = document.getElementById('measureCanvas');
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        const badge = document.getElementById('measureBadge');
        if (badge) badge.style.display = 'none';
        measureStart = null;
        isDrawing    = false;
    }

    function initMeasureCanvas() {
        const canvas    = document.getElementById('measureCanvas');
        const container = document.getElementById('container1');
        if (!canvas || !container) return;

        function resizeCanvas() {
            canvas.width  = container.clientWidth;
            canvas.height = container.clientHeight;
        }
        resizeCanvas();
        new ResizeObserver(resizeCanvas).observe(container);

        canvas.addEventListener('mousedown', (e) => {
            if (!isMeasuring) return;
            isDrawing = true;
            const rect   = canvas.getBoundingClientRect();
            measureStart = {
                x: e.clientX - rect.left,
                y: e.clientY - rect.top
            };
        });

        canvas.addEventListener('mousemove', (e) => {
            if (!isMeasuring || !isDrawing || !measureStart) return;
            const rect = canvas.getBoundingClientRect();
            const end  = { x: e.clientX - rect.left, y: e.clientY - rect.top };
            drawMeasureLine(measureStart, end, false);
        });

        canvas.addEventListener('mouseup', (e) => {
            if (!isMeasuring || !measureStart) return;
            isDrawing    = false;
            const rect   = canvas.getBoundingClientRect();
            const end    = { x: e.clientX - rect.left, y: e.clientY - rect.top };
            drawMeasureLine(measureStart, end, true);
            measureStart = null;   // allow next measurement to layer on top
        });

        // Touch support
        canvas.addEventListener('touchstart', (e) => {
            if (!isMeasuring) return;
            e.preventDefault();
            const t = e.touches[0];
            const rect = canvas.getBoundingClientRect();
            measureStart = { x: t.clientX - rect.left, y: t.clientY - rect.top };
            isDrawing    = true;
        }, { passive: false });

        canvas.addEventListener('touchmove', (e) => {
            if (!isMeasuring || !isDrawing || !measureStart) return;
            e.preventDefault();
            const t    = e.touches[0];
            const rect = canvas.getBoundingClientRect();
            drawMeasureLine(measureStart, { x: t.clientX - rect.left, y: t.clientY - rect.top }, false);
        }, { passive: false });

        canvas.addEventListener('touchend', (e) => {
            if (!isMeasuring || !measureStart) return;
            isDrawing    = false;
            const t      = e.changedTouches[0];
            const rect   = canvas.getBoundingClientRect();
            drawMeasureLine(measureStart, { x: t.clientX - rect.left, y: t.clientY - rect.top }, true);
            measureStart = null;
        });
    }

    function drawMeasureLine(start, end, isFinal) {
        const canvas = document.getElementById('measureCanvas');
        if (!canvas) return;
        const ctx = canvas.getContext('2d');

        // Don't erase previous finalised measurements — only redraw the live preview
        if (!isFinal) {
            // Save and restore only the region being updated
            ctx.clearRect(0, 0, canvas.width, canvas.height);
        }

        const dx      = end.x - start.x;
        const dy      = end.y - start.y;
        const dist    = Math.sqrt(dx * dx + dy * dy).toFixed(1);
        const angle   = Math.atan2(dy, dx);

        // ── Draw dashed stem line
        ctx.beginPath();
        ctx.moveTo(start.x, start.y);
        ctx.lineTo(end.x, end.y);
        ctx.strokeStyle = '#00d26a';
        ctx.lineWidth   = 2;
        ctx.setLineDash([6, 4]);
        ctx.stroke();
        ctx.setLineDash([]);

        // ── Draw perpendicular tick marks at endpoints (caliper style)
        const TICK = 8;
        drawTick(ctx, start, angle, TICK);
        drawTick(ctx, end,   angle, TICK);

        // ── Endpoint dots
        [start, end].forEach(pt => {
            ctx.beginPath();
            ctx.arc(pt.x, pt.y, 5, 0, Math.PI * 2);
            ctx.fillStyle    = '#00d26a';
            ctx.shadowColor  = '#00d26a';
            ctx.shadowBlur   = 8;
            ctx.fill();
            ctx.shadowBlur   = 0;
        });

        // ── Distance label at midpoint
        const mx   = (start.x + end.x) / 2;
        const my   = (start.y + end.y) / 2 - 14;
        ctx.font        = 'bold 12px "Inter", sans-serif';
        ctx.textAlign   = 'center';
        ctx.fillStyle   = '#001';
        ctx.strokeStyle = '#00d26a';
        ctx.lineWidth   = 3;
        ctx.strokeText(`${dist} px`, mx, my);
        ctx.fillStyle   = '#ffffff';
        ctx.fillText(`${dist} px`, mx, my);

        // ── Update badge
        const badge = document.getElementById('measureBadge');
        if (badge) {
            badge.textContent = `📏 ${dist} px  |  ${(dist * 0.264).toFixed(2)} mm (est.)`;
            badge.style.display = 'block';
        }
    }

    /** Draw a perpendicular tick at a point along a line defined by its angle. */
    function drawTick(ctx, pt, angle, size) {
        const perp = angle + Math.PI / 2;
        ctx.beginPath();
        ctx.moveTo(pt.x + Math.cos(perp) * size, pt.y + Math.sin(perp) * size);
        ctx.lineTo(pt.x - Math.cos(perp) * size, pt.y - Math.sin(perp) * size);
        ctx.strokeStyle = '#00d26a';
        ctx.lineWidth   = 2;
        ctx.stroke();
    }

    // Initialise the canvas
    initMeasureCanvas();

}); // end DOMContentLoaded


// ─── Session Actions ─────────────────────────────────────────────────────────
async function downloadReport() {
    window.location.href = '/api/generate_report';
}

async function clearData() {
    if (confirm('Done analyzing? This will securely wipe the image from memory and return to the upload screen.')) {
        const res = await fetch('/api/clear', { method: 'POST' });
        if (res.ok) window.location.reload();
    }
}

async function logoutSession() {
    const res = await fetch('/logout', { method: 'POST' });
    if (res.ok) window.location.href = '/login';
}
