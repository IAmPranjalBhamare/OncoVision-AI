FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libfontconfig1 \
    libfreetype6 \
    libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

COPY app.py predict_custom_image.py gradcam_xai.py ./
COPY models ./models
COPY utils ./utils
COPY templates ./templates
COPY static ./static
COPY results/edcnn_best.keras ./results/edcnn_best.keras
COPY results/unet_best.keras ./results/unet_best.keras
COPY start.sh ./start.sh
RUN chmod +x ./start.sh

CMD ["./start.sh"]
