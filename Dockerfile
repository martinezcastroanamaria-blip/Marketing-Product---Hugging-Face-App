FROM python:3.13.5-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    tesseract-ocr \
    libtesseract-dev \
    libleptonica-dev \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Haar cascade
RUN mkdir -p /app/data && \
    wget -q -O /app/data/haarcascade_frontalface_default.xml \
      https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml

# Copy requirements and code
COPY requirements.txt ./
COPY src/ ./src/

# Install Python dependencies
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

# Environment variables for OCR and face detection
ENV TESSERACT_CMD=/usr/bin/tesseract
ENV CASCADE_PATH=/app/data/haarcascade_frontalface_default.xml

# Expose Streamlit port
EXPOSE 8501

# Healthcheck
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run app
ENTRYPOINT ["streamlit", "run", "src/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
