FROM python:3.11-slim

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY image_processors.py .
COPY find_green.py .
COPY find_green_vid.py .
COPY find_red.py .
COPY Band-Tracker_VIDEO_CLIP_TEST.py .

# Copy test files
COPY Tests/ ./Tests/

# Set PYTHONPATH to ensure imports work correctly
ENV PYTHONPATH=/app

# Default command runs tests
CMD ["python", "-m", "pytest", "Tests/", "-v"]
