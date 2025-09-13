# SVOD - Smart Video Orientation Detector
# Multi-stage Docker build for optimal image size

# Build stage
FROM python:3.12-slim as builder

# Install system dependencies for OpenCV and other libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgtk2.0-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Download model files during build
RUN python -c "from video_orientation_detector import download_model_files; download_model_files()"

# Production stage
FROM python:3.12-slim as production

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libgomp1 \
    libgtk2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash svod

# Set working directory
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code and model files
COPY --from=builder /app /app

# Change ownership to non-root user
RUN chown -R svod:svod /app

# Switch to non-root user
USER svod

# Create volume for input/output files
VOLUME ["/data"]

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Expose port for potential web interface (future feature)
EXPOSE 8000

# Default command
CMD ["python", "video_orientation_detector.py", "--help"]