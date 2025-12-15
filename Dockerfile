# Base image with PyTorch and CUDA support
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage caching
COPY requirements.txt .

# Install Python dependencies
# Use Tsinghua mirror for speed in China if needed, or default
RUN pip install --no-cache-dir -r requirements.txt fastapi uvicorn python-multipart

# Copy project files
COPY . .

# Expose API port
EXPOSE 8000

# Run the API server
CMD ["python", "api_server.py"]
