# Use CUDA-enabled PyTorch image as base
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create cache directory to ensure proper permissions
RUN mkdir -p /root/.cache/huggingface

# Copy the application code
COPY . /app

# Expose the port the app runs on
EXPOSE 7007

# Command to run the application
CMD ["uvicorn", "api.index:app", "--host", "0.0.0.0", "--port", "7007"] 