FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY src/ src/
COPY main.py .

# Create output directories
RUN mkdir -p Checkpoints Attention_Visualizations Evaluation Notebook

# Default command: train and evaluate all models
CMD ["python", "main.py"]
