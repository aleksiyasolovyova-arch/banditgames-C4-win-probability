FROM python:3.10-slim

# -------------------------
# Working directory
# -------------------------
WORKDIR /app

# -------------------------
# System dependencies
# -------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# -------------------------
# Python dependencies
# -------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# -------------------------
# Application source
# -------------------------
COPY src/ /app/src/
COPY continuous_learning.py /app/continuous_learning.py

# -------------------------
# Runtime directories (Docker volumes)
# -------------------------
RUN mkdir -p \
    /workspace/datasets \
    /workspace/models

# -------------------------
# Environment
# -------------------------
ENV PYTHONPATH=/app

# -------------------------
# Default command: Continuous Learning Watcher
# -------------------------
CMD ["python", "continuous_learning.py"]
