FROM python:3.11-slim

# -------------------------------------------------------------------
# Working directory
# -------------------------------------------------------------------
WORKDIR /app

# -------------------------------------------------------------------
# System dependencies
# -------------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------------------------------------------
# Python dependencies
# -------------------------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# -------------------------------------------------------------------
# Copy source code
# -------------------------------------------------------------------
COPY src/ ./src/
COPY train_connect4_winprob.py .
# (add other entry points if you have them)

# -------------------------------------------------------------------
# Workspace directories (for volumes)
# -------------------------------------------------------------------
RUN mkdir -p \
    /workspace/datasets \
    /workspace/models \
    /workspace/logs \
    /workspace/tensorboard-logs \
    /workspace/mlflow-artifacts

VOLUME /workspace/datasets
VOLUME /workspace/models
VOLUME /workspace/logs
VOLUME /workspace/tensorboard-logs
VOLUME /workspace/mlflow-artifacts

# -------------------------------------------------------------------
# Default command (training)
# -------------------------------------------------------------------
CMD ["python", "train_connect4_winprob.py"]
