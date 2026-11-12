# Samsung XR Hackathon 2026 - Track 02: EchoGesture
# Reproducible execution environment for Python 3.11
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLCONFIGDIR=/tmp

WORKDIR /workspace

# Install system libraries required by OpenCV headless and ONNX Runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire source package
COPY . /workspace/src/

WORKDIR /workspace/src

# Verification entrypoint
CMD ["python3", "-c", "from main import predict; print('EchoGesture Container Initialized Successfully')"]

