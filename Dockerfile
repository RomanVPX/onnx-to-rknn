FROM python:3.11-slim

# 1) System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    protobuf-compiler libprotobuf-dev \
    libjpeg-dev libpng-dev libopencv-dev \
  && rm -rf /var/lib/apt/lists/*

# 2) Python packages
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    pillow==11.0.0 \
    onnx==1.14.1 \
    rknn-toolkit2==2.3.0 \
    requests

# 3) Workspace
WORKDIR /workspace

# 4) Scripts
COPY scripts/convert.py .

# 5) Entry point
ENTRYPOINT ["python", "convert.py"]
