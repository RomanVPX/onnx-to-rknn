FROM python:3.11-slim

# 1) Системные зависимости (кэшируются)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    protobuf-compiler libprotobuf-dev \
    libjpeg-dev libpng-dev libopencv-dev \
  && rm -rf /var/lib/apt/lists/*

# 2) Python-пакеты (кэшируются)
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    pillow==11.0.0 \
    onnx==1.14.1 \
    rknn-toolkit2==2.3.0 \
    requests

# 3) Рабочая директория
WORKDIR /workspace

# 4) Копируем скрипт
COPY scripts/convert.py .

# 5) Точка входа
ENTRYPOINT ["python", "convert.py"]
