FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# --------------------------
# Sistema básico
# --------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    cmake \
    ffmpeg \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# --------------------------
# Diretório de trabalho
# --------------------------
WORKDIR /workspace

# --------------------------
# Copia o repositório
# (IMPORTANTE: rode build na raiz do projeto)
# --------------------------
COPY . /workspace

# --------------------------
# Python deps
# --------------------------
RUN pip install --upgrade pip

RUN pip install \
    matplotlib \
    numpy \
    pandas \
    scikit-image \
    pillow \
    lightning \
    torchvision

# --------------------------
# Instala Minerva (ESSENCIAL)
# --------------------------
RUN pip install -e Minerva-Dev

# --------------------------
# PYTHONPATH (garantia extra)
# --------------------------
ENV PYTHONPATH="/workspace:$PYTHONPATH"