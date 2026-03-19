# Parity proof execution environment.
#
# Pins JAX + CUDA versions to avoid "Unable to initialize backend 'cuda'"
# failures in CI. See reports/parity/cuda_jax_compatibility.md for the
# full compatibility matrix.
#
# Build:
#   docker build -f docker/parity-environment.Dockerfile -t worldflux-parity .
#
# Run:
#   docker run --gpus all worldflux-parity python scripts/preflight_parity_env.py

FROM nvidia/cuda:12.3.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8

# System packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
        python3.11-dev \
        python3-pip \
        git \
        curl \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Set python3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# JAX with CUDA support - pinned versions
ENV JAX_VERSION=0.4.30
ENV JAXLIB_VERSION=0.4.30

RUN pip install --no-cache-dir \
    "jax==${JAX_VERSION}" \
    "jaxlib==${JAXLIB_VERSION}+cuda12.cudnn89" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# PyTorch with CUDA support
RUN pip install --no-cache-dir \
    torch>=2.0.0,<3.0.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Copy project and install
WORKDIR /workspace
COPY pyproject.toml uv.lock ./
COPY src/ src/
COPY scripts/ scripts/
COPY reports/ reports/

RUN uv sync --extra dev --extra training

# Environment configuration for JAX
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false
ENV JAX_PLATFORMS=cuda,cpu
ENV TF_CPP_MIN_LOG_LEVEL=2

# Healthcheck - verify both JAX and PyTorch can see GPU
HEALTHCHECK --interval=30s --timeout=10s CMD \
    python -c "import jax; assert len(jax.devices('gpu')) > 0" && \
    python -c "import torch; assert torch.cuda.is_available()"

CMD ["python", "scripts/preflight_parity_env.py"]
