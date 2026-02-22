FROM python:3.11.11-slim AS builder

WORKDIR /build

COPY pyproject.toml README.md ./
COPY src/ src/

RUN pip install --no-cache-dir build && \
    python -m build --wheel --outdir /build/wheels

FROM python:3.11.11-slim

LABEL maintainer="WorldFlux Contributors"
LABEL org.opencontainers.image.source="https://github.com/worldflux/WorldFlux"
LABEL org.opencontainers.image.version="0.1.0"
LABEL org.opencontainers.image.description="Unified Python interface for world models in reinforcement learning"
LABEL org.opencontainers.image.licenses="Apache-2.0"

RUN groupadd -r worldflux && useradd -r -g worldflux worldflux

WORKDIR /app

COPY --from=builder /build/wheels /wheels

RUN pip install --no-cache-dir --no-index --find-links /wheels worldflux && \
    rm -rf /wheels

USER worldflux

ENTRYPOINT ["worldflux"]
