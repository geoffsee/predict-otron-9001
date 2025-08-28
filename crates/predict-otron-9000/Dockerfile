# ---- Build stage ----
FROM rust:1-slim-bullseye AS builder

WORKDIR /usr/src/app

# Install build dependencies including CUDA toolkit for GPU support (needed for inference-engine dependency)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        pkg-config \
        libssl-dev \
        build-essential \
        wget \
        gnupg2 \
        curl \
        && rm -rf /var/lib/apt/lists/*

# Install CUDA toolkit (required for inference-engine dependency)
# This is a minimal CUDA installation for building
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        cuda-minimal-build-11-8 \
        libcublas-dev-11-8 \
        libcurand-dev-11-8 \
        && rm -rf /var/lib/apt/lists/* \
        && rm cuda-keyring_1.0-1_all.deb

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Copy the entire workspace to get access to all crates (needed for local dependencies)
COPY . ./

# Cache dependencies first - create dummy source files for all crates
RUN rm -rf crates/predict-otron-9000/src crates/inference-engine/src crates/embeddings-engine/src
RUN mkdir -p crates/predict-otron-9000/src crates/inference-engine/src crates/embeddings-engine/src && \
    echo "fn main() {}" > crates/predict-otron-9000/src/main.rs && \
    echo "fn main() {}" > crates/inference-engine/src/main.rs && \
    echo "fn main() {}" > crates/inference-engine/src/cli_main.rs && \
    echo "// lib" > crates/inference-engine/src/lib.rs && \
    echo "fn main() {}" > crates/embeddings-engine/src/main.rs && \
    echo "// lib" > crates/embeddings-engine/src/lib.rs && \
    cargo build --release --bin predict-otron-9000 --package predict-otron-9000

# Remove dummy sources and copy real sources
RUN rm -rf crates/predict-otron-9000/src crates/inference-engine/src crates/embeddings-engine/src
COPY . .

# Build the actual binary
RUN cargo build --release --bin predict-otron-9000 --package predict-otron-9000

# ---- Runtime stage ----
FROM debian:bullseye-slim

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libssl1.1 \
        ca-certificates \
        && rm -rf /var/lib/apt/lists/*

# Install CUDA runtime libraries (required for inference-engine dependency)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget \
        gnupg2 \
        && wget https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/cuda-keyring_1.0-1_all.deb \
        && dpkg -i cuda-keyring_1.0-1_all.deb \
        && apt-get update \
        && apt-get install -y --no-install-recommends \
            cuda-cudart-11-8 \
            libcublas11 \
            libcurand10 \
        && rm -rf /var/lib/apt/lists/* \
        && rm cuda-keyring_1.0-1_all.deb \
        && apt-get purge -y wget gnupg2

# Copy binary from builder
COPY --from=builder /usr/src/app/target/release/predict-otron-9000 /usr/local/bin/

# Run as non-root user for safety
RUN useradd -m appuser
USER appuser

EXPOSE 8080
CMD ["predict-otron-9000"]