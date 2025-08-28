# ---- Build stage ----
FROM rust:1-slim-bullseye AS builder

WORKDIR /usr/src/app

# Install build dependencies including CUDA toolkit for GPU support
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        pkg-config \
        libssl-dev \
        build-essential \
        wget \
        gnupg2 \
        curl \
        && rm -rf /var/lib/apt/lists/*

# Install CUDA toolkit (optional, for GPU support)
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

# Copy the entire workspace to get access to all crates
COPY . ./

# Cache dependencies first - create dummy source files
RUN rm -rf crates/inference-engine/src
RUN mkdir -p crates/inference-engine/src && \
    echo "fn main() {}" > crates/inference-engine/src/main.rs && \
    echo "fn main() {}" > crates/inference-engine/src/cli_main.rs && \
    echo "// lib" > crates/inference-engine/src/lib.rs && \
    cargo build --release --bin cli --package inference-engine

# Remove dummy source and copy real sources
RUN rm -rf crates/inference-engine/src
COPY . .

# Build the actual CLI binary
RUN cargo build --release --bin cli --package inference-engine

# ---- Runtime stage ----
FROM debian:bullseye-slim

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libssl1.1 \
        ca-certificates \
        && rm -rf /var/lib/apt/lists/*

# Install CUDA runtime libraries (optional, for GPU support at runtime)
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
COPY --from=builder /usr/src/app/target/release/cli /usr/local/bin/inference-cli

# Run as non-root user for safety
RUN useradd -m appuser
USER appuser

EXPOSE 8080
CMD ["inference-cli"]