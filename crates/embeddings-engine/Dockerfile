# ---- Build stage ----
FROM rust:1-slim-bullseye AS builder

WORKDIR /usr/src/app

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        pkg-config \
        libssl-dev \
        build-essential \
        && rm -rf /var/lib/apt/lists/*

# Cache deps first
COPY . ./
RUN rm -rf src
RUN mkdir src && echo "fn main() {}" > src/main.rs && echo "// lib" > src/lib.rs && cargo build --release
RUN rm -rf src

# Copy real sources and build
COPY . .
RUN cargo build --release

# ---- Runtime stage ----
FROM debian:bullseye-slim

# Install only what the compiled binary needs
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libssl1.1 \
        ca-certificates \
        && rm -rf /var/lib/apt/lists/*

# Copy binary from builder
COPY --from=builder /usr/src/app/target/release/embeddings-engine /usr/local/bin/

# Run as non-root user for safety
RUN useradd -m appuser
USER appuser

EXPOSE 8080
CMD ["embeddings-engine"]