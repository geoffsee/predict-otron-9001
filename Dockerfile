# Multi-stage build for predict-otron-9000 workspace
FROM rust:1 AS builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy workspace files
COPY Cargo.toml Cargo.lock ./
COPY crates/ ./crates/
COPY integration/ ./integration/

# Build all 3 main server binaries in release mode
RUN cargo build --release -p predict-otron-9000 --bin predict-otron-9000 --no-default-features -p embeddings-engine --bin embeddings-engine -p inference-engine --bin inference-engine

# Runtime stage
FROM debian:bookworm-slim AS runtime

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd -r -s /bin/false -m -d /app appuser

# Set working directory
WORKDIR /app

# Copy binaries from builder stage
COPY --from=builder /app/target/release/predict-otron-9000 ./bin/
COPY --from=builder /app/target/release/embeddings-engine ./bin/
COPY --from=builder /app/target/release/inference-engine ./bin/
# Make binaries executable and change ownership
RUN chmod +x ./bin/* && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose ports (adjust as needed based on your services)
EXPOSE 8080 8081 8082

# Default command (can be overridden)
CMD ["./bin/predict-otron-9000"]