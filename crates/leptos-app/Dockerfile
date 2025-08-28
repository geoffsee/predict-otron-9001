# Build stage
FROM rust:1-alpine AS builder

# Install build dependencies
RUN apk add --no-cache npm nodejs musl-dev pkgconfig openssl-dev git curl bash

RUN curl -L --proto '=https' --tlsv1.2 -sSf https://raw.githubusercontent.com/cargo-bins/cargo-binstall/main/install-from-binstall-release.sh | bash

WORKDIR /app

# Copy manifest first (cache deps) 
COPY . .

# Install cargo-leptos
RUN cargo binstall cargo-leptos

# Build release artifacts
RUN cargo leptos build --release

EXPOSE 8788
CMD ["cargo", "leptos", "serve", "--release"]