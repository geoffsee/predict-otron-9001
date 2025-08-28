# Build stage
FROM rust:1-alpine AS builder

# Install build dependencies
RUN apk add --no-cache npm nodejs musl-dev pkgconfig openssl-dev git curl bash

RUN curl -L --proto '=https' --tlsv1.2 -sSf https://raw.githubusercontent.com/cargo-bins/cargo-binstall/main/install-from-binstall-release.sh | bash

WORKDIR /app

# Copy manifest first (cache deps)
COPY . .
# Install trunk
RUN cargo install wasm-bindgen-cli

RUN cargo binstall trunk

# Build release artifacts
RUN rustup target add wasm32-unknown-unknown && export RUSTFLAGS='--cfg getrandom_backend="wasm_js"' && trunk build --release

# Final stage: static web server
FROM caddy:2-alpine

# Copy built assets into Caddy's web root
COPY --from=builder /app/dist /usr/share/caddy

EXPOSE 8788
CMD ["caddy", "file-server", "--root", "/usr/share/caddy", "--listen", ":8788"]