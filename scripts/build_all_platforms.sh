#!/bin/bash

# Cross-platform build script for predict-otron-9000
# Builds all workspace crates for common platforms

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Supported platforms
PLATFORMS=(
    "x86_64-unknown-linux-gnu"
    "x86_64-pc-windows-msvc"
    "x86_64-apple-darwin"
    "aarch64-apple-darwin"
    "aarch64-unknown-linux-gnu"
)

# Main binaries to build
MAIN_BINARIES=(
    "predict-otron-9000"
    "embeddings-engine"
)

# Inference engine binaries (with bin feature)
INFERENCE_BINARIES=(
    "gemma_inference"
    "llama_inference"
)

# Other workspace binaries
OTHER_BINARIES=(
    "helm-chart-tool"
)

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================${NC}"
}

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_dependencies() {
    print_header "Checking Dependencies"
    
    # Check rust
    if ! command -v cargo >/dev/null 2>&1; then
        print_error "Rust/Cargo is not installed"
        exit 1
    fi
    
    # Check cargo-leptos for WASM frontend
    if ! command -v cargo-leptos >/dev/null 2>&1; then
        print_warn "cargo-leptos not found. Installing..."
        cargo install cargo-leptos
    fi
    
    print_info "All dependencies available"
}

install_targets() {
    print_header "Installing Rust Targets"
    
    for platform in "${PLATFORMS[@]}"; do
        print_info "Installing target: $platform"
        rustup target add "$platform" || {
            print_warn "Failed to install target $platform (may not be available on this host)"
        }
    done
    
    # Add WASM target for leptos
    print_info "Installing wasm32-unknown-unknown target for Leptos"
    rustup target add wasm32-unknown-unknown
}

create_build_dirs() {
    print_header "Setting up Build Directory"
    
    rm -rf "$BUILD_DIR"
    mkdir -p "$BUILD_DIR"
    
    for platform in "${PLATFORMS[@]}"; do
        mkdir -p "$BUILD_DIR/$platform"
    done
    
    mkdir -p "$BUILD_DIR/web"
    print_info "Build directories created"
}

build_leptos_app() {
    print_header "Building Leptos Web Frontend"
    
    cd "$PROJECT_ROOT/crates/leptos-app"
    
    # Build the WASM frontend
    print_info "Building WASM frontend with cargo-leptos..."
    cargo leptos build --release || {
        print_error "Failed to build Leptos WASM frontend"
        return 1
    }
    
    # Copy built assets to build directory
    if [ -d "target/site" ]; then
        cp -r target/site/* "$BUILD_DIR/web/"
        print_info "Leptos frontend built and copied to $BUILD_DIR/web/"
    else
        print_error "Leptos build output not found at target/site"
        return 1
    fi
    
    cd "$PROJECT_ROOT"
}

get_platform_features() {
    local platform="$1"
    local features=""
    
    case "$platform" in
        *-apple-darwin)
            # macOS uses Metal but routes to CPU for Gemma stability
            features=""
            ;;
        *-unknown-linux-gnu|*-pc-windows-msvc)
            # Linux and Windows can use CUDA if available
            features=""
            ;;
        *)
            features=""
            ;;
    esac
    
    echo "$features"
}

build_binary_for_platform() {
    local binary_name="$1"
    local platform="$2"
    local package_name="$3"
    local additional_args="$4"
    
    print_info "Building $binary_name for $platform"
    
    local features=$(get_platform_features "$platform")
    local feature_flag=""
    if [ -n "$features" ]; then
        feature_flag="--features $features"
    fi
    
    # Build command
    local build_cmd="cargo build --release --target $platform --bin $binary_name"
    
    if [ -n "$package_name" ]; then
        build_cmd="$build_cmd --package $package_name"
    fi
    
    if [ -n "$additional_args" ]; then
        build_cmd="$build_cmd $additional_args"
    fi
    
    if [ -n "$feature_flag" ]; then
        build_cmd="$build_cmd $feature_flag"
    fi
    
    print_info "Running: $build_cmd"
    
    if eval "$build_cmd"; then
        # Copy binary to build directory
        local target_dir="target/$platform/release"
        local binary_file="$binary_name"
        
        # Add .exe extension for Windows
        if [[ "$platform" == *-pc-windows-msvc ]]; then
            binary_file="$binary_name.exe"
        fi
        
        if [ -f "$target_dir/$binary_file" ]; then
            cp "$target_dir/$binary_file" "$BUILD_DIR/$platform/"
            print_info "✓ $binary_name built and copied for $platform"
        else
            print_error "Binary not found: $target_dir/$binary_file"
            return 1
        fi
    else
        print_error "Failed to build $binary_name for $platform"
        return 1
    fi
}

build_for_platform() {
    local platform="$1"
    print_header "Building for $platform"
    
    local failed_builds=()
    
    # Build main binaries
    for binary in "${MAIN_BINARIES[@]}"; do
        if ! build_binary_for_platform "$binary" "$platform" "$binary" ""; then
            failed_builds+=("$binary")
        fi
    done
    
    # Build inference engine binaries with bin feature
    for binary in "${INFERENCE_BINARIES[@]}"; do
        if ! build_binary_for_platform "$binary" "$platform" "inference-engine" "--features bin"; then
            failed_builds+=("$binary")
        fi
    done
    
    # Build other workspace binaries
    for binary in "${OTHER_BINARIES[@]}"; do
        if ! build_binary_for_platform "$binary" "$platform" "$binary" ""; then
            failed_builds+=("$binary")
        fi
    done
    
    if [ ${#failed_builds[@]} -eq 0 ]; then
        print_info "✓ All binaries built successfully for $platform"
    else
        print_warn "Some builds failed for $platform: ${failed_builds[*]}"
    fi
}

create_archives() {
    print_header "Creating Release Archives"
    
    cd "$BUILD_DIR"
    
    for platform in "${PLATFORMS[@]}"; do
        if [ -d "$platform" ] && [ -n "$(ls -A "$platform" 2>/dev/null)" ]; then
            local archive_name="predict-otron-9000-${platform}-${TIMESTAMP}"
            
            print_info "Creating archive for $platform"
            
            # Create platform-specific directory with all files
            mkdir -p "$archive_name"
            cp -r "$platform"/* "$archive_name/"
            
            # Add web assets to each platform archive
            if [ -d "web" ]; then
                mkdir -p "$archive_name/web"
                cp -r web/* "$archive_name/web/"
            fi
            
            # Create README for the platform
            cat > "$archive_name/README.txt" << EOF
Predict-Otron-9000 - Platform: $platform
Build Date: $(date)
========================================

Binaries included:
$(ls -1 "$platform")

Web Frontend:
- Located in the 'web' directory
- Serve with any static file server on port 8788 or configure your server

Usage:
1. Start the main server: ./predict-otron-9000
2. Start embeddings service: ./embeddings-engine  
3. Access web interface at http://localhost:8080 (served by main server)

For more information, visit: https://github.com/geoffsee/predict-otron-9000
EOF
            
            # Create tar.gz archive
            tar -czf "${archive_name}.tar.gz" "$archive_name"
            rm -rf "$archive_name"
            
            print_info "✓ Created ${archive_name}.tar.gz"
        else
            print_warn "No binaries found for $platform, skipping archive"
        fi
    done
    
    cd "$PROJECT_ROOT"
}

generate_build_report() {
    print_header "Build Report"
    
    echo "Build completed at: $(date)"
    echo "Build directory: $BUILD_DIR"
    echo ""
    echo "Archives created:"
    ls -la "$BUILD_DIR"/*.tar.gz 2>/dev/null || echo "No archives created"
    echo ""
    echo "Platform directories:"
    for platform in "${PLATFORMS[@]}"; do
        if [ -d "$BUILD_DIR/$platform" ]; then
            echo "  $platform:"
            ls -la "$BUILD_DIR/$platform" | sed 's/^/    /'
        fi
    done
    
    if [ -d "$BUILD_DIR/web" ]; then
        echo ""
        echo "Web frontend assets:"
        ls -la "$BUILD_DIR/web" | head -10 | sed 's/^/    /'
        if [ $(ls -1 "$BUILD_DIR/web" | wc -l) -gt 10 ]; then
            echo "    ... and $(( $(ls -1 "$BUILD_DIR/web" | wc -l) - 10 )) more files"
        fi
    fi
}

main() {
    print_header "Predict-Otron-9000 Cross-Platform Build Script"
    
    cd "$PROJECT_ROOT"
    
    check_dependencies
    install_targets
    create_build_dirs
    
    # Build Leptos web frontend first
    build_leptos_app
    
    # Build for each platform
    for platform in "${PLATFORMS[@]}"; do
        build_for_platform "$platform"
    done
    
    create_archives
    generate_build_report
    
    print_header "Build Complete!"
    print_info "All artifacts are available in: $BUILD_DIR"
}

# Handle command line arguments
case "${1:-}" in
    --help|-h)
        echo "Usage: $0 [options]"
        echo ""
        echo "Cross-platform build script for predict-otron-9000"
        echo ""
        echo "Options:"
        echo "  --help, -h          Show this help message"
        echo "  --platforms         Show supported platforms"
        echo "  --clean             Clean build directory before building"
        echo ""
        echo "Supported platforms:"
        for platform in "${PLATFORMS[@]}"; do
            echo "  - $platform"
        done
        echo ""
        echo "Prerequisites:"
        echo "  - Rust toolchain with rustup"
        echo "  - cargo-leptos (will be installed if missing)"
        echo "  - Platform-specific toolchains for cross-compilation"
        echo ""
        exit 0
        ;;
    --platforms)
        echo "Supported platforms:"
        for platform in "${PLATFORMS[@]}"; do
            echo "  - $platform"
        done
        exit 0
        ;;
    --clean)
        print_info "Cleaning build directory..."
        rm -rf "$BUILD_DIR"
        print_info "Build directory cleaned"
        ;;
esac

main "$@"