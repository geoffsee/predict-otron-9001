# Helm Chart Tool

A Rust-based tool that automatically generates Helm charts from Cargo.toml metadata in Rust workspace projects.

## Overview

This tool scans a Rust workspace for crates containing Docker/Kubernetes metadata in their `Cargo.toml` files and generates a complete, production-ready Helm chart with deployments, services, ingress, and configuration templates.

## Features

- **Automatic Service Discovery**: Scans all `Cargo.toml` files in a workspace to find services with Kubernetes metadata
- **Complete Helm Chart Generation**: Creates Chart.yaml, values.yaml, deployment templates, service templates, ingress template, and helper templates
- **Metadata Extraction**: Uses `[package.metadata.kube]` sections from Cargo.toml files to extract:
  - Docker image names
  - Service ports
  - Replica counts
  - Service names
- **Production Ready**: Generated charts include health checks, resource limits, node selectors, affinity rules, and tolerations
- **Helm Best Practices**: Follows Helm chart conventions and passes `helm lint` validation

## Installation

Build the tool from source:

```bash
cd helm-chart-tool
cargo build --release
```

The binary will be available at `target/release/helm-chart-tool`.

## Usage

### Basic Usage

```bash
./target/release/helm-chart-tool --workspace /path/to/rust/workspace --output ./my-helm-chart
```

### Command Line Options

- `--workspace, -w PATH`: Path to the workspace root (default: `.`)
- `--output, -o PATH`: Output directory for the Helm chart (default: `./helm-chart`)
- `--name, -n NAME`: Name of the Helm chart (default: `predict-otron-9000`)

### Example

```bash
# Generate chart from current workspace
./target/release/helm-chart-tool

# Generate chart from specific workspace with custom output
./target/release/helm-chart-tool -w /path/to/my/workspace -o ./charts/my-app -n my-application
```

## Cargo.toml Metadata Format

The tool expects crates to have Kubernetes metadata in their `Cargo.toml` files:

```toml
[package]
name = "my-service"
version = "0.1.0"

# Required: Kubernetes metadata
[package.metadata.kube]
image = "ghcr.io/myorg/my-service:latest"
replicas = 1
port = 8080

# Optional: Docker Compose metadata (currently not used but parsed)
[package.metadata.compose]
image = "ghcr.io/myorg/my-service:latest"
port = 8080
```

### Required Fields

- `image`: Full Docker image name including registry and tag
- `port`: Port number the service listens on
- `replicas`: Number of replicas to deploy (optional, defaults to 1)

## Generated Chart Structure

The tool generates a complete Helm chart with the following structure:

```
helm-chart/
├── Chart.yaml              # Chart metadata
├── values.yaml             # Default configuration values
├── .helmignore            # Files to ignore when packaging
└── templates/
    ├── _helpers.tpl        # Template helper functions
    ├── ingress.yaml        # Ingress configuration (optional)
    ├── {service}-deployment.yaml    # Deployment for each service
    └── {service}-service.yaml       # Service for each service
```

### Generated Files

#### Chart.yaml
- Standard Helm v2 chart metadata
- Includes keywords for AI/ML applications
- Maintainer information

#### values.yaml
- Individual service configurations
- Resource limits and requests
- Service types and ports
- Node selectors, affinity, and tolerations
- Global settings and ingress configuration

#### Deployment Templates
- Kubernetes Deployment manifests
- Health checks (liveness and readiness probes)
- Resource management
- Container port configuration from metadata
- Support for node selectors, affinity, and tolerations

#### Service Templates
- Kubernetes Service manifests
- ClusterIP services by default
- Port mapping from metadata

#### Ingress Template
- Optional ingress configuration
- Disabled by default
- Configurable through values.yaml

## Example Output

When run against the predict-otron-9000 workspace, the tool generates:

```bash
$ ./target/release/helm-chart-tool --workspace .. --output ../generated-helm-chart
Parsing workspace at: ..
Output directory: ../generated-helm-chart
Chart name: predict-otron-9000
Found 4 services:
  - leptos-app: ghcr.io/geoffsee/leptos-app:latest (port 8788)
  - inference-engine: ghcr.io/geoffsee/inference-service:latest (port 8080)
  - embeddings-engine: ghcr.io/geoffsee/embeddings-service:latest (port 8080)
  - predict-otron-9000: ghcr.io/geoffsee/predict-otron-9000:latest (port 8080)
Helm chart generated successfully!
```

## Validation

The generated charts pass Helm validation:

```bash
$ helm lint generated-helm-chart
==> Linting generated-helm-chart
[INFO] Chart.yaml: icon is recommended
1 chart(s) linted, 0 chart(s) failed
```

## Deployment

Deploy the generated chart:

```bash
# Install the chart
helm install my-release ./generated-helm-chart

# Upgrade the chart
helm upgrade my-release ./generated-helm-chart

# Uninstall the chart
helm uninstall my-release
```

### Customization

Customize the deployment by modifying `values.yaml`:

```yaml
# Enable ingress
ingress:
  enabled: true
  className: "nginx"
  hosts:
    - host: my-app.example.com

# Adjust resources for a specific service
predict_otron_9000:
  replicas: 3
  resources:
    limits:
      memory: "4Gi"
      cpu: "2000m"
    requests:
      memory: "2Gi"
      cpu: "1000m"
```

## Requirements

- Rust 2021+ (for building the tool)
- Helm 3.x (for deploying the generated charts)
- Kubernetes cluster (for deployment)

## Limitations

- Currently assumes all services need health checks on `/health` endpoint
- Resource limits are hardcoded defaults (can be overridden in values.yaml)
- Ingress configuration is basic (can be customized through values.yaml)

## Contributing

1. Add new features to the tool
2. Test with various Cargo.toml metadata configurations
3. Validate generated charts with `helm lint`
4. Ensure charts deploy successfully to test clusters

## License

This tool is part of the predict-otron-9000 project and follows the same license terms.