# Server Configuration Guide

The predict-otron-9000 server supports two deployment modes controlled by the `SERVER_CONFIG` environment variable:

1. **Standalone Mode** (default): Runs inference and embeddings services locally within the main server process
2. **HighAvailability Mode**: Proxies requests to external inference and embeddings services

## Configuration Format

The `SERVER_CONFIG` environment variable accepts a JSON configuration with the following structure:

```json
{
  "serverMode": "Standalone",
  "services": {
    "inference_url": "http://inference-service:8080",
    "embeddings_url": "http://embeddings-service:8080"
  }
}
```

or 

```json
{
  "serverMode": "HighAvailability",
  "services": {
    "inference_url": "http://inference-service:8080",
    "embeddings_url": "http://embeddings-service:8080"
  }
}
```

**Fields:**
- `serverMode`: Either `"Local"` or `"HighAvailability"`
- `services`: Optional object containing service URLs (uses defaults if not provided)

## Standalone Mode (Default)

If `SERVER_CONFIG` is not set or contains invalid JSON, the server defaults to Local mode.

### Example: Explicit Local Mode
```bash
export SERVER_CONFIG='{"serverMode": "Standalone"}'
./run_server.sh
```

In Standalone mode:
- Inference requests are handled by the embedded inference engine
- Embeddings requests are handled by the embedded embeddings engine
- No external services are required
- Supports all existing functionality without changes

## HighAvailability Mode

In HighAvailability mode, the server acts as a proxy, forwarding requests to external services.

### Example: Basic HighAvailability Mode
```bash
export SERVER_CONFIG='{"serverMode": "HighAvailability"}'
./run_server.sh
```

This uses the default service URLs:
- Inference service: `http://inference-service:8080`
- Embeddings service: `http://embeddings-service:8080`

### Example: Custom Service URLs
```bash
export SERVER_CONFIG='{
  "serverMode": "HighAvailability",
  "services": {
    "inference_url": "http://custom-inference:9000",
    "embeddings_url": "http://custom-embeddings:9001"
  }
}'
./run_server.sh
```

## Docker Compose Example

```yaml
version: '3.8'
services:
  # Inference service
  inference-service:
    image: ghcr.io/geoffsee/inference-service:latest
    ports:
      - "8081:8080"
    environment:
      - RUST_LOG=info

  # Embeddings service  
  embeddings-service:
    image: ghcr.io/geoffsee/embeddings-service:latest
    ports:
      - "8082:8080"
    environment:
      - RUST_LOG=info

  # Main proxy server
  predict-otron-9000:
    image: ghcr.io/geoffsee/predict-otron-9000:latest
    ports:
      - "8080:8080"
    environment:
      - RUST_LOG=info
      - SERVER_CONFIG={"serverMode":"HighAvailability","services":{"inference_url":"http://inference-service:8080","embeddings_url":"http://embeddings-service:8080"}}
    depends_on:
      - inference-service
      - embeddings-service
```

## Kubernetes Example

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: server-config
data:
  SERVER_CONFIG: |
    {
      "serverMode": "HighAvailability",
      "services": {
        "inference_url": "http://inference-service:8080",
        "embeddings_url": "http://embeddings-service:8080"
      }
    }
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: predict-otron-9000
spec:
  replicas: 3
  selector:
    matchLabels:
      app: predict-otron-9000
  template:
    metadata:
      labels:
        app: predict-otron-9000
    spec:
      containers:
      - name: predict-otron-9000
        image: ghcr.io/geoffsee/predict-otron-9000:latest
        ports:
        - containerPort: 8080
        env:
        - name: RUST_LOG
          value: "info"
        - name: SERVER_CONFIG
          valueFrom:
            configMapKeyRef:
              name: server-config
              key: SERVER_CONFIG
```

## API Compatibility

Both modes expose the same OpenAI-compatible API endpoints:

- `POST /v1/chat/completions` - Chat completions (streaming and non-streaming)
- `GET /v1/models` - List available models
- `POST /v1/embeddings` - Generate text embeddings
- `GET /health` - Health check
- `GET /` - Root endpoint

## Logging

The server logs the selected mode on startup:

**Local Mode:**
```
INFO predict_otron_9000: Running in Standalone mode
```

**HighAvailability Mode:**
```
INFO predict_otron_9000: Running in HighAvailability mode - proxying to external services
INFO predict_otron_9000: Inference service URL: http://inference-service:8080
INFO predict_otron_9000: Embeddings service URL: http://embeddings-service:8080
```

## Error Handling

- Invalid JSON in `SERVER_CONFIG` falls back to Local mode with a warning
- Missing `SERVER_CONFIG` defaults to Local mode
- Network errors to external services return HTTP 502 (Bad Gateway)
- Request/response proxying preserves original HTTP status codes and headers

## Performance Considerations

**Local Mode:**
- Lower latency (no network overhead)
- Higher memory usage (models loaded locally)
- Single point of failure

**HighAvailability Mode:**
- Higher latency (network requests)
- Lower memory usage (no local models)
- Horizontal scaling possible
- Network reliability dependent
- 5-minute timeout for long-running inference requests

## Troubleshooting

1. **Configuration not applied**: Check JSON syntax and restart the server
2. **External services unreachable**: Verify service URLs and network connectivity
3. **Timeouts**: Check if inference requests exceed the 5-minute timeout limit
4. **502 errors**: External services may be down or misconfigured

## Migration

To migrate from Local to HighAvailability mode:

1. Deploy separate inference and embeddings services
2. Update `SERVER_CONFIG` to point to the new services
3. Restart the predict-otron-9000 server
4. Verify endpoints are working with test requests

The API contract remains identical, ensuring zero-downtime migration possibilities.
