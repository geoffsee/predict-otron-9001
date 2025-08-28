# Predict-Otron-9000 Architecture Documentation

This document provides comprehensive architectural diagrams for the Predict-Otron-9000 multi-service AI platform, showing all supported configurations and deployment patterns.

## Table of Contents

- [System Overview](#system-overview)
- [Workspace Structure](#workspace-structure)
- [Deployment Configurations](#deployment-configurations)
  - [Development Mode](#development-mode)
  - [Docker Monolithic](#docker-monolithic)
  - [Kubernetes Microservices](#kubernetes-microservices)
- [Service Interactions](#service-interactions)
- [Platform-Specific Configurations](#platform-specific-configurations)
- [Data Flow Patterns](#data-flow-patterns)

## System Overview

The Predict-Otron-9000 is a comprehensive multi-service AI platform built around local LLM inference, embeddings, and web interfaces. The system supports flexible deployment patterns from monolithic to microservices architectures.

```mermaid
graph TB
    subgraph "Core Components"
        A[Main Server<br/>predict-otron-9000]
        B[Inference Engine<br/>Gemma via Candle]
        C[Embeddings Engine<br/>FastEmbed]
        D[Web Frontend<br/>Leptos WASM]
    end
    
    subgraph "Client Interfaces"
        E[TypeScript CLI<br/>Bun/Node.js]
        F[Web Browser<br/>HTTP/WebSocket]
        G[HTTP API Clients<br/>OpenAI Compatible]
    end
    
    subgraph "Platform Support"
        H[CPU Fallback<br/>All Platforms]
        I[CUDA Support<br/>Linux GPU]
        J[Metal Support<br/>macOS GPU]
    end
    
    A --- B
    A --- C
    A --- D
    E -.-> A
    F -.-> A
    G -.-> A
    B --- H
    B --- I
    B --- J
```

## Workspace Structure

The project uses a 4-crate Rust workspace with TypeScript tooling, designed for maximum flexibility in deployment configurations.

```mermaid
graph TD
    subgraph "Rust Workspace"
        subgraph "Main Orchestrator"
            A[predict-otron-9000<br/>Edition: 2024<br/>Port: 8080]
        end
        
        subgraph "AI Services"
            B[inference-engine<br/>Edition: 2021<br/>Port: 8080<br/>Candle ML]
            C[embeddings-engine<br/>Edition: 2024<br/>Port: 8080<br/>FastEmbed]
        end
        
        subgraph "Frontend"
            D[leptos-app<br/>Edition: 2021<br/>Port: 3000/8788<br/>WASM/SSR]
        end
    end
    
    subgraph "External Tooling"
        E[cli.ts<br/>TypeScript/Bun<br/>OpenAI SDK]
    end
    
    subgraph "Dependencies"
        A --> B
        A --> C
        A --> D
        B -.-> F[Candle 0.9.1]
        C -.-> G[FastEmbed 4.x]
        D -.-> H[Leptos 0.8.0]
        E -.-> I[OpenAI SDK 5.16+]
    end
    
    style A fill:#e1f5fe
    style B fill:#f3e5f5
    style C fill:#e8f5e8
    style D fill:#fff3e0
    style E fill:#fce4ec
```

## Deployment Configurations

### Development Mode

Local development runs all services integrated within the main server for simplicity.

```mermaid
graph LR
    subgraph "Development Environment"
        subgraph "Single Process - Port 8080"
            A[predict-otron-9000 Server]
            A --> B[Embedded Inference Engine]
            A --> C[Embedded Embeddings Engine]
            A --> D[SSR Leptos Frontend]
        end
        
        subgraph "Separate Frontend - Port 8788"
            E[Trunk Dev Server<br/>Hot Reload: 3001]
        end
    end
    
    subgraph "External Clients"
        F[CLI Client<br/>cli.ts via Bun]
        G[Web Browser]
        H[HTTP API Clients]
    end
    
    F -.-> A
    G -.-> A
    G -.-> E
    H -.-> A
    
    style A fill:#e3f2fd
    style E fill:#f1f8e9
```

### Docker Monolithic

Docker Compose runs a single containerized service handling all functionality.

```mermaid
graph TB
    subgraph "Docker Environment"
        subgraph "predict-otron-9000 Container"
            A[Main Server :8080]
            A --> B[Inference Engine<br/>Library Mode]
            A --> C[Embeddings Engine<br/>Library Mode]
            A --> D[Leptos Frontend<br/>SSR Mode]
        end
        
        subgraph "Persistent Storage"
            E[HF Cache Volume<br/>/.hf-cache]
            F[FastEmbed Cache Volume<br/>/.fastembed_cache]
        end
        
        subgraph "Network"
            G[predict-otron-network<br/>Bridge Driver]
        end
    end
    
    subgraph "External Access"
        H[Host Port 8080]
        I[External Clients]
    end
    
    A --- E
    A --- F
    A --- G
    H --> A
    I -.-> H
    
    style A fill:#e8f5e8
    style E fill:#fff3e0
    style F fill:#fff3e0
```

### Kubernetes Microservices

Kubernetes deployment separates all services for horizontal scalability and fault isolation.

```mermaid
graph TB
    subgraph "Kubernetes Namespace"
        subgraph "Main Orchestrator"
            A[predict-otron-9000 Pod<br/>:8080<br/>ClusterIP Service]
        end
        
        subgraph "AI Services"
            B[inference-engine Pod<br/>:8080<br/>ClusterIP Service]
            C[embeddings-engine Pod<br/>:8080<br/>ClusterIP Service]
        end
        
        subgraph "Frontend"
            D[leptos-app Pod<br/>:8788<br/>ClusterIP Service]
        end
        
        subgraph "Ingress"
            E[Ingress Controller<br/>predict-otron-9000.local]
        end
    end
    
    subgraph "External"
        F[External Clients]
        G[Container Registry<br/>ghcr.io/geoffsee/*]
    end
    
    A <--> B
    A <--> C
    E --> A
    E --> D
    F -.-> E
    
    G -.-> A
    G -.-> B
    G -.-> C
    G -.-> D
    
    style A fill:#e3f2fd
    style B fill:#f3e5f5
    style C fill:#e8f5e8
    style D fill:#fff3e0
    style E fill:#fce4ec
```

## Service Interactions

### API Flow and Communication Patterns

```mermaid
sequenceDiagram
    participant Client as External Client
    participant Main as Main Server<br/>(Port 8080)
    participant Inf as Inference Engine
    participant Emb as Embeddings Engine
    participant Web as Web Frontend
    
    Note over Client, Web: Development/Monolithic Mode
    Client->>Main: POST /v1/chat/completions
    Main->>Inf: Internal call (library)
    Inf-->>Main: Generated response
    Main-->>Client: Streaming/Non-streaming response
    
    Client->>Main: POST /v1/embeddings
    Main->>Emb: Internal call (library)
    Emb-->>Main: Vector embeddings
    Main-->>Client: Embeddings response
    
    Note over Client, Web: Kubernetes Microservices Mode
    Client->>Main: POST /v1/chat/completions
    Main->>Inf: HTTP POST :8080/v1/chat/completions
    Inf-->>Main: HTTP Response (streaming)
    Main-->>Client: Proxied response
    
    Client->>Main: POST /v1/embeddings
    Main->>Emb: HTTP POST :8080/v1/embeddings
    Emb-->>Main: HTTP Response
    Main-->>Client: Proxied response
    
    Note over Client, Web: Web Interface Flow
    Web->>Main: WebSocket connection
    Web->>Main: Chat message
    Main->>Inf: Process inference
    Inf-->>Main: Streaming tokens
    Main-->>Web: WebSocket stream
```

### Port Configuration Matrix

```mermaid
graph TB
    subgraph "Port Allocation by Mode"
        subgraph "Development"
            A[Main Server: 8080<br/>All services embedded]
            B[Leptos Dev: 8788<br/>Hot reload: 3001]
        end
        
        subgraph "Docker Monolithic"
            C[Main Server: 8080<br/>All services embedded<br/>Host mapped]
        end
        
        subgraph "Kubernetes Microservices"
            D[Main Server: 8080]
            E[Inference Engine: 8080]
            F[Embeddings Engine: 8080]
            G[Leptos Frontend: 8788]
        end
    end
    
    style A fill:#e3f2fd
    style C fill:#e8f5e8
    style D fill:#f3e5f5
    style E fill:#f3e5f5
    style F fill:#e8f5e8
    style G fill:#fff3e0
```

## Platform-Specific Configurations

### Hardware Acceleration Support

```mermaid
graph TB
    subgraph "Platform Detection"
        A[Build System]
    end
    
    subgraph "macOS"
        A --> B[Metal Features Available]
        B --> C[CPU Fallback<br/>Stability Priority]
        C --> D[F32 Precision<br/>Gemma Compatibility]
    end
    
    subgraph "Linux"
        A --> E[CUDA Features Available]
        E --> F[GPU Acceleration<br/>Performance Priority]
        F --> G[BF16 Precision<br/>GPU Optimized]
        E --> H[CPU Fallback<br/>F32 Precision]
    end
    
    subgraph "Other Platforms"
        A --> I[CPU Only<br/>Universal Compatibility]
        I --> J[F32 Precision<br/>Standard Support]
    end
    
    style B fill:#e8f5e8
    style E fill:#e3f2fd
    style I fill:#fff3e0
```

### Model Loading and Caching

```mermaid
graph LR
    subgraph "Model Access Flow"
        A[Application Start] --> B{Model Cache Exists?}
        B -->|Yes| C[Load from Cache]
        B -->|No| D[HuggingFace Authentication]
        D --> E{HF Token Valid?}
        E -->|Yes| F[Download Model]
        E -->|No| G[Authentication Error]
        F --> H[Save to Cache]
        H --> C
        C --> I[Initialize Inference]
    end
    
    subgraph "Cache Locations"
        J[HF_HOME Cache<br/>.hf-cache]
        K[FastEmbed Cache<br/>.fastembed_cache]
    end
    
    F -.-> J
    F -.-> K
    
    style D fill:#fce4ec
    style G fill:#ffebee
```

## Data Flow Patterns

### Request Processing Pipeline

```mermaid
flowchart TD
    A[Client Request] --> B{Request Type}
    
    B -->|Chat Completion| C[Parse Messages]
    B -->|Model List| D[Return Available Models]
    B -->|Embeddings| E[Process Text Input]
    
    C --> F[Apply Prompt Template]
    F --> G{Streaming?}
    
    G -->|Yes| H[Initialize Stream]
    G -->|No| I[Generate Complete Response]
    
    H --> J[Token Generation Loop]
    J --> K[Send Chunk]
    K --> L{More Tokens?}
    L -->|Yes| J
    L -->|No| M[End Stream]
    
    I --> N[Return Complete Response]
    
    E --> O[Generate Embeddings]
    O --> P[Return Vectors]
    
    D --> Q[Return Model Metadata]
    
    style A fill:#e3f2fd
    style H fill:#e8f5e8
    style I fill:#f3e5f5
    style O fill:#fff3e0
```

### Authentication and Security Flow

```mermaid
sequenceDiagram
    participant User as User/Client
    participant App as Application
    participant HF as HuggingFace Hub
    participant Model as Model Cache
    
    Note over User, Model: First-time Setup
    User->>App: Start application
    App->>HF: Check model access (gated)
    HF-->>App: 401 Unauthorized
    App-->>User: Requires HF authentication
    
    User->>User: huggingface-cli login
    User->>App: Retry start
    App->>HF: Check model access (with token)
    HF-->>App: 200 OK + model metadata
    App->>HF: Download model files
    HF-->>App: Model data stream
    App->>Model: Cache model locally
    
    Note over User, Model: Subsequent Runs
    User->>App: Start application  
    App->>Model: Load cached model
    Model-->>App: Ready for inference
```

---

## Summary

The Predict-Otron-9000 architecture provides maximum flexibility through:

- **Monolithic Mode**: Single server embedding all services for development and simple deployments
- **Microservices Mode**: Separate services for production scalability and fault isolation
- **Hybrid Capabilities**: Each service can operate as both library and standalone service
- **Platform Optimization**: Conditional compilation for optimal performance across CPU/GPU configurations
- **OpenAI Compatibility**: Standard API interfaces for seamless integration with existing tools

This flexible architecture allows teams to start with simple monolithic deployments and scale to distributed microservices as needs grow, all while maintaining API compatibility and leveraging platform-specific optimizations.