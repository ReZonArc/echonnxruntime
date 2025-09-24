# ONNX Runtime Technical Architecture

This document provides a comprehensive overview of the ONNX Runtime technical architecture, including system design, component relationships, data flow, and deployment patterns.

## Table of Contents

1. [High-Level Architecture](#high-level-architecture)
2. [Core Components](#core-components)
3. [Execution Provider Ecosystem](#execution-provider-ecosystem)
4. [Data Flow and Processing Pipeline](#data-flow-and-processing-pipeline)
5. [Language Bindings and API Layers](#language-bindings-and-api-layers)
6. [Build System and Dependencies](#build-system-and-dependencies)
7. [Deployment Architectures](#deployment-architectures)
8. [Memory Management](#memory-management)
9. [Training vs Inference](#training-vs-inference)

## High-Level Architecture

ONNX Runtime is designed as a cross-platform, high-performance inference and training engine with a modular architecture that supports multiple hardware accelerators and programming languages.

```mermaid
graph TB
    subgraph "Application Layer"
        APP1[Python Applications]
        APP2[C/C++ Applications]
        APP3[JavaScript/Node.js]
        APP4[C# .NET Applications]
        APP5[Java Applications]
        APP6[Objective-C/Swift Apps]
        APP7[Rust Applications]
    end
    
    subgraph "Language Bindings"
        PY[Python API]
        CPP[C/C++ API]
        JS[JavaScript API]
        CS[C# API]
        JAVA[Java API]
        OBJC[Objective-C API]
        RUST[Rust API]
    end
    
    subgraph "ONNX Runtime Core"
        CORE[Core Runtime Engine]
        SESSION[Session Management]
        GRAPH[Graph Processing]
        OPT[Graph Optimization]
        SCHED[Execution Scheduling]
    end
    
    subgraph "Execution Providers"
        CPU[CPU Provider]
        CUDA[CUDA Provider]
        DNNL[DNNL Provider]
        TENSORRT[TensorRT Provider]
        COREML[CoreML Provider]
        DML[DirectML Provider]
        WEBGL[WebGL Provider]
        WEBGPU[WebGPU Provider]
        OTHERS[Other Providers...]
    end
    
    subgraph "Hardware/Platforms"
        X86[x86/x64 CPUs]
        ARM[ARM CPUs]
        NVIDIA[NVIDIA GPUs]
        AMD[AMD GPUs]
        INTEL[Intel GPUs]
        MOBILE[Mobile Devices]
        WEB[Web Browsers]
        CLOUD[Cloud Platforms]
    end
    
    APP1 --> PY
    APP2 --> CPP
    APP3 --> JS
    APP4 --> CS
    APP5 --> JAVA
    APP6 --> OBJC
    APP7 --> RUST
    
    PY --> CORE
    CPP --> CORE
    JS --> CORE
    CS --> CORE
    JAVA --> CORE
    OBJC --> CORE
    RUST --> CORE
    
    CORE --> SESSION
    CORE --> GRAPH
    CORE --> OPT
    CORE --> SCHED
    
    SESSION --> CPU
    SESSION --> CUDA
    SESSION --> DNNL
    SESSION --> TENSORRT
    SESSION --> COREML
    SESSION --> DML
    SESSION --> WEBGL
    SESSION --> WEBGPU
    SESSION --> OTHERS
    
    CPU --> X86
    CPU --> ARM
    CUDA --> NVIDIA
    DNNL --> X86
    TENSORRT --> NVIDIA
    COREML --> MOBILE
    DML --> INTEL
    WEBGL --> WEB
    WEBGPU --> WEB
    OTHERS --> CLOUD
```

## Core Components

The ONNX Runtime core consists of several key components that work together to provide efficient model execution:

```mermaid
graph LR
    subgraph "ONNX Runtime Core Architecture"
        subgraph "Session Layer"
            IS[Inference Session]
            TS[Training Session]
            SM[Session Manager]
        end
        
        subgraph "Graph Layer"
            GP[Graph Parser]
            GV[Graph Viewer]
            GO[Graph Optimizer]
            GT[Graph Transformer]
        end
        
        subgraph "Execution Layer"
            ES[Execution Scheduler]
            EF[Execution Frame]
            KR[Kernel Registry]
            MM[Memory Manager]
        end
        
        subgraph "Provider Layer"
            PA[Provider API]
            PF[Provider Factory]
            PE[Provider Engine]
        end
        
        subgraph "Utilities"
            LOG[Logging]
            PROF[Profiler]
            ALLOC[Allocators]
            THREAD[Threading]
        end
    end
    
    IS --> GP
    TS --> GP
    SM --> IS
    SM --> TS
    
    GP --> GV
    GV --> GO
    GO --> GT
    GT --> ES
    
    ES --> EF
    EF --> KR
    KR --> MM
    MM --> PA
    
    PA --> PF
    PF --> PE
    
    LOG --> ES
    PROF --> EF
    ALLOC --> MM
    THREAD --> ES
```

## Execution Provider Ecosystem

ONNX Runtime's execution provider architecture enables support for diverse hardware accelerators:

```mermaid
graph TB
    subgraph "Execution Provider Architecture"
        subgraph "Core Interface"
            IEP[IExecutionProvider Interface]
            EPF[ExecutionProvider Factory]
            EPR[ExecutionProvider Registry]
        end
        
        subgraph "CPU Providers"
            CPU[CPU Provider]
            DNNL[DNNL Provider]
            XNNPACK[XNNPACK Provider]
            NNAPI[NNAPI Provider]
        end
        
        subgraph "GPU Providers"
            CUDA[CUDA Provider]
            ROCM[ROCm Provider]
            DML[DirectML Provider]
            OPENVINO[OpenVINO Provider]
        end
        
        subgraph "Specialized Providers"
            TENSORRT[TensorRT Provider]
            COREML[CoreML Provider]
            QNN[QNN Provider]
            SNPE[SNPE Provider]
        end
        
        subgraph "Web Providers"
            WEBGL[WebGL Provider]
            WEBGPU[WebGPU Provider]
            WEBNN[WebNN Provider]
            JSEP[JSEP Provider]
        end
        
        subgraph "Cloud/Edge Providers"
            AZURE[Azure Provider]
            ACL[ACL Provider]
            ARMNN[ArmNN Provider]
            VITISAI[Vitis-AI Provider]
        end
    end
    
    IEP --> EPF
    EPF --> EPR
    
    EPR --> CPU
    EPR --> DNNL
    EPR --> XNNPACK
    EPR --> NNAPI
    
    EPR --> CUDA
    EPR --> ROCM
    EPR --> DML
    EPR --> OPENVINO
    
    EPR --> TENSORRT
    EPR --> COREML
    EPR --> QNN
    EPR --> SNPE
    
    EPR --> WEBGL
    EPR --> WEBGPU
    EPR --> WEBNN
    EPR --> JSEP
    
    EPR --> AZURE
    EPR --> ACL
    EPR --> ARMNN
    EPR --> VITISAI
    
    CPU --> |"Fallback"| DNNL
    CUDA --> |"Optimization"| TENSORRT
    WEBGL --> |"Alternative"| WEBGPU
```

## Data Flow and Processing Pipeline

The data processing pipeline shows how models and data flow through the system:

```mermaid
flowchart TD
    subgraph "Model Loading and Preparation"
        MODEL[ONNX Model File]
        PARSER[Model Parser]
        GRAPH[Graph Representation]
        VALIDATE[Model Validation]
    end
    
    subgraph "Graph Optimization"
        OPT1[Constant Folding]
        OPT2[Operator Fusion]
        OPT3[Layout Optimization]
        OPT4[Memory Optimization]
        OPT5[Provider-Specific Opts]
    end
    
    subgraph "Session Creation"
        CONFIG[Session Configuration]
        PROVIDER[Provider Selection]
        ALLOC_SETUP[Allocator Setup]
        KERNEL_REG[Kernel Registration]
    end
    
    subgraph "Execution Phase"
        INPUT[Input Tensors]
        SCHEDULE[Execution Scheduling]
        COMPUTE[Kernel Execution]
        MEMORY[Memory Management]
        OUTPUT[Output Tensors]
    end
    
    subgraph "Post-Processing"
        COPY[Data Copy]
        SYNC[Synchronization]
        CLEANUP[Resource Cleanup]
    end
    
    MODEL --> PARSER
    PARSER --> GRAPH
    GRAPH --> VALIDATE
    VALIDATE --> OPT1
    
    OPT1 --> OPT2
    OPT2 --> OPT3
    OPT3 --> OPT4
    OPT4 --> OPT5
    
    OPT5 --> CONFIG
    CONFIG --> PROVIDER
    PROVIDER --> ALLOC_SETUP
    ALLOC_SETUP --> KERNEL_REG
    
    KERNEL_REG --> INPUT
    INPUT --> SCHEDULE
    SCHEDULE --> COMPUTE
    COMPUTE --> MEMORY
    MEMORY --> OUTPUT
    
    OUTPUT --> COPY
    COPY --> SYNC
    SYNC --> CLEANUP
```

## Language Bindings and API Layers

ONNX Runtime provides APIs for multiple programming languages:

```mermaid
graph TB
    subgraph "Language Bindings Architecture"
        subgraph "Native C++ Core"
            CAPI[C API Layer]
            CORE[C++ Core Engine]
            ABI[ABI Interface]
        end
        
        subgraph "High-Level Languages"
            PYTHON[Python Bindings]
            CSHARP[C# Bindings]
            JAVA[Java Bindings]
            NODE[Node.js Bindings]
        end
        
        subgraph "Systems Languages"
            RUST[Rust Bindings]
            GO[Go Bindings]
            OBJC[Objective-C Bindings]
        end
        
        subgraph "Web Technologies"
            WASM[WebAssembly]
            WEBJS[Web JavaScript]
            REACTNATIVE[React Native]
        end
        
        subgraph "Mobile Platforms"
            IOS[iOS Framework]
            ANDROID[Android AAR]
            XAMARIN[Xamarin Bindings]
        end
    end
    
    CORE --> CAPI
    CAPI --> ABI
    
    ABI --> PYTHON
    ABI --> CSHARP
    ABI --> JAVA
    ABI --> NODE
    
    ABI --> RUST
    ABI --> GO
    ABI --> OBJC
    
    CORE --> WASM
    WASM --> WEBJS
    NODE --> REACTNATIVE
    
    OBJC --> IOS
    JAVA --> ANDROID
    CSHARP --> XAMARIN
    
    PYTHON -.-> |"Extends"| WASM
    CSHARP -.-> |"Shared"| CORE
    JAVA -.-> |"JNI"| ABI
```

## Build System and Dependencies

The build system architecture showing how different components are built and packaged:

```mermaid
graph LR
    subgraph "Build System Architecture"
        subgraph "Build Tools"
            CMAKE[CMake]
            PYTHON_BUILD[Python Setup]
            NPM[NPM Build]
            GRADLE[Gradle Build]
            XCODE[Xcode Build]
        end
        
        subgraph "Core Dependencies"
            ONNX[ONNX Library]
            PROTOBUF[Protocol Buffers]
            EIGEN[Eigen Library]
            FLATBUF[FlatBuffers]
            ABSEIL[Abseil Library]
        end
        
        subgraph "Provider Dependencies"
            CUDNN[cuDNN]
            TENSORRT_LIB[TensorRT]
            DNNL_LIB[oneDNN]
            COREML_LIB[CoreML]
            DIRECTML[DirectML]
        end
        
        subgraph "Platform Dependencies"
            VCPKG[vcpkg]
            CONAN[Conan]
            APT[APT Packages]
            HOMEBREW[Homebrew]
            NUGET[NuGet Packages]
        end
        
        subgraph "Output Artifacts"
            SHARED_LIB[Shared Libraries]
            STATIC_LIB[Static Libraries]
            PYTHON_WHEEL[Python Wheels]
            NPM_PKG[NPM Packages]
            NUGET_PKG[NuGet Packages]
            FRAMEWORK[iOS Framework]
            AAR[Android AAR]
        end
    end
    
    CMAKE --> ONNX
    CMAKE --> PROTOBUF
    CMAKE --> EIGEN
    CMAKE --> FLATBUF
    CMAKE --> ABSEIL
    
    CMAKE --> CUDNN
    CMAKE --> TENSORRT_LIB
    CMAKE --> DNNL_LIB
    CMAKE --> COREML_LIB
    CMAKE --> DIRECTML
    
    VCPKG --> CMAKE
    CONAN --> CMAKE
    APT --> CMAKE
    HOMEBREW --> CMAKE
    NUGET --> CMAKE
    
    CMAKE --> SHARED_LIB
    CMAKE --> STATIC_LIB
    PYTHON_BUILD --> PYTHON_WHEEL
    NPM --> NPM_PKG
    CMAKE --> NUGET_PKG
    XCODE --> FRAMEWORK
    GRADLE --> AAR
```

## Deployment Architectures

Different deployment patterns and architectures supported by ONNX Runtime:

```mermaid
graph TB
    subgraph "Deployment Architectures"
        subgraph "Cloud Deployment"
            CLOUD_API[REST API Server]
            CLOUD_BATCH[Batch Processing]
            CLOUD_STREAM[Streaming]
            CLOUD_SCALE[Auto Scaling]
        end
        
        subgraph "Edge Deployment"
            EDGE_DEVICE[Edge Devices]
            EDGE_IOT[IoT Devices]
            EDGE_MOBILE[Mobile Apps]
            EDGE_EMBEDDED[Embedded Systems]
        end
        
        subgraph "Web Deployment"
            WEB_BROWSER[Browser Runtime]
            WEB_NODE[Node.js Server]
            WEB_WORKER[Web Workers]
            WEB_SERVICE[Web Services]
        end
        
        subgraph "Enterprise Deployment"
            ENT_DATACENTER[Data Center]
            ENT_HYBRID[Hybrid Cloud]
            ENT_ONPREM[On-Premises]
            ENT_CONTAINER[Containerized]
        end
    end
    
    subgraph "Runtime Configurations"
        CONFIG_PERF[High Performance]
        CONFIG_SIZE[Minimal Size]
        CONFIG_POWER[Low Power]
        CONFIG_LATENCY[Low Latency]
    end
    
    CLOUD_API --> CONFIG_PERF
    CLOUD_BATCH --> CONFIG_PERF
    CLOUD_STREAM --> CONFIG_LATENCY
    CLOUD_SCALE --> CONFIG_PERF
    
    EDGE_DEVICE --> CONFIG_POWER
    EDGE_IOT --> CONFIG_SIZE
    EDGE_MOBILE --> CONFIG_SIZE
    EDGE_EMBEDDED --> CONFIG_SIZE
    
    WEB_BROWSER --> CONFIG_SIZE
    WEB_NODE --> CONFIG_PERF
    WEB_WORKER --> CONFIG_LATENCY
    WEB_SERVICE --> CONFIG_PERF
    
    ENT_DATACENTER --> CONFIG_PERF
    ENT_HYBRID --> CONFIG_PERF
    ENT_ONPREM --> CONFIG_PERF
    ENT_CONTAINER --> CONFIG_LATENCY
```

## Memory Management

Memory management architecture across different execution providers:

```mermaid
graph TB
    subgraph "Memory Management Architecture"
        subgraph "Memory Allocators"
            CPU_ALLOC[CPU Allocator]
            GPU_ALLOC[GPU Allocator]
            ARENA_ALLOC[Arena Allocator]
            POOL_ALLOC[Pool Allocator]
        end
        
        subgraph "Memory Optimization"
            MEM_PLAN[Memory Planning]
            MEM_REUSE[Memory Reuse]
            MEM_SHARING[Memory Sharing]
            MEM_PATTERN[Memory Pattern]
        end
        
        subgraph "Memory Types"
            HOST_MEM[Host Memory]
            DEVICE_MEM[Device Memory]
            SHARED_MEM[Shared Memory]
            PINNED_MEM[Pinned Memory]
        end
        
        subgraph "Memory Operations"
            COPY_H2D[Host to Device Copy]
            COPY_D2H[Device to Host Copy]
            COPY_D2D[Device to Device Copy]
            SYNC_OP[Synchronization]
        end
    end
    
    CPU_ALLOC --> HOST_MEM
    GPU_ALLOC --> DEVICE_MEM
    ARENA_ALLOC --> SHARED_MEM
    POOL_ALLOC --> PINNED_MEM
    
    MEM_PLAN --> MEM_REUSE
    MEM_REUSE --> MEM_SHARING
    MEM_SHARING --> MEM_PATTERN
    
    HOST_MEM --> COPY_H2D
    DEVICE_MEM --> COPY_D2H
    DEVICE_MEM --> COPY_D2D
    COPY_H2D --> SYNC_OP
    COPY_D2H --> SYNC_OP
    COPY_D2D --> SYNC_OP
    
    MEM_PATTERN --> CPU_ALLOC
    MEM_PATTERN --> GPU_ALLOC
```

## Training vs Inference

Architectural differences between training and inference modes:

```mermaid
graph TB
    subgraph "Training Architecture"
        subgraph "Training Components"
            ORTMODULE[ORTModule]
            BACKWARD[Backward Pass]
            GRADIENT[Gradient Computation]
            OPTIMIZER[Optimizer Integration]
        end
        
        subgraph "Training Features"
            CHECKPOINT[Checkpointing]
            MIXED_PREC[Mixed Precision]
            GRAD_ACCUM[Gradient Accumulation]
            MEM_OPT[Memory Optimization]
        end
    end
    
    subgraph "Inference Architecture"
        subgraph "Inference Components"
            INFERENCE_SESSION[Inference Session]
            FORWARD[Forward Pass Only]
            PREDICTION[Prediction Output]
            BATCH_PROC[Batch Processing]
        end
        
        subgraph "Inference Features"
            QUANTIZATION[Quantization]
            PRUNING[Model Pruning]
            BATCHING[Dynamic Batching]
            CACHING[Result Caching]
        end
    end
    
    subgraph "Shared Components"
        GRAPH_OPT[Graph Optimization]
        EXEC_PROVIDERS[Execution Providers]
        KERNEL_LIB[Kernel Library]
        MEMORY_MGR[Memory Manager]
    end
    
    ORTMODULE --> BACKWARD
    BACKWARD --> GRADIENT
    GRADIENT --> OPTIMIZER
    
    CHECKPOINT --> MEM_OPT
    MIXED_PREC --> GRAD_ACCUM
    GRAD_ACCUM --> MEM_OPT
    
    INFERENCE_SESSION --> FORWARD
    FORWARD --> PREDICTION
    PREDICTION --> BATCH_PROC
    
    QUANTIZATION --> PRUNING
    PRUNING --> BATCHING
    BATCHING --> CACHING
    
    ORTMODULE --> GRAPH_OPT
    INFERENCE_SESSION --> GRAPH_OPT
    GRAPH_OPT --> EXEC_PROVIDERS
    EXEC_PROVIDERS --> KERNEL_LIB
    KERNEL_LIB --> MEMORY_MGR
    
    BACKWARD -.-> |"Uses"| EXEC_PROVIDERS
    FORWARD -.-> |"Uses"| EXEC_PROVIDERS
```

## Component Interaction Sequence

Detailed sequence of interactions during model execution:

```mermaid
sequenceDiagram
    participant App as Application
    participant API as Language API
    participant Session as Inference Session
    participant Graph as Graph Engine
    participant Provider as Execution Provider
    participant Kernel as Kernel
    participant Memory as Memory Manager
    
    App->>API: Load Model
    API->>Session: Create Session
    Session->>Graph: Parse Model
    Graph->>Graph: Validate Graph
    Graph->>Graph: Optimize Graph
    Session->>Provider: Select Provider
    Provider->>Kernel: Register Kernels
    Kernel->>Memory: Allocate Memory
    
    App->>API: Run Inference
    API->>Session: Execute
    Session->>Graph: Schedule Execution
    Graph->>Provider: Execute Subgraph
    Provider->>Kernel: Run Kernel
    Kernel->>Memory: Access Memory
    Memory-->>Kernel: Return Data
    Kernel-->>Provider: Return Result
    Provider-->>Graph: Return Result
    Graph-->>Session: Return Result
    Session-->>API: Return Output
    API-->>App: Return Tensors
```

---

## Related Documentation

This document provides a comprehensive view of the ONNX Runtime architecture. For more detailed information about specific components, refer to the following documentation:

### Detailed Architecture Documentation
- **[Architecture Directory](architecture/)** - Detailed architectural documentation
- **[Execution Providers](architecture/execution_providers.md)** - Comprehensive execution provider architecture
- **[Graph Processing](architecture/graph_processing.md)** - Graph optimization and transformation pipeline
- **[Language Bindings](architecture/language_bindings.md)** - Multi-language API implementation details
- **[Memory Management](architecture/memory_management.md)** - Advanced memory management strategies
- **[Build System](architecture/build_system.md)** - Build system architecture and CI/CD
- **[Deployment Patterns](architecture/deployment_patterns.md)** - Deployment strategies and patterns
- **[Training vs Inference](architecture/training_vs_inference.md)** - Architectural differences between modes

### Implementation Guides
- **[Memory Optimizer](Memory_Optimizer.md)** - Memory optimization techniques and configuration
- **[C API Guidelines](C_API_Guidelines.md)** - C API design and usage principles
- **[Build Guidelines](cmake_guideline.md)** - Build system configuration and usage

### Provider-Specific Documentation
- **[Execution Providers](execution_providers/)** - Provider-specific implementation details
- **[Python API Documentation](python/)** - Python binding implementation
- **[C/C++ Documentation](c_cxx/)** - Native API documentation

### Additional Resources
- **[FAQ](FAQ.md)** - Frequently asked questions
- **[Coding Conventions](Coding_Conventions_and_Standards.md)** - Development standards
- **[Contributing Guidelines](../CONTRIBUTING.md)** - How to contribute to the project