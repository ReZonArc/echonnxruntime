# Execution Provider Architecture

This document details the execution provider architecture in ONNX Runtime, which enables support for diverse hardware accelerators and runtime environments.

## Overview

Execution Providers (EPs) are the abstraction layer that allows ONNX Runtime to leverage different hardware and software acceleration libraries. Each provider implements a common interface while providing hardware-specific optimizations.

## Execution Provider Interface

```mermaid
classDiagram
    class IExecutionProvider {
        <<interface>>
        +GetCapability() ExecutionProviderCapability
        +Compile() Status
        +CreateKernel() OpKernel
        +GetDataLayout() DataLayout
        +GetAllocator() AllocatorPtr
    }
    
    class ExecutionProviderCapability {
        +sub_graph : GraphViewer
        +required_input_count : int
        +supported_ops : vector~string~
    }
    
    class CPUExecutionProvider {
        +GetCapability() ExecutionProviderCapability
        +CreateKernel() OpKernel
    }
    
    class CUDAExecutionProvider {
        +GetCapability() ExecutionProviderCapability
        +CreateKernel() OpKernel
        +GetCudaStream() cudaStream_t
    }
    
    class CoreMLExecutionProvider {
        +GetCapability() ExecutionProviderCapability
        +CreateKernel() OpKernel
        +CompileModel() Status
    }
    
    IExecutionProvider <|-- CPUExecutionProvider
    IExecutionProvider <|-- CUDAExecutionProvider
    IExecutionProvider <|-- CoreMLExecutionProvider
    IExecutionProvider --> ExecutionProviderCapability
```

## Provider Selection and Fallback

The execution provider selection follows a priority-based system with fallback mechanisms:

```mermaid
graph TD
    subgraph "Provider Selection Flow"
        START[Model Loading]
        ANALYZE[Analyze Graph]
        SELECT[Select Providers]
        PARTITION[Graph Partitioning]
        FALLBACK[Fallback Handling]
        EXECUTE[Execute]
    end
    
    subgraph "Provider Priority"
        P1[Custom Provider]
        P2[Hardware Accelerated]
        P3[Optimized CPU]
        P4[Default CPU]
    end
    
    subgraph "Partitioning Logic"
        CHECK_SUPPORT[Check Op Support]
        CREATE_SUBGRAPH[Create Subgraph]
        ASSIGN_PROVIDER[Assign Provider]
        VALIDATE[Validate Assignment]
    end
    
    START --> ANALYZE
    ANALYZE --> SELECT
    SELECT --> P1
    P1 --> P2
    P2 --> P3
    P3 --> P4
    P4 --> PARTITION
    
    PARTITION --> CHECK_SUPPORT
    CHECK_SUPPORT --> CREATE_SUBGRAPH
    CREATE_SUBGRAPH --> ASSIGN_PROVIDER
    ASSIGN_PROVIDER --> VALIDATE
    VALIDATE --> FALLBACK
    FALLBACK --> EXECUTE
    
    VALIDATE -.-> |Unsupported| CHECK_SUPPORT
```

## Provider Categories

### CPU Providers

```mermaid
graph LR
    subgraph "CPU Execution Providers"
        CPU[CPU Provider]
        DNNL[oneDNN Provider]
        XNNPACK[XNNPACK Provider]
        NNAPI[NNAPI Provider]
        
        subgraph "CPU Features"
            SIMD[SIMD Instructions]
            THREADING[Multi-threading]
            VECTORIZATION[Auto-vectorization]
            MEMORY_OPT[Memory Optimization]
        end
    end
    
    CPU --> SIMD
    DNNL --> SIMD
    DNNL --> THREADING
    XNNPACK --> VECTORIZATION
    NNAPI --> MEMORY_OPT
    
    CPU -.-> |Fallback| DNNL
    DNNL -.-> |Mobile| XNNPACK
    XNNPACK -.-> |Android| NNAPI
```

### GPU Providers

```mermaid
graph TB
    subgraph "GPU Execution Providers"
        CUDA[CUDA Provider]
        ROCM[ROCm Provider]
        DML[DirectML Provider]
        OPENVINO[OpenVINO Provider]
        
        subgraph "GPU Optimizations"
            KERNEL_FUSION[Kernel Fusion]
            MEMORY_POOL[Memory Pooling]
            STREAM_EXEC[Stream Execution]
            TENSOR_CORE[Tensor Core Usage]
        end
        
        subgraph "Hardware Targets"
            NVIDIA[NVIDIA GPUs]
            AMD[AMD GPUs]
            INTEL[Intel GPUs]
            QUALCOMM[Qualcomm GPUs]
        end
    end
    
    CUDA --> KERNEL_FUSION
    CUDA --> TENSOR_CORE
    CUDA --> NVIDIA
    
    ROCM --> STREAM_EXEC
    ROCM --> AMD
    
    DML --> MEMORY_POOL
    DML --> INTEL
    
    OPENVINO --> KERNEL_FUSION
    OPENVINO --> INTEL
    OPENVINO --> QUALCOMM
```

### Specialized Providers

```mermaid
graph TD
    subgraph "Specialized Execution Providers"
        TENSORRT[TensorRT Provider]
        COREML[CoreML Provider]
        QNN[QNN Provider]
        SNPE[SNPE Provider]
        
        subgraph "Optimization Features"
            INT8_QUANT[INT8 Quantization]
            LAYER_FUSION[Layer Fusion]
            DYNAMIC_SHAPE[Dynamic Shapes]
            CALIBRATION[Calibration]
        end
        
        subgraph "Target Platforms"
            DATACENTER[Data Center]
            MOBILE_IOS[iOS Devices]
            MOBILE_ANDROID[Android Devices]
            EDGE_DEVICES[Edge Devices]
        end
    end
    
    TENSORRT --> INT8_QUANT
    TENSORRT --> LAYER_FUSION
    TENSORRT --> DATACENTER
    
    COREML --> DYNAMIC_SHAPE
    COREML --> MOBILE_IOS
    
    QNN --> CALIBRATION
    QNN --> MOBILE_ANDROID
    
    SNPE --> INT8_QUANT
    SNPE --> EDGE_DEVICES
```

### Web Providers

```mermaid
graph LR
    subgraph "Web Execution Providers"
        WEBGL[WebGL Provider]
        WEBGPU[WebGPU Provider]
        WEBNN[WebNN Provider]
        JSEP[JSEP Provider]
        
        subgraph "Web Features"
            SHADER[Shader Programs]
            COMPUTE[Compute Shaders]
            WASM[WebAssembly]
            WORKER[Web Workers]
        end
        
        subgraph "Browser Support"
            CHROME[Chrome/Chromium]
            FIREFOX[Firefox]
            SAFARI[Safari]
            EDGE[Edge]
        end
    end
    
    WEBGL --> SHADER
    WEBGPU --> COMPUTE
    WEBNN --> WASM
    JSEP --> WORKER
    
    WEBGL --> CHROME
    WEBGL --> FIREFOX
    WEBGPU --> CHROME
    WEBNN --> CHROME
    WEBNN --> EDGE
    JSEP --> SAFARI
```

## Provider Lifecycle

The lifecycle of an execution provider from registration to execution:

```mermaid
sequenceDiagram
    participant Registry as Provider Registry
    participant Factory as Provider Factory
    participant Provider as Execution Provider
    participant Session as Inference Session
    participant Graph as Graph Partitioner
    participant Kernel as Kernel Executor
    
    Note over Registry, Factory: Provider Registration
    Factory->>Registry: Register Provider
    Registry->>Registry: Store Provider Info
    
    Note over Session, Provider: Session Creation
    Session->>Registry: Get Available Providers
    Registry-->>Session: Return Provider List
    Session->>Factory: Create Provider Instance
    Factory-->>Session: Return Provider
    
    Note over Session, Graph: Graph Partitioning
    Session->>Provider: Get Capability
    Provider-->>Session: Return Supported Ops
    Session->>Graph: Partition Graph
    Graph->>Provider: Assign Subgraph
    
    Note over Provider, Kernel: Execution
    Provider->>Provider: Compile Subgraph
    Provider->>Kernel: Create Kernel
    Session->>Provider: Execute
    Provider->>Kernel: Run Kernel
    Kernel-->>Provider: Return Result
    Provider-->>Session: Return Output
```

## Graph Partitioning

How the graph is partitioned across multiple execution providers:

```mermaid
graph TB
    subgraph "Original Graph"
        OP1[Conv2D]
        OP2[BatchNorm]
        OP3[ReLU]
        OP4[MaxPool]
        OP5[Flatten]
        OP6[MatMul]
        OP7[Softmax]
        
        OP1 --> OP2
        OP2 --> OP3
        OP3 --> OP4
        OP4 --> OP5
        OP5 --> OP6
        OP6 --> OP7
    end
    
    subgraph "Partitioned Graph"
        subgraph "CUDA Subgraph"
            COP1[Conv2D]
            COP2[BatchNorm]
            COP3[ReLU]
            COP4[MaxPool]
            
            COP1 --> COP2
            COP2 --> COP3
            COP3 --> COP4
        end
        
        subgraph "CPU Subgraph"
            CCOP5[Flatten]
            CCOP6[MatMul]
            CCOP7[Softmax]
            
            CCOP5 --> CCOP6
            CCOP6 --> CCOP7
        end
        
        subgraph "Memory Transfers"
            H2D[Host to Device]
            D2H[Device to Host]
        end
    end
    
    COP4 --> H2D
    H2D --> D2H
    D2H --> CCOP5
```

## Provider-Specific Optimizations

Each provider implements specific optimizations:

```mermaid
graph TD
    subgraph "Provider Optimizations"
        subgraph "CUDA Optimizations"
            CUDA_FUSION[Operator Fusion]
            CUDA_MEMORY[Memory Optimization]
            CUDA_STREAMS[CUDA Streams]
            CUDA_GRAPH[CUDA Graphs]
        end
        
        subgraph "CPU Optimizations"
            CPU_SIMD[SIMD Vectorization]
            CPU_THREADS[Thread Parallelism]
            CPU_CACHE[Cache Optimization]
            CPU_LAYOUT[Memory Layout]
        end
        
        subgraph "WebGL Optimizations"
            WEBGL_TEXTURE[Texture Packing]
            WEBGL_SHADER[Shader Optimization]
            WEBGL_BATCH[Batch Operations]
            WEBGL_PRECISION[Precision Handling]
        end
        
        subgraph "TensorRT Optimizations"
            TRT_FUSION[Layer Fusion]
            TRT_PRECISION[Mixed Precision]
            TRT_CALIBRATION[INT8 Calibration]
            TRT_DYNAMIC[Dynamic Shapes]
        end
    end
    
    CUDA_FUSION --> CUDA_STREAMS
    CUDA_MEMORY --> CUDA_GRAPH
    
    CPU_SIMD --> CPU_THREADS
    CPU_CACHE --> CPU_LAYOUT
    
    WEBGL_TEXTURE --> WEBGL_SHADER
    WEBGL_BATCH --> WEBGL_PRECISION
    
    TRT_FUSION --> TRT_PRECISION
    TRT_CALIBRATION --> TRT_DYNAMIC
```

## Error Handling and Fallback

Error handling and fallback mechanisms in execution providers:

```mermaid
flowchart TD
    START[Execute Operator]
    CHECK_SUPPORT{Provider Supports Op?}
    EXECUTE_PRIMARY[Execute with Primary Provider]
    CHECK_SUCCESS{Execution Successful?}
    EXECUTE_FALLBACK[Execute with Fallback Provider]
    CHECK_FALLBACK_SUCCESS{Fallback Successful?}
    RETURN_SUCCESS[Return Success]
    RETURN_ERROR[Return Error]
    
    START --> CHECK_SUPPORT
    CHECK_SUPPORT -->|Yes| EXECUTE_PRIMARY
    CHECK_SUPPORT -->|No| EXECUTE_FALLBACK
    
    EXECUTE_PRIMARY --> CHECK_SUCCESS
    CHECK_SUCCESS -->|Yes| RETURN_SUCCESS
    CHECK_SUCCESS -->|No| EXECUTE_FALLBACK
    
    EXECUTE_FALLBACK --> CHECK_FALLBACK_SUCCESS
    CHECK_FALLBACK_SUCCESS -->|Yes| RETURN_SUCCESS
    CHECK_FALLBACK_SUCCESS -->|No| RETURN_ERROR
```

## Performance Considerations

Key performance considerations for execution providers:

```mermaid
mindmap
    root((Performance))
        Memory Management
            Allocation Strategy
            Memory Pools
            Data Locality
            Transfer Overhead
        Compute Optimization
            Kernel Fusion
            Parallelization
            Vectorization
            Precision Selection
        Hardware Utilization
            Resource Scheduling
            Load Balancing
            Power Management
            Thermal Throttling
        Software Optimization
            Compiler Optimizations
            Runtime Adaptation
            Caching Strategies
            Pipeline Optimization
```

## Provider Configuration

Configuration options available for different execution providers:

```mermaid
graph LR
    subgraph "Provider Configuration"
        subgraph "Common Config"
            DEVICE_ID[Device ID]
            MEMORY_LIMIT[Memory Limit]
            THREAD_COUNT[Thread Count]
            PRECISION[Precision Mode]
        end
        
        subgraph "CUDA Config"
            CUDA_STREAM[CUDA Stream]
            CUDA_MEMORY_POOL[Memory Pool Size]
            CUDA_GRAPH_ENABLE[CUDA Graph]
            CUDA_CONV_ALGO[Convolution Algorithm]
        end
        
        subgraph "CPU Config"
            CPU_NUMA[NUMA Policy]
            CPU_AFFINITY[Thread Affinity]
            CPU_INTER_OP[Inter-op Threads]
            CPU_INTRA_OP[Intra-op Threads]
        end
        
        subgraph "Provider Selection"
            PRIORITY[Provider Priority]
            FALLBACK[Fallback Chain]
            EXCLUDE[Exclude Providers]
            FORCE[Force Provider]
        end
    end
    
    DEVICE_ID --> CUDA_STREAM
    MEMORY_LIMIT --> CUDA_MEMORY_POOL
    THREAD_COUNT --> CPU_INTER_OP
    PRECISION --> CUDA_CONV_ALGO
    
    PRIORITY --> FALLBACK
    FALLBACK --> EXCLUDE
    EXCLUDE --> FORCE
```

This execution provider architecture enables ONNX Runtime to provide optimal performance across a wide range of hardware platforms while maintaining a consistent programming interface for developers.