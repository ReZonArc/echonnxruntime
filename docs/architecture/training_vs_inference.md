# Training vs Inference Architecture

This document details the architectural differences between ONNX Runtime's training and inference modes, highlighting specialized components, optimizations, and use cases for each mode.

## Overview

ONNX Runtime supports both inference and training workloads with distinct architectural optimizations. While sharing core components, each mode has specialized features tailored to its specific requirements.

## Architectural Comparison

```mermaid
graph TB
    subgraph "ONNX Runtime Architecture Comparison"
        subgraph "Shared Components"
            CORE_RUNTIME[Core Runtime]
            GRAPH_ENGINE[Graph Engine]
            MEMORY_MANAGER[Memory Manager]
            EXECUTION_PROVIDERS[Execution Providers]
        end
        
        subgraph "Inference-Specific"
            INFERENCE_SESSION[Inference Session]
            MODEL_OPTIMIZATION[Model Optimization]
            QUANTIZATION[Quantization]
            BATCHING[Dynamic Batching]
        end
        
        subgraph "Training-Specific"
            TRAINING_SESSION[Training Session]
            ORTMODULE[ORTModule]
            GRADIENT_ENGINE[Gradient Engine]
            CHECKPOINT_MANAGER[Checkpoint Manager]
        end
        
        subgraph "Optimization Differences"
            INFERENCE_OPTS[Inference Optimizations]
            TRAINING_OPTS[Training Optimizations]
            MEMORY_STRATEGIES[Memory Strategies]
            COMPUTE_PATTERNS[Compute Patterns]
        end
    end
    
    CORE_RUNTIME --> INFERENCE_SESSION
    CORE_RUNTIME --> TRAINING_SESSION
    GRAPH_ENGINE --> MODEL_OPTIMIZATION
    GRAPH_ENGINE --> ORTMODULE
    MEMORY_MANAGER --> QUANTIZATION
    MEMORY_MANAGER --> GRADIENT_ENGINE
    EXECUTION_PROVIDERS --> BATCHING
    EXECUTION_PROVIDERS --> CHECKPOINT_MANAGER
    
    INFERENCE_SESSION --> INFERENCE_OPTS
    TRAINING_SESSION --> TRAINING_OPTS
    MODEL_OPTIMIZATION --> MEMORY_STRATEGIES
    ORTMODULE --> COMPUTE_PATTERNS
```

## Inference Architecture

### Inference Pipeline

```mermaid
flowchart TD
    START[Model Loading]
    PARSE[Parse ONNX Model]
    VALIDATE[Validate Model]
    OPTIMIZE[Graph Optimization]
    PARTITION[Graph Partitioning]
    COMPILE[Kernel Compilation]
    INITIALIZE[Session Initialization]
    
    subgraph "Inference Execution"
        INPUT[Input Preparation]
        PREPROCESS[Preprocessing]
        FORWARD[Forward Pass]
        POSTPROCESS[Postprocessing]
        OUTPUT[Output Generation]
    end
    
    START --> PARSE
    PARSE --> VALIDATE
    VALIDATE --> OPTIMIZE
    OPTIMIZE --> PARTITION
    PARTITION --> COMPILE
    COMPILE --> INITIALIZE
    
    INITIALIZE --> INPUT
    INPUT --> PREPROCESS
    PREPROCESS --> FORWARD
    FORWARD --> POSTPROCESS
    POSTPROCESS --> OUTPUT
    
    OUTPUT -.-> INPUT
```

### Inference Optimizations

```mermaid
graph LR
    subgraph "Inference-Specific Optimizations"
        subgraph "Graph Optimizations"
            CONSTANT_FOLDING[Constant Folding]
            OPERATOR_FUSION[Operator Fusion]
            LAYOUT_OPTIMIZATION[Layout Optimization]
            DEAD_CODE_ELIMINATION[Dead Code Elimination]
        end
        
        subgraph "Runtime Optimizations"
            KERNEL_SELECTION[Optimal Kernel Selection]
            MEMORY_REUSE[Memory Reuse]
            PARALLEL_EXECUTION[Parallel Execution]
            CACHING[Result Caching]
        end
        
        subgraph "Model Optimizations"
            QUANTIZATION_INT8[INT8 Quantization]
            PRUNING[Model Pruning]
            DISTILLATION[Knowledge Distillation]
            DYNAMIC_SHAPES[Dynamic Shape Handling]
        end
        
        subgraph "Hardware Optimizations"
            SIMD_VECTORIZATION[SIMD Vectorization]
            GPU_ACCELERATION[GPU Acceleration]
            MIXED_PRECISION[Mixed Precision]
            TENSOR_CORE[Tensor Core Usage]
        end
    end
    
    CONSTANT_FOLDING --> KERNEL_SELECTION
    OPERATOR_FUSION --> MEMORY_REUSE
    LAYOUT_OPTIMIZATION --> PARALLEL_EXECUTION
    DEAD_CODE_ELIMINATION --> CACHING
    
    KERNEL_SELECTION --> QUANTIZATION_INT8
    MEMORY_REUSE --> PRUNING
    PARALLEL_EXECUTION --> DISTILLATION
    CACHING --> DYNAMIC_SHAPES
    
    QUANTIZATION_INT8 --> SIMD_VECTORIZATION
    PRUNING --> GPU_ACCELERATION
    DISTILLATION --> MIXED_PRECISION
    DYNAMIC_SHAPES --> TENSOR_CORE
```

## Training Architecture

### ORTModule Architecture

```mermaid
classDiagram
    class ORTModule {
        +forward() Tensor
        +backward() void
        +training: bool
        +_torch_module: torch.nn.Module
        +_training_manager: TrainingManager
        +_gradient_graph: GradientGraph
    }
    
    class TrainingManager {
        +create_training_session() TrainingSession
        +get_gradient_graph() GradientGraph
        +optimize_training_graph() Status
        +checkpoint() Status
    }
    
    class GradientGraph {
        +build_gradient_graph() Status
        +backward_pass_builder: BackwardPassBuilder
        +loss_scaler: LossScaler
        +gradient_accumulation: GradientAccumulation
    }
    
    class TrainingSession {
        +run_forward() OrtValues
        +run_backward() OrtValues
        +update_weights() Status
        +get_gradients() OrtValues
    }
    
    class BackwardPassBuilder {
        +build_gradient_operators() Status
        +create_gradient_graph() GraphProto
        +optimize_gradient_computation() Status
    }
    
    ORTModule --> TrainingManager
    TrainingManager --> GradientGraph
    TrainingManager --> TrainingSession
    GradientGraph --> BackwardPassBuilder
    TrainingSession --> BackwardPassBuilder
```

### Training Pipeline

```mermaid
sequenceDiagram
    participant PyTorch as PyTorch Frontend
    participant ORTModule as ORTModule
    participant TrainingManager as Training Manager
    participant GradientGraph as Gradient Graph
    participant TrainingSession as Training Session
    participant ExecutionProvider as Execution Provider
    
    Note over PyTorch, ExecutionProvider: Initialization Phase
    PyTorch->>ORTModule: Create ORTModule
    ORTModule->>TrainingManager: Initialize Training
    TrainingManager->>GradientGraph: Build Gradient Graph
    GradientGraph->>TrainingSession: Create Training Session
    
    Note over PyTorch, ExecutionProvider: Training Loop
    loop Training Iteration
        PyTorch->>ORTModule: forward(inputs)
        ORTModule->>TrainingSession: Run Forward Pass
        TrainingSession->>ExecutionProvider: Execute Forward
        ExecutionProvider-->>TrainingSession: Forward Results
        TrainingSession-->>ORTModule: Forward Outputs
        ORTModule-->>PyTorch: Return Outputs
        
        PyTorch->>ORTModule: backward(loss)
        ORTModule->>TrainingSession: Run Backward Pass
        TrainingSession->>ExecutionProvider: Execute Backward
        ExecutionProvider-->>TrainingSession: Gradients
        TrainingSession-->>ORTModule: Return Gradients
        ORTModule-->>PyTorch: Gradients Ready
        
        PyTorch->>PyTorch: Update Weights
    end
```

## Memory Management Differences

### Inference Memory Patterns

```mermaid
graph TD
    subgraph "Inference Memory Management"
        subgraph "Memory Allocation"
            STATIC_ALLOCATION[Static Allocation]
            PRE_ALLOCATION[Pre-allocation]
            ARENA_ALLOCATION[Arena Allocation]
        end
        
        subgraph "Memory Reuse"
            TENSOR_REUSE[Tensor Memory Reuse]
            BUFFER_SHARING[Buffer Sharing]
            IN_PLACE_OPS[In-place Operations]
        end
        
        subgraph "Memory Optimization"
            MEMORY_PLANNING[Memory Planning]
            LAYOUT_TRANSFORMATION[Layout Transformation]
            COMPRESSION[Memory Compression]
        end
        
        subgraph "Inference Patterns"
            FORWARD_ONLY[Forward Pass Only]
            NO_GRADIENT[No Gradient Storage]
            IMMEDIATE_CLEANUP[Immediate Cleanup]
        end
    end
    
    STATIC_ALLOCATION --> TENSOR_REUSE
    PRE_ALLOCATION --> BUFFER_SHARING
    ARENA_ALLOCATION --> IN_PLACE_OPS
    
    TENSOR_REUSE --> MEMORY_PLANNING
    BUFFER_SHARING --> LAYOUT_TRANSFORMATION
    IN_PLACE_OPS --> COMPRESSION
    
    MEMORY_PLANNING --> FORWARD_ONLY
    LAYOUT_TRANSFORMATION --> NO_GRADIENT
    COMPRESSION --> IMMEDIATE_CLEANUP
```

### Training Memory Patterns

```mermaid
graph TD
    subgraph "Training Memory Management"
        subgraph "Gradient Storage"
            ACTIVATION_STORAGE[Activation Storage]
            GRADIENT_BUFFER[Gradient Buffers]
            INTERMEDIATE_TENSORS[Intermediate Tensors]
        end
        
        subgraph "Memory Strategy"
            RECOMPUTATION[Activation Recomputation]
            GRADIENT_ACCUMULATION[Gradient Accumulation]
            MEMORY_OFFLOADING[Memory Offloading]
        end
        
        subgraph "Optimization Techniques"
            ZERO_OPTIMIZER[ZeRO Optimizer]
            GRADIENT_COMPRESSION[Gradient Compression]
            MIXED_PRECISION_TRAIN[Mixed Precision Training]
        end
        
        subgraph "Training Patterns"
            FORWARD_BACKWARD[Forward-Backward Pass]
            GRADIENT_STORAGE_PATTERN[Gradient Storage]
            CHECKPOINT_MEMORY[Checkpoint Memory]
        end
    end
    
    ACTIVATION_STORAGE --> RECOMPUTATION
    GRADIENT_BUFFER --> GRADIENT_ACCUMULATION
    INTERMEDIATE_TENSORS --> MEMORY_OFFLOADING
    
    RECOMPUTATION --> ZERO_OPTIMIZER
    GRADIENT_ACCUMULATION --> GRADIENT_COMPRESSION
    MEMORY_OFFLOADING --> MIXED_PRECISION_TRAIN
    
    ZERO_OPTIMIZER --> FORWARD_BACKWARD
    GRADIENT_COMPRESSION --> GRADIENT_STORAGE_PATTERN
    MIXED_PRECISION_TRAIN --> CHECKPOINT_MEMORY
```

## Performance Characteristics

### Inference Performance Focus

```mermaid
mindmap
    root((Inference Performance))
        Latency Optimization
            Single Request Latency
            Batch Processing
            Pipeline Parallelism
            Cache Hit Rates
        Throughput Optimization
            Concurrent Requests
            Resource Utilization
            Load Balancing
            Connection Pooling
        Resource Efficiency
            Memory Usage
            CPU Utilization
            GPU Utilization
            Power Consumption
        Scalability
            Horizontal Scaling
            Auto-scaling
            Load Distribution
            Resource Elasticity
```

### Training Performance Focus

```mermaid
mindmap
    root((Training Performance))
        Training Speed
            Forward Pass Speed
            Backward Pass Speed
            Gradient Computation
            Parameter Updates
        Memory Efficiency
            Gradient Accumulation
            Activation Checkpointing
            Memory Recomputation
            Dynamic Memory Management
        Scale-out Training
            Data Parallelism
            Model Parallelism
            Pipeline Parallelism
            Distributed Training
        Convergence
            Gradient Quality
            Numerical Stability Mixed Precision
            Learning Rate Scheduling
```

## Execution Provider Differences

### Provider Optimization Focus

```mermaid
graph LR
    subgraph "Execution Provider Specialization"
        subgraph "Inference Focused"
            INF_CUDA[CUDA Inference]
            INF_TENSORRT[TensorRT]
            INF_COREML[CoreML]
            INF_DNNL[oneDNN Inference]
        end
        
        subgraph "Training Focused"
            TRAIN_CUDA[CUDA Training]
            TRAIN_ROCM[ROCm Training]
            TRAIN_CPU[CPU Training]
            TRAIN_DISTRIBUTED[Distributed Training]
        end
        
        subgraph "Inference Optimizations"
            KERNEL_FUSION_INF[Kernel Fusion]
            PRECISION_OPT[Precision Optimization]
            BATCH_OPT[Batch Optimization]
            CACHE_OPT[Cache Optimization]
        end
        
        subgraph "Training Optimizations"
            GRADIENT_OPT[Gradient Optimization]
            MEMORY_OPT_TRAIN[Memory Optimization]
            COMMUNICATION_OPT[Communication Optimization]
            CHECKPOINT_OPT[Checkpoint Optimization]
        end
    end
    
    INF_CUDA --> KERNEL_FUSION_INF
    INF_TENSORRT --> PRECISION_OPT
    INF_COREML --> BATCH_OPT
    INF_DNNL --> CACHE_OPT
    
    TRAIN_CUDA --> GRADIENT_OPT
    TRAIN_ROCM --> MEMORY_OPT_TRAIN
    TRAIN_CPU --> COMMUNICATION_OPT
    TRAIN_DISTRIBUTED --> CHECKPOINT_OPT
```

## Graph Optimization Differences

### Inference Graph Optimizations

```mermaid
graph TB
    subgraph "Inference Graph Optimizations"
        subgraph "Static Optimizations"
            CONST_FOLD_INF[Constant Folding]
            OP_FUSION_INF[Operator Fusion]
            LAYOUT_OPT_INF[Layout Optimization]
            SHAPE_OPT[Shape Optimization]
        end
        
        subgraph "Runtime Optimizations"
            KERNEL_SELECT[Kernel Selection]
            MEMORY_PLAN_INF[Memory Planning]
            EXECUTION_PLAN[Execution Planning]
            PARALLEL_OPT[Parallelization]
        end
        
        subgraph "Model Transformations"
            QUANTIZATION_TRANSFORM[Quantization]
            PRUNING_TRANSFORM[Pruning]
            SPARSITY_OPT[Sparsity Optimization]
            PRECISION_TRANSFORM[Precision Transformation]
        end
    end
    
    CONST_FOLD_INF --> KERNEL_SELECT
    OP_FUSION_INF --> MEMORY_PLAN_INF
    LAYOUT_OPT_INF --> EXECUTION_PLAN
    SHAPE_OPT --> PARALLEL_OPT
    
    KERNEL_SELECT --> QUANTIZATION_TRANSFORM
    MEMORY_PLAN_INF --> PRUNING_TRANSFORM
    EXECUTION_PLAN --> SPARSITY_OPT
    PARALLEL_OPT --> PRECISION_TRANSFORM
```

### Training Graph Optimizations

```mermaid
graph TB
    subgraph "Training Graph Optimizations"
        subgraph "Gradient Graph"
            GRADIENT_BUILDER[Gradient Graph Builder]
            BACKWARD_PASS[Backward Pass Optimization]
            GRADIENT_FUSION[Gradient Fusion]
            ACTIVATION_CHECKPOINTING[Activation Checkpointing]
        end
        
        subgraph "Memory Optimizations"
            RECOMPUTATION_OPT[Recomputation Optimization]
            GRADIENT_ACCUMULATION_OPT[Gradient Accumulation]
            MEMORY_OFFLOAD[Memory Offloading]
            ZERO_REDUNDANCY[Zero Redundancy Optimization]
        end
        
        subgraph "Communication Optimizations"
            ALLREDUCE_OPT[AllReduce Optimization]
            GRADIENT_COMPRESSION_OPT[Gradient Compression]
            OVERLAP_COMM[Communication Overlap]
            BANDWIDTH_OPT[Bandwidth Optimization]
        end
    end
    
    GRADIENT_BUILDER --> RECOMPUTATION_OPT
    BACKWARD_PASS --> GRADIENT_ACCUMULATION_OPT
    GRADIENT_FUSION --> MEMORY_OFFLOAD
    ACTIVATION_CHECKPOINTING --> ZERO_REDUNDANCY
    
    RECOMPUTATION_OPT --> ALLREDUCE_OPT
    GRADIENT_ACCUMULATION_OPT --> GRADIENT_COMPRESSION_OPT
    MEMORY_OFFLOAD --> OVERLAP_COMM
    ZERO_REDUNDANCY --> BANDWIDTH_OPT
```

## Distributed Training Architecture

```mermaid
graph TB
    subgraph "Distributed Training Architecture"
        subgraph "Data Parallelism"
            DATA_PARALLEL[Data Parallel Training]
            GRADIENT_SYNC[Gradient Synchronization]
            PARAMETER_SERVER[Parameter Server]
            ALLREDUCE[AllReduce Communication]
        end
        
        subgraph "Model Parallelism"
            MODEL_SHARDING[Model Sharding]
            PIPELINE_PARALLEL[Pipeline Parallelism]
            TENSOR_PARALLEL[Tensor Parallelism]
            EXPERT_PARALLEL[Expert Parallelism]
        end
        
        subgraph "Hybrid Approaches"
            MIXED_PARALLEL[Mixed Parallelism]
            ELASTIC_TRAINING[Elastic Training]
            FAULT_TOLERANCE[Fault Tolerance]
            DYNAMIC_SCALING[Dynamic Scaling]
        end
        
        subgraph "Communication Backend"
            NCCL[NCCL Backend]
            MPI[MPI Backend]
            GLOO[Gloo Backend]
            CUSTOM_BACKEND[Custom Backend]
        end
    end
    
    DATA_PARALLEL --> MODEL_SHARDING
    GRADIENT_SYNC --> PIPELINE_PARALLEL
    PARAMETER_SERVER --> TENSOR_PARALLEL
    ALLREDUCE --> EXPERT_PARALLEL
    
    MODEL_SHARDING --> MIXED_PARALLEL
    PIPELINE_PARALLEL --> ELASTIC_TRAINING
    TENSOR_PARALLEL --> FAULT_TOLERANCE
    EXPERT_PARALLEL --> DYNAMIC_SCALING
    
    MIXED_PARALLEL --> NCCL
    ELASTIC_TRAINING --> MPI
    FAULT_TOLERANCE --> GLOO
    DYNAMIC_SCALING --> CUSTOM_BACKEND
```

## Configuration Differences

### Inference Configuration

```mermaid
graph LR
    subgraph "Inference Configuration"
        subgraph "Performance Settings"
            BATCH_SIZE[Batch Size]
            THREAD_COUNT[Thread Count]
            INTER_OP_THREADS[Inter-op Threads]
            INTRA_OP_THREADS[Intra-op Threads]
        end
        
        subgraph "Optimization Settings"
            GRAPH_OPT_LEVEL[Graph Optimization Level]
            EXECUTION_MODE[Execution Mode]
            PROVIDER_OPTIONS[Provider Options]
            MEMORY_LIMIT[Memory Limit]
        end
        
        subgraph "Runtime Settings"
            LOG_LEVEL[Log Level]
            PROFILING[Profiling Mode]
            DISABLE_FALLBACK[Disable Fallback]
            ENABLE_CACHING[Enable Caching]
        end
    end
    
    BATCH_SIZE --> GRAPH_OPT_LEVEL
    THREAD_COUNT --> EXECUTION_MODE
    INTER_OP_THREADS --> PROVIDER_OPTIONS
    INTRA_OP_THREADS --> MEMORY_LIMIT
    
    GRAPH_OPT_LEVEL --> LOG_LEVEL
    EXECUTION_MODE --> PROFILING
    PROVIDER_OPTIONS --> DISABLE_FALLBACK
    MEMORY_LIMIT --> ENABLE_CACHING
```

### Training Configuration

```mermaid
graph LR
    subgraph "Training Configuration"
        subgraph "Training Settings"
            LEARNING_RATE[Learning Rate]
            BATCH_SIZE_TRAIN[Batch Size]
            GRADIENT_ACCUMULATION_STEPS[Gradient Accumulation Steps]
            MIXED_PRECISION_CONFIG[Mixed Precision]
        end
        
        subgraph "Memory Settings"
            MEMORY_OPTIMIZER[Memory Optimizer]
            RECOMPUTATION_CONFIG[Recomputation Config]
            GRADIENT_CHECKPOINTING[Gradient Checkpointing]
            OFFLOAD_CONFIG[Offload Configuration]
        end
        
        subgraph "Distributed Settings"
            WORLD_SIZE[World Size]
            RANK[Process Rank]
            BACKEND_CONFIG[Communication Backend]
            REDUCTION_CONFIG[Reduction Configuration]
        end
    end
    
    LEARNING_RATE --> MEMORY_OPTIMIZER
    BATCH_SIZE_TRAIN --> RECOMPUTATION_CONFIG
    GRADIENT_ACCUMULATION_STEPS --> GRADIENT_CHECKPOINTING
    MIXED_PRECISION_CONFIG --> OFFLOAD_CONFIG
    
    MEMORY_OPTIMIZER --> WORLD_SIZE
    RECOMPUTATION_CONFIG --> RANK
    GRADIENT_CHECKPOINTING --> BACKEND_CONFIG
    OFFLOAD_CONFIG --> REDUCTION_CONFIG
```

## Use Case Patterns

### Inference Use Cases

```mermaid
graph TD
    subgraph "Inference Use Cases"
        subgraph "Real-time Inference"
            ONLINE_SERVING[Online Serving]
            API_ENDPOINTS[API Endpoints]
            STREAMING[Streaming Inference]
            EDGE_INFERENCE[Edge Inference]
        end
        
        subgraph "Batch Inference"
            BATCH_PROCESSING[Batch Processing]
            ETL_PIPELINES[ETL Pipelines]
            OFFLINE_ANALYSIS[Offline Analysis]
            BULK_SCORING[Bulk Scoring]
        end
        
        subgraph "Specialized Inference"
            EMBEDDED_SYSTEMS[Embedded Systems]
            MOBILE_APPS[Mobile Applications]
            WEB_INFERENCE[Web Inference]
            IOT_DEVICES[IoT Devices]
        end
    end
    
    ONLINE_SERVING --> BATCH_PROCESSING
    API_ENDPOINTS --> ETL_PIPELINES
    STREAMING --> OFFLINE_ANALYSIS
    EDGE_INFERENCE --> BULK_SCORING
    
    BATCH_PROCESSING --> EMBEDDED_SYSTEMS
    ETL_PIPELINES --> MOBILE_APPS
    OFFLINE_ANALYSIS --> WEB_INFERENCE
    BULK_SCORING --> IOT_DEVICES
```

### Training Use Cases

```mermaid
graph TD
    subgraph "Training Use Cases"
        subgraph "Research Training"
            EXPERIMENT_TRAINING[Experimental Training]
            HYPERPARAMETER_TUNING[Hyperparameter Tuning]
            MODEL_DEVELOPMENT[Model Development]
            ABLATION_STUDIES[Ablation Studies]
        end
        
        subgraph "Production Training"
            CONTINUOUS_TRAINING[Continuous Training]
            FEDERATED_LEARNING[Federated Learning]
            TRANSFER_LEARNING[Transfer Learning]
            FINE_TUNING[Fine-tuning]
        end
        
        subgraph "Large-scale Training"
            DISTRIBUTED_TRAINING_USE[Distributed Training]
            MULTI_NODE_TRAINING[Multi-node Training]
            HIGH_THROUGHPUT[High Throughput Training]
            LARGE_MODEL_TRAINING[Large Model Training]
        end
    end
    
    EXPERIMENT_TRAINING --> CONTINUOUS_TRAINING
    HYPERPARAMETER_TUNING --> FEDERATED_LEARNING
    MODEL_DEVELOPMENT --> TRANSFER_LEARNING
    ABLATION_STUDIES --> FINE_TUNING
    
    CONTINUOUS_TRAINING --> DISTRIBUTED_TRAINING_USE
    FEDERATED_LEARNING --> MULTI_NODE_TRAINING
    TRANSFER_LEARNING --> HIGH_THROUGHPUT
    FINE_TUNING --> LARGE_MODEL_TRAINING
```

This comprehensive comparison highlights how ONNX Runtime's architecture adapts to the distinct requirements of training and inference workloads while maintaining a shared foundation for consistency and efficiency.