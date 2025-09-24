# Language Bindings Architecture

This document details the architecture of ONNX Runtime's language bindings, showing how different programming languages interface with the core C++ runtime engine.

## Overview

ONNX Runtime provides native APIs for multiple programming languages through a layered architecture that maintains performance while offering language-specific ergonomics and idioms.

## Language Binding Architecture

```mermaid
graph TB
    subgraph "Application Layer"
        PYTHON_APP[Python Applications]
        CSHARP_APP[C# Applications]
        JAVA_APP[Java Applications]
        JS_APP[JavaScript/Node.js Apps]
        RUST_APP[Rust Applications]
        OBJC_APP[Objective-C/Swift Apps]
        CPP_APP[C++ Applications]
    end
    
    subgraph "Language APIs"
        PYTHON_API[Python API]
        CSHARP_API[C# API]
        JAVA_API[Java API]
        JS_API[JavaScript API]
        RUST_API[Rust API]
        OBJC_API[Objective-C API]
        CPP_API[C++ API]
    end
    
    subgraph "Binding Layer"
        PYTHON_BIND[pybind11]
        CSHARP_BIND[P/Invoke]
        JAVA_BIND[JNI]
        JS_BIND[N-API/WASM]
        RUST_BIND[FFI]
        OBJC_BIND[Objective-C++]
        CPP_DIRECT[Direct Access]
    end
    
    subgraph "C API Layer"
        C_API[C API Interface]
        ABI_STABLE[ABI Stable Interface]
        ERROR_HANDLING[Error Handling]
        MEMORY_MGMT[Memory Management]
    end
    
    subgraph "Core Engine"
        CPP_CORE[C++ Core Runtime]
        SESSION_IMPL[Session Implementation]
        PROVIDER_MGMT[Provider Management]
        TENSOR_IMPL[Tensor Implementation]
    end
    
    PYTHON_APP --> PYTHON_API
    CSHARP_APP --> CSHARP_API
    JAVA_APP --> JAVA_API
    JS_APP --> JS_API
    RUST_APP --> RUST_API
    OBJC_APP --> OBJC_API
    CPP_APP --> CPP_API
    
    PYTHON_API --> PYTHON_BIND
    CSHARP_API --> CSHARP_BIND
    JAVA_API --> JAVA_BIND
    JS_API --> JS_BIND
    RUST_API --> RUST_BIND
    OBJC_API --> OBJC_BIND
    CPP_API --> CPP_DIRECT
    
    PYTHON_BIND --> C_API
    CSHARP_BIND --> C_API
    JAVA_BIND --> C_API
    JS_BIND --> C_API
    RUST_BIND --> C_API
    OBJC_BIND --> C_API
    CPP_DIRECT --> CPP_CORE
    
    C_API --> ABI_STABLE
    ABI_STABLE --> ERROR_HANDLING
    ERROR_HANDLING --> MEMORY_MGMT
    MEMORY_MGMT --> CPP_CORE
    
    CPP_CORE --> SESSION_IMPL
    CPP_CORE --> PROVIDER_MGMT
    CPP_CORE --> TENSOR_IMPL
```

## C API Foundation

The C API provides a stable ABI interface for all language bindings:

```mermaid
classDiagram
    class OrtApi {
        +CreateEnv() OrtStatus
        +CreateSession() OrtStatus
        +CreateSessionOptions() OrtStatus
        +Run() OrtStatus
        +CreateTensorWithDataAsOrtValue() OrtStatus
        +GetTensorData() OrtStatus
        +ReleaseSession() void
        +ReleaseValue() void
    }
    
    class OrtEnv {
        +logging_level: OrtLoggingLevel
        +log_id: string
        +logging_function: OrtLoggingFunction
    }
    
    class OrtSession {
        +session_options: OrtSessionOptions
        +model_data: void*
        +providers: vector~string~
    }
    
    class OrtValue {
        +tensor_type_info: OrtTensorTypeAndShapeInfo
        +data: void*
        +allocator: OrtAllocator
    }
    
    class OrtSessionOptions {
        +execution_providers: vector~string~
        +graph_optimization_level: GraphOptimizationLevel
        +execution_mode: ExecutionMode
        +log_severity_level: int
    }
    
    OrtApi --> OrtEnv : creates
    OrtApi --> OrtSession : creates
    OrtApi --> OrtValue : creates
    OrtSession --> OrtSessionOptions : uses
    OrtValue --> OrtTensorTypeAndShapeInfo : contains
```

## Python Bindings

Python bindings use pybind11 for seamless C++ integration:

```mermaid
graph LR
    subgraph "Python Binding Architecture"
        subgraph "Python Layer"
            PY_SESSION[InferenceSession]
            PY_TENSOR[OrtValue/Tensor]
            PY_PROVIDERS[Execution Providers]
            PY_OPTIONS[Session Options]
        end
        
        subgraph "pybind11 Layer"
            PYBIND_WRAPPER[pybind11 Wrappers]
            TYPE_CONVERSION[Type Conversion]
            EXCEPTION_TRANS[Exception Translation]
            MEMORY_VIEW[Memory Views]
        end
        
        subgraph "C++ Integration"
            CPP_SESSION[C++ Session]
            CPP_TENSOR[C++ Tensor]
            CPP_ALLOCATOR[C++ Allocator]
            CPP_PROVIDER[C++ Provider]
        end
        
        subgraph "NumPy Integration"
            NUMPY_ARRAY[NumPy Arrays]
            BUFFER_PROTOCOL[Buffer Protocol]
            ZERO_COPY[Zero-Copy Views]
            DTYPE_MAPPING[Data Type Mapping]
        end
    end
    
    PY_SESSION --> PYBIND_WRAPPER
    PY_TENSOR --> TYPE_CONVERSION
    PY_PROVIDERS --> EXCEPTION_TRANS
    PY_OPTIONS --> MEMORY_VIEW
    
    PYBIND_WRAPPER --> CPP_SESSION
    TYPE_CONVERSION --> CPP_TENSOR
    EXCEPTION_TRANS --> CPP_ALLOCATOR
    MEMORY_VIEW --> CPP_PROVIDER
    
    PY_TENSOR --> NUMPY_ARRAY
    NUMPY_ARRAY --> BUFFER_PROTOCOL
    BUFFER_PROTOCOL --> ZERO_COPY
    ZERO_COPY --> DTYPE_MAPPING
```

### Python API Flow

```mermaid
sequenceDiagram
    participant App as Python App
    participant API as Python API
    participant Bind as pybind11
    participant Core as C++ Core
    participant NumPy as NumPy
    
    App->>API: import onnxruntime
    API->>Bind: Load C++ Module
    Bind->>Core: Initialize Runtime
    
    App->>API: InferenceSession(model)
    API->>Bind: Create Session
    Bind->>Core: CreateSession()
    Core-->>Bind: Session Handle
    Bind-->>API: Wrapped Session
    
    App->>NumPy: Create Input Array
    NumPy-->>App: ndarray
    App->>API: session.run(inputs)
    API->>Bind: Convert Inputs
    Bind->>Core: OrtValue from NumPy
    Core->>Core: Execute Model
    Core-->>Bind: Output OrtValues
    Bind->>NumPy: Convert to ndarray
    NumPy-->>API: Output Arrays
    API-->>App: Results
```

## C# Bindings

C# bindings use P/Invoke for .NET interoperability:

```mermaid
graph TB
    subgraph "C# Binding Architecture"
        subgraph ".NET Layer"
            CS_SESSION[InferenceSession]
            CS_TENSOR[NamedOnnxValue]
            CS_OPTIONS[SessionOptions]
            CS_METADATA[ModelMetadata]
        end
        
        subgraph "P/Invoke Layer"
            PINVOKE[P/Invoke Declarations]
            MARSHAL[Marshaling]
            HANDLE_MGMT[Handle Management]
            DISPOSE_PATTERN[Dispose Pattern]
        end
        
        subgraph "Memory Management"
            SAFE_HANDLE[SafeHandle]
            FINALIZER[Finalizers]
            MEMORY_PIN[Memory Pinning]
            GC_HANDLE[GC Handles]
        end
        
        subgraph "Data Integration"
            TENSOR_WRAP[Tensor Wrappers]
            ARRAY_CONVERSION[Array Conversion]
            SPAN_SUPPORT[Span<T> Support]
            UNSAFE_CODE[Unsafe Code Blocks]
        end
    end
    
    CS_SESSION --> PINVOKE
    CS_TENSOR --> MARSHAL
    CS_OPTIONS --> HANDLE_MGMT
    CS_METADATA --> DISPOSE_PATTERN
    
    PINVOKE --> SAFE_HANDLE
    MARSHAL --> FINALIZER
    HANDLE_MGMT --> MEMORY_PIN
    DISPOSE_PATTERN --> GC_HANDLE
    
    CS_TENSOR --> TENSOR_WRAP
    TENSOR_WRAP --> ARRAY_CONVERSION
    ARRAY_CONVERSION --> SPAN_SUPPORT
    SPAN_SUPPORT --> UNSAFE_CODE
```

### C# Memory Management

```mermaid
flowchart TD
    CREATE[Create OrtValue]
    PIN_MEMORY[Pin Managed Memory]
    CALL_NATIVE[Call Native Function]
    UNPIN_MEMORY[Unpin Memory]
    WRAP_HANDLE[Wrap in SafeHandle]
    USE_OBJECT[Use Object]
    DISPOSE{Dispose Called?}
    FINALIZER[Finalizer Runs]
    RELEASE_NATIVE[Release Native Resource]
    COMPLETE[Cleanup Complete]
    
    CREATE --> PIN_MEMORY
    PIN_MEMORY --> CALL_NATIVE
    CALL_NATIVE --> UNPIN_MEMORY
    UNPIN_MEMORY --> WRAP_HANDLE
    WRAP_HANDLE --> USE_OBJECT
    
    USE_OBJECT --> DISPOSE
    DISPOSE -->|Yes| RELEASE_NATIVE
    DISPOSE -->|No| FINALIZER
    FINALIZER --> RELEASE_NATIVE
    RELEASE_NATIVE --> COMPLETE
```

## Java Bindings

Java bindings use JNI for native integration:

```mermaid
graph LR
    subgraph "Java Binding Architecture"
        subgraph "Java Layer"
            JAVA_SESSION[OrtSession]
            JAVA_TENSOR[OnnxTensor]
            JAVA_ENV[OrtEnvironment]
            JAVA_OPTIONS[SessionOptions]
        end
        
        subgraph "JNI Layer"
            JNI_WRAPPER[JNI Wrappers]
            METHOD_MAPPING[Native Method Mapping]
            OBJECT_LIFECYCLE[Object Lifecycle]
            EXCEPTION_MAPPING[Exception Mapping]
        end
        
        subgraph "Memory Bridges"
            DIRECT_BUFFER[Direct ByteBuffers]
            ARRAY_COPY[Array Copying]
            NATIVE_HEAP[Native Heap Access]
            GC_SAFE[GC-Safe References]
        end
        
        subgraph "Threading"
            THREAD_ATTACH[Thread Attachment]
            LOCAL_REFS[Local References]
            GLOBAL_REFS[Global References]
            CRITICAL_SECTION[Critical Sections]
        end
    end
    
    JAVA_SESSION --> JNI_WRAPPER
    JAVA_TENSOR --> METHOD_MAPPING
    JAVA_ENV --> OBJECT_LIFECYCLE
    JAVA_OPTIONS --> EXCEPTION_MAPPING
    
    JNI_WRAPPER --> DIRECT_BUFFER
    METHOD_MAPPING --> ARRAY_COPY
    OBJECT_LIFECYCLE --> NATIVE_HEAP
    EXCEPTION_MAPPING --> GC_SAFE
    
    DIRECT_BUFFER --> THREAD_ATTACH
    ARRAY_COPY --> LOCAL_REFS
    NATIVE_HEAP --> GLOBAL_REFS
    GC_SAFE --> CRITICAL_SECTION
```

### JNI Object Lifecycle

```mermaid
stateDiagram-v2
    [*] --> Created: Java Object Created
    Created --> NativeAllocated: JNI Allocates Native
    NativeAllocated --> InUse: Object In Use
    InUse --> InUse: Method Calls
    InUse --> Disposing: close() Called
    InUse --> Finalizing: Finalizer Runs
    Disposing --> NativeReleased: Release Native Resources
    Finalizing --> NativeReleased: Release Native Resources
    NativeReleased --> [*]: Object Destroyed
```

## JavaScript/WebAssembly Bindings

JavaScript bindings support both Node.js and browser environments:

```mermaid
graph TB
    subgraph "JavaScript Binding Architecture"
        subgraph "JavaScript Layer"
            JS_SESSION[InferenceSession]
            JS_TENSOR[Tensor]
            JS_BACKEND[Backend Selection]
            JS_OPTIONS[Session Options]
        end
        
        subgraph "Runtime Environments"
            NODE_ENV[Node.js Environment]
            BROWSER_ENV[Browser Environment]
            WORKER_ENV[Web Worker Environment]
            REACT_NATIVE[React Native]
        end
        
        subgraph "Execution Backends"
            WASM_BACKEND[WebAssembly Backend]
            WEBGL_BACKEND[WebGL Backend]
            WEBGPU_BACKEND[WebGPU Backend]
            WEBNN_BACKEND[WebNN Backend]
        end
        
        subgraph "Data Handling"
            TYPED_ARRAYS[Typed Arrays]
            ARRAY_BUFFER[Array Buffers]
            SHARED_BUFFER[Shared Array Buffers]
            MEMORY_MAPPING[Memory Mapping]
        end
    end
    
    JS_SESSION --> NODE_ENV
    JS_SESSION --> BROWSER_ENV
    JS_TENSOR --> WORKER_ENV
    JS_BACKEND --> REACT_NATIVE
    
    NODE_ENV --> WASM_BACKEND
    BROWSER_ENV --> WEBGL_BACKEND
    WORKER_ENV --> WEBGPU_BACKEND
    REACT_NATIVE --> WEBNN_BACKEND
    
    JS_TENSOR --> TYPED_ARRAYS
    TYPED_ARRAYS --> ARRAY_BUFFER
    ARRAY_BUFFER --> SHARED_BUFFER
    SHARED_BUFFER --> MEMORY_MAPPING
```

### WebAssembly Integration

```mermaid
sequenceDiagram
    participant JS as JavaScript
    participant WASM as WebAssembly
    participant Memory as WASM Memory
    participant Runtime as ORT Runtime
    
    JS->>WASM: Load WASM Module
    WASM-->>JS: Module Ready
    
    JS->>Memory: Allocate Input Buffer
    Memory-->>JS: Buffer Pointer
    JS->>Memory: Copy Input Data
    
    JS->>WASM: Call ort_run(ptr, size)
    WASM->>Runtime: Execute Model
    Runtime->>Runtime: Process Tensors
    Runtime-->>WASM: Return Results
    WASM-->>JS: Result Pointer
    
    JS->>Memory: Read Output Data
    Memory-->>JS: Output Arrays
    JS->>Memory: Free Buffers
```

## Rust Bindings

Rust bindings provide safe FFI access to the C API:

```mermaid
graph LR
    subgraph "Rust Binding Architecture"
        subgraph "Safe Rust Layer"
            RUST_SESSION[Session]
            RUST_TENSOR[Value]
            RUST_ENV[Environment]
            RUST_ERROR[OrtError]
        end
        
        subgraph "FFI Layer"
            FFI_BINDINGS[FFI Bindings]
            UNSAFE_BLOCKS[Unsafe Blocks]
            POINTER_MGMT[Pointer Management]
            LIFETIME_MGMT[Lifetime Management]
        end
        
        subgraph "Memory Safety"
            RAII_PATTERN[RAII Pattern]
            DROP_TRAIT[Drop Trait]
            BORROW_CHECKER[Borrow Checker]
            ZERO_COST[Zero-Cost Abstractions]
        end
        
        subgraph "Type Safety"
            TYPE_WRAPPER[Type Wrappers]
            PHANTOM_DATA[PhantomData]
            MARKER_TRAITS[Marker Traits]
            COMPILE_CHECKS[Compile-Time Checks]
        end
    end
    
    RUST_SESSION --> FFI_BINDINGS
    RUST_TENSOR --> UNSAFE_BLOCKS
    RUST_ENV --> POINTER_MGMT
    RUST_ERROR --> LIFETIME_MGMT
    
    FFI_BINDINGS --> RAII_PATTERN
    UNSAFE_BLOCKS --> DROP_TRAIT
    POINTER_MGMT --> BORROW_CHECKER
    LIFETIME_MGMT --> ZERO_COST
    
    RAII_PATTERN --> TYPE_WRAPPER
    DROP_TRAIT --> PHANTOM_DATA
    BORROW_CHECKER --> MARKER_TRAITS
    ZERO_COST --> COMPILE_CHECKS
```

## Objective-C/Swift Bindings

Native iOS and macOS integration:

```mermaid
graph TB
    subgraph "iOS/macOS Binding Architecture"
        subgraph "Swift Layer"
            SWIFT_SESSION[ORTSession]
            SWIFT_VALUE[ORTValue]
            SWIFT_ENV[ORTEnv]
            SWIFT_ERROR[ORTError]
        end
        
        subgraph "Objective-C Layer"
            OBJC_WRAPPER[Objective-C Wrappers]
            OBJC_CATEGORIES[Categories]
            OBJC_PROTOCOLS[Protocols]
            OBJC_BLOCKS[Blocks]
        end
        
        subgraph "Foundation Integration"
            NS_ARRAY[NSArray]
            NS_DATA[NSData]
            NS_ERROR[NSError]
            CORE_ML[Core ML Integration]
        end
        
        subgraph "Memory Management"
            ARC[Automatic Reference Counting]
            AUTORELEASE[Autorelease Pools]
            WEAK_REFS[Weak References]
            BRIDGE_CAST[Bridge Casting]
        end
    end
    
    SWIFT_SESSION --> OBJC_WRAPPER
    SWIFT_VALUE --> OBJC_CATEGORIES
    SWIFT_ENV --> OBJC_PROTOCOLS
    SWIFT_ERROR --> OBJC_BLOCKS
    
    OBJC_WRAPPER --> NS_ARRAY
    OBJC_CATEGORIES --> NS_DATA
    OBJC_PROTOCOLS --> NS_ERROR
    OBJC_BLOCKS --> CORE_ML
    
    NS_ARRAY --> ARC
    NS_DATA --> AUTORELEASE
    NS_ERROR --> WEAK_REFS
    CORE_ML --> BRIDGE_CAST
```

## Cross-Language Data Types

Mapping between language-specific types and ONNX Runtime types:

```mermaid
graph LR
    subgraph "Data Type Mapping"
        subgraph "ONNX Runtime Types"
            ORT_FLOAT[ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT]
            ORT_INT64[ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64]
            ORT_STRING[ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING]
            ORT_BOOL[ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL]
        end
        
        subgraph "Language Types"
            PY_FLOAT32[numpy.float32]
            CS_FLOAT[System.Single]
            JAVA_FLOAT[float]
            JS_FLOAT32[Float32Array]
            RUST_F32[f32]
            OBJC_FLOAT[float]
        end
        
        subgraph "Memory Layout"
            ROW_MAJOR[Row Major]
            COLUMN_MAJOR[Column Major]
            STRIDED[Strided Arrays]
            CONTIGUOUS[Contiguous Memory]
        end
    end
    
    ORT_FLOAT --> PY_FLOAT32
    ORT_FLOAT --> CS_FLOAT
    ORT_FLOAT --> JAVA_FLOAT
    ORT_FLOAT --> JS_FLOAT32
    ORT_FLOAT --> RUST_F32
    ORT_FLOAT --> OBJC_FLOAT
    
    PY_FLOAT32 --> ROW_MAJOR
    CS_FLOAT --> COLUMN_MAJOR
    JAVA_FLOAT --> STRIDED
    JS_FLOAT32 --> CONTIGUOUS
```

## Error Handling Across Languages

Consistent error handling patterns across all language bindings:

```mermaid
graph TD
    subgraph "Error Handling Strategy"
        subgraph "C API Errors"
            ORT_STATUS[OrtStatus]
            ERROR_CODE[Error Code]
            ERROR_MSG[Error Message]
        end
        
        subgraph "Language-Specific Errors"
            PY_EXCEPTION[Python Exception]
            CS_EXCEPTION[C# Exception]
            JAVA_EXCEPTION[Java Exception]
            JS_ERROR[JavaScript Error]
            RUST_RESULT[Rust Result<T, E>]
            OBJC_NSERROR[NSError]
        end
        
        subgraph "Error Categories"
            INVALID_ARG[Invalid Argument]
            OUT_OF_MEMORY[Out of Memory]
            MODEL_ERROR[Model Error]
            RUNTIME_ERROR[Runtime Error]
            PROVIDER_ERROR[Provider Error]
        end
    end
    
    ORT_STATUS --> ERROR_CODE
    ERROR_CODE --> ERROR_MSG
    
    ERROR_MSG --> PY_EXCEPTION
    ERROR_MSG --> CS_EXCEPTION
    ERROR_MSG --> JAVA_EXCEPTION
    ERROR_MSG --> JS_ERROR
    ERROR_MSG --> RUST_RESULT
    ERROR_MSG --> OBJC_NSERROR
    
    ERROR_CODE --> INVALID_ARG
    ERROR_CODE --> OUT_OF_MEMORY
    ERROR_CODE --> MODEL_ERROR
    ERROR_CODE --> RUNTIME_ERROR
    ERROR_CODE --> PROVIDER_ERROR
```

## Performance Considerations

Key performance aspects of language bindings:

```mermaid
mindmap
    root((Performance))
        Zero-Copy Operations
            NumPy Buffer Protocol
            .NET Span<T>
            Java Direct Buffers
            Rust Slices
            JavaScript Typed Arrays
        Memory Management
            Automatic Memory Management
            Manual Resource Cleanup
            Reference Counting
            Garbage Collection Integration
        Call Overhead
            Direct C++ Calls
            FFI Overhead
            Marshaling Costs
            Type Conversion
        Threading
            GIL Considerations
            Thread Safety
            Async Operations
            Parallel Execution
```

## Language-Specific Optimizations

Each language binding implements specific optimizations:

```mermaid
graph TB
    subgraph "Language Optimizations"
        subgraph "Python Optimizations"
            PY_NOGIL[Release GIL]
            PY_BUFFER[Buffer Protocol]
            PY_VECTORIZE[NumPy Vectorization]
            PY_THREADING[Threading Support]
        end
        
        subgraph "C# Optimizations"
            CS_UNSAFE[Unsafe Code]
            CS_SPAN[Span<T> Usage]
            CS_PINNING[Memory Pinning]
            CS_ASYNC[Async/Await]
        end
        
        subgraph "Java Optimizations"
            JAVA_DIRECT[Direct Buffers]
            JAVA_CRITICAL[Critical Arrays]
            JAVA_WEAK[Weak References]
            JAVA_CONCURRENT[Concurrent Collections]
        end
        
        subgraph "JavaScript Optimizations"
            JS_WORKERS[Web Workers]
            JS_WASM[WebAssembly]
            JS_SIMD[SIMD.js]
            JS_OFFSCREEN[OffscreenCanvas]
        end
    end
    
    PY_NOGIL --> PY_BUFFER
    PY_BUFFER --> PY_VECTORIZE
    PY_VECTORIZE --> PY_THREADING
    
    CS_UNSAFE --> CS_SPAN
    CS_SPAN --> CS_PINNING
    CS_PINNING --> CS_ASYNC
    
    JAVA_DIRECT --> JAVA_CRITICAL
    JAVA_CRITICAL --> JAVA_WEAK
    JAVA_WEAK --> JAVA_CONCURRENT
    
    JS_WORKERS --> JS_WASM
    JS_WASM --> JS_SIMD
    JS_SIMD --> JS_OFFSCREEN
```

This comprehensive language binding architecture enables ONNX Runtime to provide native, high-performance APIs across multiple programming languages while maintaining consistency and safety.