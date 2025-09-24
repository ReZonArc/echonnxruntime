# Build System Architecture

This document details the comprehensive build system architecture of ONNX Runtime, covering cross-platform builds, dependency management, packaging, and CI/CD integration.

## Overview

ONNX Runtime uses a sophisticated build system based on CMake that supports multiple platforms, architectures, execution providers, and language bindings while maintaining consistency and flexibility.

## Build System Architecture

```mermaid
graph TB
    subgraph "Build System Overview"
        subgraph "Build Tools"
            CMAKE[CMake Build System]
            PYTHON_BUILD[Python Build Scripts]
            DOCKER_BUILD[Docker Build Environments]
            CI_SCRIPTS[CI/CD Scripts]
        end
        
        subgraph "Configuration Management"
            CMAKE_PRESETS[CMake Presets]
            BUILD_CONFIGS[Build Configurations]
            FEATURE_FLAGS[Feature Flags]
            PLATFORM_CONFIG[Platform Configuration]
        end
        
        subgraph "Dependency Management"
            VCPKG[vcpkg Package Manager]
            CONAN[Conan Package Manager]
            SUBMODULES[Git Submodules]
            EXTERNAL_DEPS[External Dependencies]
        end
        
        subgraph "Output Artifacts"
            SHARED_LIBS[Shared Libraries]
            STATIC_LIBS[Static Libraries]
            EXECUTABLES[Executables]
            PACKAGES[Language Packages]
        end
    end
    
    CMAKE --> CMAKE_PRESETS
    PYTHON_BUILD --> BUILD_CONFIGS
    DOCKER_BUILD --> FEATURE_FLAGS
    CI_SCRIPTS --> PLATFORM_CONFIG
    
    CMAKE_PRESETS --> VCPKG
    BUILD_CONFIGS --> CONAN
    FEATURE_FLAGS --> SUBMODULES
    PLATFORM_CONFIG --> EXTERNAL_DEPS
    
    VCPKG --> SHARED_LIBS
    CONAN --> STATIC_LIBS
    SUBMODULES --> EXECUTABLES
    EXTERNAL_DEPS --> PACKAGES
```

## CMake Build Structure

The hierarchical CMake structure organizes the complex build system:

```mermaid
graph LR
    subgraph "CMake Build Structure"
        subgraph "Root Configuration"
            ROOT_CMAKE[CMakeLists.txt]
            CMAKE_PRESETS_JSON[CMakePresets.json]
            CMAKE_SETTINGS[CMakeSettings.json]
        end
        
        subgraph "Core Components"
            CORE_CMAKE[onnxruntime_core.cmake]
            FRAMEWORK_CMAKE[onnxruntime_framework.cmake]
            SESSION_CMAKE[onnxruntime_session.cmake]
            GRAPH_CMAKE[onnxruntime_graph.cmake]
        end
        
        subgraph "Provider Modules"
            CPU_CMAKE[onnxruntime_providers_cpu.cmake]
            CUDA_CMAKE[onnxruntime_providers_cuda.cmake]
            TENSORRT_CMAKE[onnxruntime_providers_tensorrt.cmake]
            COREML_CMAKE[onnxruntime_providers_coreml.cmake]
        end
        
        subgraph "Language Bindings"
            PYTHON_CMAKE[onnxruntime_python.cmake]
            CSHARP_CMAKE[onnxruntime_csharp.cmake]
            JAVA_CMAKE[onnxruntime_java.cmake]
            NODEJS_CMAKE[onnxruntime_nodejs.cmake]
        end
    end
    
    ROOT_CMAKE --> CORE_CMAKE
    ROOT_CMAKE --> FRAMEWORK_CMAKE
    ROOT_CMAKE --> SESSION_CMAKE
    ROOT_CMAKE --> GRAPH_CMAKE
    
    CORE_CMAKE --> CPU_CMAKE
    CORE_CMAKE --> CUDA_CMAKE
    CORE_CMAKE --> TENSORRT_CMAKE
    CORE_CMAKE --> COREML_CMAKE
    
    FRAMEWORK_CMAKE --> PYTHON_CMAKE
    FRAMEWORK_CMAKE --> CSHARP_CMAKE
    FRAMEWORK_CMAKE --> JAVA_CMAKE
    FRAMEWORK_CMAKE --> NODEJS_CMAKE
```

## Platform Support Matrix

Cross-platform build support across different operating systems and architectures:

```mermaid
graph TD
    subgraph "Platform Support Matrix"
        subgraph "Operating Systems"
            WINDOWS[Windows]
            LINUX[Linux]
            MACOS[macOS]
            ANDROID[Android]
            IOS[iOS]
            WEB[Web/WASM]
        end
        
        subgraph "Architectures"
            X86_64[x86_64]
            ARM64[ARM64]
            ARM32[ARM32]
            WASM32[WebAssembly]
        end
        
        subgraph "Compilers"
            MSVC[Microsoft Visual C++]
            GCC[GNU Compiler Collection]
            CLANG[Clang/LLVM]
            EMSCRIPTEN[Emscripten]
        end
        
        subgraph "Build Types"
            DEBUG[Debug]
            RELEASE[Release]
            RELWITHDEBINFO[RelWithDebInfo]
            MINSIZEREL[MinSizeRel]
        end
    end
    
    WINDOWS --> X86_64
    WINDOWS --> ARM64
    LINUX --> X86_64
    LINUX --> ARM64
    LINUX --> ARM32
    MACOS --> X86_64
    MACOS --> ARM64
    ANDROID --> ARM64
    ANDROID --> ARM32
    IOS --> ARM64
    WEB --> WASM32
    
    WINDOWS --> MSVC
    LINUX --> GCC
    LINUX --> CLANG
    MACOS --> CLANG
    ANDROID --> CLANG
    IOS --> CLANG
    WEB --> EMSCRIPTEN
    
    X86_64 --> DEBUG
    X86_64 --> RELEASE
    ARM64 --> RELWITHDEBINFO
    ARM32 --> MINSIZEREL
```

## Dependency Management

Comprehensive dependency management across different package managers:

```mermaid
graph TB
    subgraph "Dependency Management System"
        subgraph "Package Managers"
            VCPKG_MGR[vcpkg]
            CONAN_MGR[Conan]
            CMAKE_FIND[CMake Find Modules]
            SYSTEM_PKGS[System Packages]
        end
        
        subgraph "Core Dependencies"
            ONNX_LIB[ONNX Library]
            PROTOBUF_LIB[Protocol Buffers]
            EIGEN_LIB[Eigen3]
            FLATBUF_LIB[FlatBuffers]
            ABSEIL_LIB[Abseil-cpp]
            JSON_LIB[nlohmann/json]
        end
        
        subgraph "Provider Dependencies"
            CUDA_LIB[CUDA Toolkit]
            CUDNN_LIB[cuDNN]
            TENSORRT_LIB[TensorRT]
            DNNL_LIB[oneDNN]
            OPENVINO_LIB[OpenVINO]
            COREML_LIB[CoreML]
        end
        
        subgraph "Build Tools"
            PYTHON_DEPS[Python Dependencies]
            NODE_DEPS[Node.js Dependencies]
            DOTNET_DEPS[.NET Dependencies]
            JAVA_DEPS[Java Dependencies]
        end
    end
    
    VCPKG_MGR --> ONNX_LIB
    VCPKG_MGR --> PROTOBUF_LIB
    CONAN_MGR --> EIGEN_LIB
    CMAKE_FIND --> FLATBUF_LIB
    SYSTEM_PKGS --> ABSEIL_LIB
    
    VCPKG_MGR --> CUDA_LIB
    CMAKE_FIND --> CUDNN_LIB
    SYSTEM_PKGS --> TENSORRT_LIB
    VCPKG_MGR --> DNNL_LIB
    CMAKE_FIND --> OPENVINO_LIB
    SYSTEM_PKGS --> COREML_LIB
    
    ONNX_LIB --> PYTHON_DEPS
    PROTOBUF_LIB --> NODE_DEPS
    EIGEN_LIB --> DOTNET_DEPS
    FLATBUF_LIB --> JAVA_DEPS
```

## Build Configuration System

Flexible configuration system for different build scenarios:

```mermaid
graph LR
    subgraph "Build Configuration System"
        subgraph "Configuration Types"
            MINIMAL_BUILD[Minimal Build]
            STANDARD_BUILD[Standard Build]
            FULL_BUILD[Full Build]
            CUSTOM_BUILD[Custom Build]
        end
        
        subgraph "Feature Toggles"
            ENABLE_CUDA[--use_cuda]
            ENABLE_TENSORRT[--use_tensorrt]
            ENABLE_DNNL[--use_dnnl]
            ENABLE_TRAINING[--enable_training]
            ENABLE_PYTHON[--build_python]
            ENABLE_TESTING[--build_tests]
        end
        
        subgraph "Optimization Options"
            GRAPH_OPT[--enable_graph_opt]
            LTO[--enable_lto]
            PARALLEL_BUILD[--parallel]
            CCACHE[--use_ccache]
        end
        
        subgraph "Platform Options"
            ANDROID_BUILD[--android]
            IOS_BUILD[--ios]
            WASM_BUILD[--build_wasm]
            ARM_BUILD[--arm64]
        end
    end
    
    MINIMAL_BUILD --> ENABLE_CUDA
    STANDARD_BUILD --> ENABLE_TENSORRT
    FULL_BUILD --> ENABLE_DNNL
    CUSTOM_BUILD --> ENABLE_TRAINING
    
    ENABLE_CUDA --> GRAPH_OPT
    ENABLE_TENSORRT --> LTO
    ENABLE_DNNL --> PARALLEL_BUILD
    ENABLE_TRAINING --> CCACHE
    
    GRAPH_OPT --> ANDROID_BUILD
    LTO --> IOS_BUILD
    PARALLEL_BUILD --> WASM_BUILD
    CCACHE --> ARM_BUILD
```

## Language Binding Build Process

Specific build processes for different language bindings:

```mermaid
sequenceDiagram
    participant Build as Build Script
    participant CMake as CMake
    participant Core as Core Libraries
    participant Python as Python Build
    participant CSharp as C# Build
    participant Java as Java Build
    participant Node as Node.js Build
    
    Note over Build, Node: Core Build Phase
    Build->>CMake: Configure CMake
    CMake->>Core: Build Core Libraries
    Core-->>CMake: Libraries Ready
    CMake-->>Build: Core Build Complete
    
    Note over Build, Node: Language Binding Phase
    par Python Binding
        Build->>Python: Build Python Package
        Python->>Core: Link with Core
        Core-->>Python: Python Package Ready
        Python-->>Build: Wheel Generated
    and C# Binding
        Build->>CSharp: Build NuGet Package
        CSharp->>Core: Link with Core
        Core-->>CSharp: C# Package Ready
        CSharp-->>Build: NuGet Generated
    and Java Binding
        Build->>Java: Build JAR Package
        Java->>Core: Link with Core
        Core-->>Java: Java Package Ready
        Java-->>Build: JAR Generated
    and Node.js Binding
        Build->>Node: Build Node Package
        Node->>Core: Link with Core
        Core-->>Node: Node Package Ready
        Node-->>Build: NPM Package Generated
    end
```

## Docker Build Environments

Containerized build environments for consistent builds:

```mermaid
graph TB
    subgraph "Docker Build Environments"
        subgraph "Base Images"
            UBUNTU_BASE[Ubuntu Base]
            CENTOS_BASE[CentOS Base]
            WINDOWS_BASE[Windows Base]
            ANDROID_NDK[Android NDK]
        end
        
        subgraph "Development Images"
            DEV_UBUNTU[Ubuntu Dev Environment]
            DEV_CENTOS[CentOS Dev Environment]
            DEV_CUDA[CUDA Dev Environment]
            DEV_TENSORRT[TensorRT Dev Environment]
        end
        
        subgraph "CI/CD Images"
            CI_LINUX[Linux CI Image]
            CI_WINDOWS[Windows CI Image]
            CI_MACOS[macOS CI Image]
            CI_ANDROID[Android CI Image]
        end
        
        subgraph "Specialized Images"
            MINIMAL_IMAGE[Minimal Runtime]
            GPU_IMAGE[GPU Runtime]
            WEB_BUILD[Web Build Environment]
            CROSS_COMPILE[Cross-Compile Environment]
        end
    end
    
    UBUNTU_BASE --> DEV_UBUNTU
    CENTOS_BASE --> DEV_CENTOS
    UBUNTU_BASE --> DEV_CUDA
    DEV_CUDA --> DEV_TENSORRT
    
    DEV_UBUNTU --> CI_LINUX
    WINDOWS_BASE --> CI_WINDOWS
    DEV_UBUNTU --> CI_ANDROID
    
    CI_LINUX --> MINIMAL_IMAGE
    DEV_CUDA --> GPU_IMAGE
    UBUNTU_BASE --> WEB_BUILD
    DEV_UBUNTU --> CROSS_COMPILE
```

## CI/CD Pipeline Architecture

Comprehensive CI/CD pipeline for automated builds and testing:

```mermaid
flowchart TD
    TRIGGER[Code Push/PR]
    
    subgraph "Pre-Build"
        LINT[Code Linting]
        FORMAT_CHECK[Format Check]
        LICENSE_CHECK[License Check]
        DEPENDENCY_CHECK[Dependency Check]
    end
    
    subgraph "Build Matrix"
        LINUX_BUILD[Linux Builds]
        WINDOWS_BUILD[Windows Builds]
        MACOS_BUILD[macOS Builds]
        ANDROID_BUILD[Android Builds]
        IOS_BUILD[iOS Builds]
        WEB_BUILD[Web Builds]
    end
    
    subgraph "Testing"
        UNIT_TESTS[Unit Tests]
        INTEGRATION_TESTS[Integration Tests]
        PERFORMANCE_TESTS[Performance Tests]
        MODEL_TESTS[Model Tests]
    end
    
    subgraph "Packaging"
        PYTHON_WHEEL[Python Wheels]
        NUGET_PACKAGE[NuGet Packages]
        NPM_PACKAGE[NPM Packages]
        JAR_PACKAGE[JAR Packages]
        DOCKER_IMAGES[Docker Images]
    end
    
    subgraph "Release"
        ARTIFACT_UPLOAD[Upload Artifacts]
        DOCKER_PUSH[Push Docker Images]
        PACKAGE_PUBLISH[Publish Packages]
        RELEASE_NOTES[Generate Release Notes]
    end
    
    TRIGGER --> LINT
    LINT --> FORMAT_CHECK
    FORMAT_CHECK --> LICENSE_CHECK
    LICENSE_CHECK --> DEPENDENCY_CHECK
    
    DEPENDENCY_CHECK --> LINUX_BUILD
    DEPENDENCY_CHECK --> WINDOWS_BUILD
    DEPENDENCY_CHECK --> MACOS_BUILD
    DEPENDENCY_CHECK --> ANDROID_BUILD
    DEPENDENCY_CHECK --> IOS_BUILD
    DEPENDENCY_CHECK --> WEB_BUILD
    
    LINUX_BUILD --> UNIT_TESTS
    WINDOWS_BUILD --> INTEGRATION_TESTS
    MACOS_BUILD --> PERFORMANCE_TESTS
    ANDROID_BUILD --> MODEL_TESTS
    
    UNIT_TESTS --> PYTHON_WHEEL
    INTEGRATION_TESTS --> NUGET_PACKAGE
    PERFORMANCE_TESTS --> NPM_PACKAGE
    MODEL_TESTS --> JAR_PACKAGE
    
    PYTHON_WHEEL --> ARTIFACT_UPLOAD
    NUGET_PACKAGE --> DOCKER_PUSH
    NPM_PACKAGE --> PACKAGE_PUBLISH
    JAR_PACKAGE --> RELEASE_NOTES
```

## Build Optimization Strategies

Various strategies to optimize build performance:

```mermaid
graph LR
    subgraph "Build Optimization Strategies"
        subgraph "Parallelization"
            PARALLEL_COMPILE[Parallel Compilation]
            DISTRIBUTED_BUILD[Distributed Builds]
            CONCURRENT_JOBS[Concurrent Jobs]
        end
        
        subgraph "Caching"
            CCACHE[Compiler Cache]
            BUILD_CACHE[Build Cache]
            DEPENDENCY_CACHE[Dependency Cache]
            DOCKER_CACHE[Docker Layer Cache]
        end
        
        subgraph "Incremental Builds"
            INCREMENTAL_COMPILE[Incremental Compilation]
            PARTIAL_REBUILD[Partial Rebuilds]
            CHANGE_DETECTION[Change Detection]
        end
        
        subgraph "Resource Optimization"
            MEMORY_LIMIT[Memory Limits]
            CPU_OPTIMIZATION[CPU Optimization]
            DISK_OPTIMIZATION[Disk I/O Optimization]
            NETWORK_OPTIMIZATION[Network Optimization]
        end
    end
    
    PARALLEL_COMPILE --> CCACHE
    DISTRIBUTED_BUILD --> BUILD_CACHE
    CONCURRENT_JOBS --> DEPENDENCY_CACHE
    
    CCACHE --> INCREMENTAL_COMPILE
    BUILD_CACHE --> PARTIAL_REBUILD
    DEPENDENCY_CACHE --> CHANGE_DETECTION
    
    INCREMENTAL_COMPILE --> MEMORY_LIMIT
    PARTIAL_REBUILD --> CPU_OPTIMIZATION
    CHANGE_DETECTION --> DISK_OPTIMIZATION
    
    MEMORY_LIMIT --> NETWORK_OPTIMIZATION
```

## Package Generation Process

Process for generating packages for different platforms and languages:

```mermaid
graph TD
    subgraph "Package Generation Process"
        subgraph "Python Packaging"
            SETUP_PY[setup.py]
            WHEEL_BUILD[Wheel Building]
            PYPI_UPLOAD[PyPI Upload]
        end
        
        subgraph "C# Packaging"
            NUSPEC[NuGet Specification]
            NUGET_PACK[NuGet Packaging]
            NUGET_UPLOAD[NuGet Gallery Upload]
        end
        
        subgraph "Java Packaging"
            POM_XML[Maven POM]
            JAR_BUILD[JAR Building]
            MAVEN_DEPLOY[Maven Deploy]
        end
        
        subgraph "Node.js Packaging"
            PACKAGE_JSON[package.json]
            NPM_PACK[NPM Packaging]
            NPM_PUBLISH[NPM Publish]
        end
        
        subgraph "Native Packaging"
            CONAN_PACKAGE[Conan Package]
            VCPKG_PORT[vcpkg Port]
            SYSTEM_PACKAGES[System Packages]
        end
    end
    
    SETUP_PY --> WHEEL_BUILD
    WHEEL_BUILD --> PYPI_UPLOAD
    
    NUSPEC --> NUGET_PACK
    NUGET_PACK --> NUGET_UPLOAD
    
    POM_XML --> JAR_BUILD
    JAR_BUILD --> MAVEN_DEPLOY
    
    PACKAGE_JSON --> NPM_PACK
    NPM_PACK --> NPM_PUBLISH
    
    CONAN_PACKAGE --> VCPKG_PORT
    VCPKG_PORT --> SYSTEM_PACKAGES
```

## Build Tool Integration

Integration with various development tools and IDEs:

```mermaid
graph LR
    subgraph "Build Tool Integration"
        subgraph "IDEs"
            VISUAL_STUDIO[Visual Studio]
            VSCODE[VS Code]
            CLION[CLion]
            XCODE[Xcode]
        end
        
        subgraph "Build Systems"
            NINJA[Ninja]
            MAKE[Make]
            MSBUILD[MSBuild]
            XCODEBUILD[xcodebuild]
        end
        
        subgraph "Analysis Tools"
            STATIC_ANALYSIS[Static Analysis]
            CODE_COVERAGE[Code Coverage]
            MEMORY_SANITIZER[Memory Sanitizer]
            THREAD_SANITIZER[Thread Sanitizer]
        end
        
        subgraph "Debugging Tools"
            GDB[GDB]
            LLDB[LLDB]
            VISUAL_STUDIO_DEBUGGER[VS Debugger]
            CUDA_GDB[CUDA-GDB]
        end
    end
    
    VISUAL_STUDIO --> MSBUILD
    VSCODE --> NINJA
    CLION --> MAKE
    XCODE --> XCODEBUILD
    
    NINJA --> STATIC_ANALYSIS
    MAKE --> CODE_COVERAGE
    MSBUILD --> MEMORY_SANITIZER
    XCODEBUILD --> THREAD_SANITIZER
    
    STATIC_ANALYSIS --> GDB
    CODE_COVERAGE --> LLDB
    MEMORY_SANITIZER --> VISUAL_STUDIO_DEBUGGER
    THREAD_SANITIZER --> CUDA_GDB
```

## Build Performance Metrics

Key metrics for monitoring and optimizing build performance:

```mermaid
mindmap
    root((Build Performance))
        Build Times
            Total Build Time
            Incremental Build Time
            Clean Build Time
            Package Generation Time
        Resource Usage
            CPU Utilization
            Memory Usage
            Disk I/O
            Network Bandwidth
        Cache Efficiency
            Cache Hit Ratio
            Cache Size
            Cache Miss Penalty
            Dependency Resolution
        Parallelization
            Thread Utilization
            Job Queue Depth
            Load Balancing
            Synchronization Overhead
```

## Build System Maintenance

Ongoing maintenance and updates to the build system:

```mermaid
graph TB
    subgraph "Build System Maintenance"
        subgraph "Dependency Updates"
            VERSION_BUMPS[Version Bumps]
            SECURITY_PATCHES[Security Patches]
            COMPATIBILITY_CHECKS[Compatibility Checks]
            DEPENDENCY_AUDIT[Dependency Audit]
        end
        
        subgraph "Platform Support"
            NEW_PLATFORMS[New Platform Support]
            DEPRECATED_PLATFORMS[Deprecated Platform Removal]
            TOOLCHAIN_UPDATES[Toolchain Updates]
            COMPILER_UPGRADES[Compiler Upgrades]
        end
        
        subgraph "Build Optimization"
            PERFORMANCE_TUNING[Performance Tuning]
            CACHE_OPTIMIZATION[Cache Optimization]
            PARALLELIZATION_IMPROVEMENTS[Parallelization Improvements]
            RESOURCE_OPTIMIZATION[Resource Optimization]
        end
        
        subgraph "Documentation"
            BUILD_GUIDES[Build Guides]
            TROUBLESHOOTING[Troubleshooting Guides]
            CONFIGURATION_DOCS[Configuration Documentation]
            MIGRATION_GUIDES[Migration Guides]
        end
    end
    
    VERSION_BUMPS --> NEW_PLATFORMS
    SECURITY_PATCHES --> DEPRECATED_PLATFORMS
    COMPATIBILITY_CHECKS --> TOOLCHAIN_UPDATES
    DEPENDENCY_AUDIT --> COMPILER_UPGRADES
    
    NEW_PLATFORMS --> PERFORMANCE_TUNING
    DEPRECATED_PLATFORMS --> CACHE_OPTIMIZATION
    TOOLCHAIN_UPDATES --> PARALLELIZATION_IMPROVEMENTS
    COMPILER_UPGRADES --> RESOURCE_OPTIMIZATION
    
    PERFORMANCE_TUNING --> BUILD_GUIDES
    CACHE_OPTIMIZATION --> TROUBLESHOOTING
    PARALLELIZATION_IMPROVEMENTS --> CONFIGURATION_DOCS
    RESOURCE_OPTIMIZATION --> MIGRATION_GUIDES
```

This comprehensive build system architecture ensures reliable, efficient, and maintainable builds across all supported platforms and configurations while providing flexibility for different deployment scenarios.