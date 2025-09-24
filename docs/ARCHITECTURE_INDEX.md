# ONNX Runtime Architecture Documentation Index

This document serves as a comprehensive index to all architectural documentation for ONNX Runtime, providing organized access to technical documentation across different aspects of the system.

## 🏗️ Core Architecture

### System Overview
- **[Technical Architecture](technical_architecture.md)** - Comprehensive system architecture overview with mermaid diagrams
  - High-level system architecture
  - Core components and their relationships
  - Data flow and processing pipeline
  - Component interaction diagrams

### Detailed Component Architecture
- **[Architecture Directory](architecture/)** - Detailed component-specific documentation
  - **[Execution Provider Architecture](architecture/execution_providers.md)** - Hardware acceleration layer
  - **[Graph Processing Pipeline](architecture/graph_processing.md)** - Model optimization and transformation
  - **[Memory Management System](architecture/memory_management.md)** - Advanced memory strategies
  - **[Language Binding Architecture](architecture/language_bindings.md)** - Multi-language API implementations

## 🚀 Development & Deployment

### Build System
- **[Build System Architecture](architecture/build_system.md)** - Comprehensive build system documentation
  - Cross-platform build configuration
  - Dependency management
  - CI/CD pipeline architecture
  - Package generation and distribution

### Deployment Strategies
- **[Deployment Patterns](architecture/deployment_patterns.md)** - Environment-specific deployment architectures
  - Cloud deployment patterns
  - Edge computing architectures
  - Mobile deployment strategies
  - Web deployment patterns
  - Hybrid deployment models

## 🧠 Training vs Inference

### Specialized Architectures
- **[Training vs Inference Architecture](architecture/training_vs_inference.md)** - Architectural differences and optimizations
  - ORTModule architecture for training
  - Inference-specific optimizations
  - Memory management differences
  - Performance characteristics comparison

## 🛠️ Implementation Guides

### Core Implementation
- **[C API Guidelines](C_API_Guidelines.md)** - Native API design principles
- **[Memory Optimizer](Memory_Optimizer.md)** - Memory optimization techniques
- **[Build Guidelines](cmake_guideline.md)** - Build system usage guide

### Development Standards
- **[Coding Conventions](Coding_Conventions_and_Standards.md)** - Development standards and best practices
- **[Contributing Guidelines](../CONTRIBUTING.md)** - How to contribute to the project
- **[PR Guidelines](PR_Guidelines.md)** - Pull request standards and process

## 📚 Provider-Specific Documentation

### Execution Providers
- **[Execution Provider Documentation](execution_providers/)** - Provider-specific implementation details
- **GPU Providers**: CUDA, ROCm, DirectML, OpenVINO
- **CPU Providers**: CPU, oneDNN, XNNPACK
- **Mobile Providers**: CoreML, NNAPI, QNN, SNPE
- **Web Providers**: WebGL, WebGPU, WebNN, JSEP

## 🌐 Language Bindings

### API Documentation
- **[Python API](python/)** - Python binding implementation and usage
- **[C/C++ API](c_cxx/)** - Native C/C++ API documentation
- **[JavaScript/Web API](../js/web/)** - Web and Node.js implementations
- **[C# API](../csharp/)** - .NET binding documentation
- **[Java API](../java/)** - Java binding implementation
- **[Objective-C API](../objectivec/)** - iOS/macOS native integration

## 📋 Reference Documentation

### Model Support
- **[Model Testing](Model_Test.md)** - Model testing and validation procedures
- **[Operator Kernels](OperatorKernels.md)** - Operator implementation details
- **[Contrib Operators](ContribOperators.md)** - Custom operator implementation

### Operational Guides
- **[Server Usage](ONNX_Runtime_Server_Usage.md)** - Server deployment and configuration
- **[Android Testing](Android_testing.md)** - Android-specific testing procedures
- **[FAQ](FAQ.md)** - Frequently asked questions and troubleshooting

## 🔧 Advanced Topics

### Performance & Optimization
- **[Memory Optimizer](Memory_Optimizer.md)** - Advanced memory optimization strategies
- **[ORT Use Triton Kernel](ORT_Use_Triton_Kernel.md)** - Custom kernel development with Triton
- **[Reduced Operator Kernel Build](Reduced_Operator_Kernel_build.md)** - Minimal build configurations

### Training-Specific
- **[ORTModule Training Guidelines](ORTModule_Training_Guidelines.md)** - Training framework usage
- **[ORTModule Convergence Notes](ORTModule_Convergence_Notes.md)** - Training convergence optimization
- **[ORTModule Python Op Notes](ORTModule_PythonOp_Notes.md)** - Custom Python operator integration

## 🔄 Version Management

### Release Information
- **[Versioning](Versioning.md)** - Version numbering and compatibility
- **[Release Management](ReleaseManagement.md)** - Release process and lifecycle
- **[Roadmap](Roadmap.md)** - Future development plans
- **[ORT Format Updates](ORT_Format_Update_in_1.13.md)** - Format version changes

## 🔍 Development Notes

### Internal Documentation
- **[ABI Development Notes](ABI_Dev_Notes.md)** - ABI compatibility considerations
- **[Threading Notes](NotesOnThreading.md)** - Threading model and considerations
- **[ONNX Update Notes](How_To_Update_ONNX_Dev_Notes.md)** - ONNX version update procedures

## 📊 Visual Architecture Summary

The documentation includes comprehensive mermaid diagrams covering:

- **System Architecture**: High-level component relationships
- **Data Flow**: Processing pipeline and data transformations
- **Execution Providers**: Hardware abstraction and provider ecosystem
- **Memory Management**: Allocation strategies and optimization techniques
- **Build System**: Dependencies, packaging, and CI/CD workflows
- **Deployment Patterns**: Environment-specific architectures
- **Language Bindings**: FFI implementations and API layers
- **Training vs Inference**: Specialized architectural patterns

## 🔗 External Resources

- **[ONNX Runtime Website](https://onnxruntime.ai/)** - Official project website
- **[GitHub Repository](https://github.com/microsoft/onnxruntime)** - Source code and issues
- **[Documentation Site](https://onnxruntime.ai/docs/)** - User-facing documentation
- **[Model Hub](https://github.com/onnx/models)** - Pre-trained ONNX models

---

This index provides structured access to all architectural documentation. Each document contains detailed mermaid diagrams and technical specifications for its respective domain. For the most comprehensive overview, start with the [Technical Architecture](technical_architecture.md) document.