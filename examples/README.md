# ONNX Runtime Cosmetic Chemistry Examples

This directory contains comprehensive examples demonstrating the cosmetic chemistry specializations within the ONNX Runtime cheminformatics framework.

## Overview

The cosmetic chemistry framework extends ONNX Runtime with specialized atom types and knowledge representation capabilities for cosmetic formulation analysis, optimization, and compliance checking.

## Directory Structure

```
examples/
├── python/                           # Python examples and tests
│   ├── cosmetic_intro_example.py     # Basic introduction to cosmetic atom types
│   ├── cosmetic_chemistry_example.py # Advanced formulation analysis
│   └── test_cosmetic_chemistry.py    # Comprehensive test suite
└── scheme/                           # Scheme examples
    ├── cosmetic_formulation.scm      # Complex formulation modeling
    └── cosmetic_compatibility.scm    # Simple compatibility checking
```

## Python Examples

### 1. Basic Introduction (`cosmetic_intro_example.py`)

A beginner-friendly introduction that covers:
- Basic cosmetic ingredient definition using atom types
- Simple formulation creation and ingredient combination
- Compatibility analysis using link types
- Formulation analysis and completeness checking

**Run the example:**
```bash
cd examples/python
python3 cosmetic_intro_example.py
```

### 2. Advanced Analysis (`cosmetic_chemistry_example.py`)

A comprehensive example demonstrating:
- Advanced ingredient library with specialized atom types
- Complex formulation creation and property modeling
- Stability prediction using interaction matrices
- Regulatory compliance checking and concentration limits
- Multi-objective formulation optimization
- Machine learning integration for predictive modeling

**Run the example:**
```bash
cd examples/python
python3 cosmetic_chemistry_example.py
```

### 3. Test Suite (`test_cosmetic_chemistry.py`)

Comprehensive validation of the framework including:
- Unit tests for all atom types and functionality
- Integration tests for complete workflows
- Validation of stability prediction and optimization
- Regulatory compliance testing

**Run the tests:**
```bash
cd examples/python
python3 test_cosmetic_chemistry.py
```

## Scheme Examples

### 1. Complex Formulation Modeling (`cosmetic_formulation.scm`)

Demonstrates advanced formulation modeling using Scheme including:
- Detailed ingredient definitions with properties
- Complex compatibility and interaction links
- Property assignments (pH, viscosity, stability)
- Safety and regulatory assessments
- Automated analysis rules and optimization

### 2. Simple Compatibility Checking (`cosmetic_compatibility.scm`)

A focused example on ingredient compatibility:
- Basic ingredient definitions
- Compatibility matrix generation
- Simple formulation safety analysis
- Individual ingredient pair checking
- Practical recommendations

## Key Features Demonstrated

### Ingredient Modeling
```python
# Define cosmetic ingredients with functional classifications
hyaluronic_acid = ACTIVE_INGREDIENT('hyaluronic_acid')
glycerin = HUMECTANT('glycerin')
phenoxyethanol = PRESERVATIVE('phenoxyethanol')
```

### Formulation Creation
```python
# Create complex cosmetic formulations
moisturizer = SKINCARE_FORMULATION(
    'moisturizer',
    hyaluronic_acid,    # Hydrating active
    cetyl_alcohol,      # Emulsifier
    glycerin,           # Humectant
    phenoxyethanol      # Preservative
)
```

### Compatibility Analysis  
```python
# Define ingredient interactions
compatible = COMPATIBILITY_LINK(hyaluronic_acid, niacinamide)
incompatible = INCOMPATIBILITY_LINK(vitamin_c, retinol)
synergy = SYNERGY_LINK(vitamin_c, vitamin_e)
```

## Practical Applications

The framework enables:

- **Formulation Optimization**: Systematic ingredient selection and compatibility checking
- **Stability Prediction**: Analysis of formulation stability factors and pH requirements  
- **Regulatory Compliance**: Automated checking of concentration limits and allergen declarations
- **Ingredient Substitution**: Finding compatible alternatives for formulation improvements
- **Property Modeling**: pH, viscosity, SPF, and sensory property analysis

## Getting Started

1. **Start with the basic introduction**: Run `cosmetic_intro_example.py` to understand fundamental concepts
2. **Explore advanced features**: Run `cosmetic_chemistry_example.py` for comprehensive analysis capabilities
3. **Validate your understanding**: Run `test_cosmetic_chemistry.py` to see all components in action
4. **Review the Scheme examples**: Examine the `.scm` files for knowledge representation approaches

## Requirements

- Python 3.6+
- Basic understanding of cosmetic chemistry concepts
- Familiarity with ONNX Runtime framework (helpful but not required)

## Documentation

For detailed documentation on the cosmetic chemistry framework, see:
- [COSMETIC_CHEMISTRY.md](../docs/COSMETIC_CHEMISTRY.md) - Complete reference guide
- [Atom Types Reference](../cheminformatics/types/atom_types.script) - Full atom type definitions

## Support

For questions about the cosmetic chemistry examples:
1. Review the comprehensive documentation in `docs/COSMETIC_CHEMISTRY.md`
2. Run the test suite to validate your environment
3. Examine the example code for implementation patterns
4. File issues in the main ONNX Runtime repository with the `cosmetic-chemistry` label

## Contributing

When contributing new examples:
1. Follow the existing code style and patterns
2. Include comprehensive docstrings and comments
3. Add corresponding tests in `test_cosmetic_chemistry.py`
4. Update this README with new example descriptions
5. Ensure examples run successfully and produce expected output