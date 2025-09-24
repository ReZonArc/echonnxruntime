# Cosmetic Chemistry Framework for ONNX Runtime

This document provides comprehensive guidance for using the cosmetic chemistry specializations within the ONNX Runtime cheminformatics framework. These extensions enable systematic analysis, optimization, and knowledge representation for cosmetic formulations.

## Table of Contents

1. [Overview](#overview)
2. [Atom Type Reference](#atom-type-reference)
3. [Common Cosmetic Ingredients Database](#common-cosmetic-ingredients-database)
4. [Formulation Guidelines](#formulation-guidelines)
5. [Regulatory Compliance](#regulatory-compliance)
6. [Advanced Applications](#advanced-applications)
7. [Examples and Usage](#examples-and-usage)

## Overview

The cosmetic chemistry framework extends ONNX Runtime with specialized atom types and knowledge representation capabilities for:

- **Ingredient Modeling**: Systematic classification and property modeling of cosmetic ingredients
- **Formulation Analysis**: Compatibility checking and optimization of product formulations
- **Regulatory Compliance**: Automated checking of concentration limits and safety requirements
- **Property Prediction**: Analysis of pH, stability, texture, and sensory properties

## Atom Type Reference

### Ingredient Categories

#### Primary Functional Types

| Atom Type | Description | Example Ingredients |
|-----------|-------------|-------------------|
| `ACTIVE_INGREDIENT` | Bioactive compounds with specific efficacy | Retinol, Niacinamide, Salicylic Acid |
| `PRESERVATIVE` | Antimicrobial protection agents | Phenoxyethanol, Parabens, Potassium Sorbate |
| `EMULSIFIER` | Oil-water phase stabilizers | Cetyl Alcohol, Polysorbate 20, Lecithin |
| `HUMECTANT` | Moisture-binding compounds | Glycerin, Hyaluronic Acid, Propylene Glycol |
| `SURFACTANT` | Cleansing and foaming agents | Sodium Lauryl Sulfate, Coco Glucoside |
| `THICKENER` | Viscosity enhancement agents | Xanthan Gum, Carbomer, Hydroxyethylcellulose |
| `EMOLLIENT` | Skin conditioning agents | Squalane, Jojoba Oil, Shea Butter |
| `ANTIOXIDANT` | Oxidation prevention compounds | Vitamin E, Vitamin C, BHT |
| `UV_FILTER` | Sun protection agents | Octinoxate, Zinc Oxide, Titanium Dioxide |
| `FRAGRANCE` | Scenting compounds | Essential Oils, Synthetic Fragrances |
| `COLORANT` | Coloring agents | Iron Oxides, Mica, FD&C Dyes |
| `PH_ADJUSTER` | pH modification agents | Citric Acid, Sodium Hydroxide, TEA |

### Formulation Types

| Atom Type | Description | Typical Ingredients |
|-----------|-------------|-------------------|
| `SKINCARE_FORMULATION` | Face and body care products | Moisturizers, serums, cleansers |
| `HAIRCARE_FORMULATION` | Hair care products | Shampoos, conditioners, treatments |
| `MAKEUP_FORMULATION` | Color cosmetics | Foundations, lipsticks, eyeshadows |
| `FRAGRANCE_FORMULATION` | Scented products | Perfumes, body sprays, candles |

### Property Types

| Atom Type | Description | Usage |
|-----------|-------------|-------|
| `PH_PROPERTY` | pH measurements (1-14 scale) | Formulation stability analysis |
| `VISCOSITY_PROPERTY` | Flow resistance measurements | Texture optimization |
| `STABILITY_PROPERTY` | Formulation stability metrics | Shelf-life prediction |
| `TEXTURE_PROPERTY` | Sensory texture attributes | Consumer acceptance |
| `SPF_PROPERTY` | Sun protection factor values | UV protection efficacy |

### Interaction Types

| Link Type | Description | Usage |
|-----------|-------------|-------|
| `COMPATIBILITY_LINK` | Compatible ingredient pairs | Safe formulation combinations |
| `INCOMPATIBILITY_LINK` | Incompatible ingredient pairs | Formulation warnings |
| `SYNERGY_LINK` | Synergistic interactions | Enhanced efficacy combinations |
| `ANTAGONISM_LINK` | Antagonistic interactions | Reduced efficacy combinations |

## Common Cosmetic Ingredients Database

### Active Ingredients

#### Anti-Aging Actives
- **Retinol** (Vitamin A): Cell turnover acceleration, anti-aging
- **Niacinamide** (Vitamin B3): Pore minimizing, oil control
- **Vitamin C (L-Ascorbic Acid)**: Antioxidant, brightening
- **Hyaluronic Acid**: Deep hydration, plumping
- **Peptides**: Collagen stimulation, firming

#### Exfoliating Actives
- **Salicylic Acid (BHA)**: Pore clearing, acne treatment
- **Glycolic Acid (AHA)**: Surface exfoliation, texture improvement
- **Lactic Acid (AHA)**: Gentle exfoliation, hydrating

### Base Ingredients

#### Emulsifiers
- **Cetyl Alcohol**: W/O emulsifier, thickening
- **Polysorbate 20**: O/W emulsifier, solubilizer
- **Lecithin**: Natural emulsifier, skin conditioning

#### Preservatives
- **Phenoxyethanol**: Broad-spectrum, gentle preservation
- **Benzyl Alcohol**: Natural preservative, fragrance component
- **Potassium Sorbate**: Natural preservative, food-grade

## Formulation Guidelines

### pH Considerations

| Ingredient Type | Optimal pH Range | Stability Notes |
|----------------|------------------|-----------------|
| Vitamin C | 3.0 - 4.0 | Requires acidic environment |
| Retinol | 5.5 - 6.5 | Neutral pH for stability |
| AHA/BHA | 3.0 - 4.0 | Acidic pH for efficacy |
| Niacinamide | 5.0 - 7.0 | Stable across wide range |

### Stability Factors

#### Temperature Sensitivity
- **Heat-Sensitive**: Vitamin C (L-Ascorbic Acid), Retinol
- **Heat-Stable**: Niacinamide, Hyaluronic Acid
- **Processing Temperature**: Most actives &lt; 60°C

#### Light Sensitivity
- **Photosensitive**: Retinol, Vitamin C, Essential Oils
- **Light-Stable**: Niacinamide, Hyaluronic Acid, Ceramides

#### Oxygen Sensitivity
- **Oxidation-Prone**: Vitamin C, Natural Oils, Retinol
- **Antioxidant Protection**: Vitamin E, BHT, Packaging considerations

### Compatibility Matrix

#### Known Incompatibilities

| Ingredient A | Ingredient B | Issue | Solution |
|-------------|-------------|-------|----------|
| Vitamin C | Retinol | pH conflict, irritation | Separate applications |
| Vitamin C | Niacinamide | pH conflict (disputed) | Monitor formulation pH |
| AHA/BHA | Retinol | Over-exfoliation | Alternate usage days |
| Benzoyl Peroxide | Retinol | Deactivation | Separate applications |

#### Synergistic Combinations

| Ingredient A | Ingredient B | Benefit | Usage Notes |
|-------------|-------------|---------|-------------|
| Vitamin C | Vitamin E | Enhanced antioxidant protection | Stable combination |
| Niacinamide | Zinc | Oil control synergy | Popular for acne |
| Hyaluronic Acid | Ceramides | Barrier function enhancement | Deep hydration |
| Peptides | Niacinamide | Anti-aging + pore minimizing | Gentle combination |

## Regulatory Compliance

### Concentration Limits (EU/US)

#### Active Ingredients

| Ingredient | EU Limit | US FDA Limit | Notes |
|-----------|----------|--------------|-------|
| Salicylic Acid | 2% (rinse-off), 0.5% (leave-on) | 2% (OTC) | BHA exfoliant |
| Benzoyl Peroxide | 10% | 10% (OTC) | Acne treatment |
| Retinol | No specific limit | No specific limit | Cosmetic use |
| Hydroquinone | Banned | 2% (OTC) | Regional difference |

#### Preservatives

| Preservative | EU Limit | US FDA Limit | Usage |
|-------------|----------|--------------|--------|
| Phenoxyethanol | 1% | No specific limit | Most common |
| Parabens | 0.4% (single), 0.8% (total) | No specific limit | Traditional |
| Formaldehyde Releasers | 0.2% (as HCHO) | No specific limit | Various types |

### INCI Naming Requirements

All cosmetic ingredients must use standardized International Nomenclature of Cosmetic Ingredients (INCI) names:

- **Water**: Aqua
- **Glycerin**: Glycerin
- **Vitamin E**: Tocopherol
- **Vitamin C**: L-Ascorbic Acid (pure form)

## Advanced Applications

### Formulation Optimization

#### Multi-Objective Optimization
```python
# Example optimization criteria
objectives = {
    'efficacy': maximize_active_concentration,
    'stability': minimize_degradation_rate,
    'cost': minimize_ingredient_cost,
    'sensory': maximize_consumer_acceptance
}
```

#### Constraint Satisfaction
```python
# Example constraints
constraints = {
    'ph_range': (5.0, 7.0),
    'viscosity_range': (1000, 10000),  # cP
    'total_active_concentration': ('<=', 10.0),  # %
    'preservative_system': 'required'
}
```

### Predictive Modeling

#### Stability Prediction
- Temperature stress testing simulation
- Oxidation kinetics modeling
- Microbial challenge test prediction

#### Sensory Prediction
- Texture attribute modeling
- Consumer preference prediction
- Absorption rate estimation

### Quality Control Integration

#### Batch Analysis
- Real-time pH monitoring
- Viscosity tracking
- Color consistency verification

#### Shelf-Life Prediction
- Accelerated aging models
- Chemical degradation tracking
- Package interaction analysis

## Examples and Usage

### Basic Ingredient Definition

```python
# Define basic cosmetic ingredients
hyaluronic_acid = ACTIVE_INGREDIENT('hyaluronic_acid')
glycerin = HUMECTANT('glycerin')
phenoxyethanol = PRESERVATIVE('phenoxyethanol')
cetyl_alcohol = EMULSIFIER('cetyl_alcohol')
```

### Formulation Creation

```python
# Create a skincare formulation
moisturizer = SKINCARE_FORMULATION(
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

### Property Assignment

```python
# Assign properties to formulations
ph_property = PH_PROPERTY(moisturizer, 6.2)
viscosity_property = VISCOSITY_PROPERTY(moisturizer, 5000)  # cP
spf_property = SPF_PROPERTY(sunscreen, 30)
```

### Safety Assessment

```python
# Safety and regulatory compliance
safety_assessment = SAFETY_ASSESSMENT(retinol, 'safe_concentration_0.1%')
allergen_classification = ALLERGEN_CLASSIFICATION(fragrance, 'potential_allergen')
concentration_limit = CONCENTRATION_LIMIT(salicylic_acid, '2%_max')
```

## Integration with ONNX Runtime

The cosmetic chemistry framework integrates seamlessly with ONNX Runtime's machine learning capabilities to enable:

### Predictive Analytics
- Formulation stability prediction using trained models
- Consumer preference modeling based on ingredient profiles
- Shelf-life estimation using accelerated aging data

### Automated Optimization
- Multi-objective formulation optimization
- Cost-efficacy trade-off analysis
- Regulatory compliance checking

### Knowledge Discovery
- Pattern recognition in successful formulations
- Ingredient interaction discovery
- Market trend analysis

## Getting Started

1. **Load the atom types**: Import the cosmetic chemistry atom types into your ONNX Runtime environment
2. **Define ingredients**: Create ingredient atoms using the appropriate functional categories  
3. **Build formulations**: Combine ingredients into formulation atoms
4. **Analyze interactions**: Use link types to model ingredient compatibility
5. **Optimize properties**: Apply machine learning models for property prediction and optimization

For detailed code examples, see the Python examples in `examples/python/` and Scheme examples in `examples/scheme/`.

## Support and Contributing

For questions, bug reports, or feature requests related to the cosmetic chemistry framework, please file issues in the main ONNX Runtime repository with the `cosmetic-chemistry` label.

Contributions are welcome! Please follow the standard ONNX Runtime contribution guidelines and ensure all cosmetic chemistry extensions maintain compatibility with the core framework.