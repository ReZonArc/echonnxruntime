#!/usr/bin/env python3
"""
Advanced Cosmetic Chemistry Formulation Analysis and Optimization

This example demonstrates sophisticated cosmetic formulation analysis using
the ONNX Runtime cheminformatics framework. It includes property modeling,
stability analysis, regulatory compliance checking, and formulation optimization.

Author: ONNX Runtime Cosmetic Chemistry Team
"""

import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# Import basic atom types from the intro example
from cosmetic_intro_example import *

# Additional specialized atom types for advanced analysis
class PH_PROPERTY(CosmeticAtom):
    """pH measurement values"""
    def __init__(self, target, ph_value):
        super().__init__(f"ph_{ph_value}")
        self.target = target
        self.ph_value = float(ph_value)

class VISCOSITY_PROPERTY(CosmeticAtom):
    """Flow resistance measurements"""
    def __init__(self, target, viscosity_cp):
        super().__init__(f"viscosity_{viscosity_cp}")
        self.target = target
        self.viscosity_cp = int(viscosity_cp)

class STABILITY_PROPERTY(CosmeticAtom):
    """Formulation stability metrics"""
    def __init__(self, target, stability_score):
        super().__init__(f"stability_{stability_score}")
        self.target = target
        self.stability_score = float(stability_score)

class CONCENTRATION_LIMIT(CosmeticAtom):
    """Maximum usage concentration"""
    def __init__(self, ingredient, max_concentration):
        super().__init__(f"limit_{max_concentration}")
        self.ingredient = ingredient
        self.max_concentration = max_concentration

class SAFETY_ASSESSMENT(CosmeticAtom):
    """Safety evaluation data"""
    def __init__(self, ingredient, safety_rating):
        super().__init__(f"safety_{safety_rating}")
        self.ingredient = ingredient
        self.safety_rating = safety_rating

# Advanced ingredient types
class SURFACTANT(CosmeticAtom):
    """Surface-active agents"""
    pass

class THICKENER(CosmeticAtom):
    """Viscosity modifiers"""
    pass

class UV_FILTER(CosmeticAtom):
    """UV radiation protection"""
    pass

@dataclass
class FormulationConstraints:
    """Constraints for formulation optimization"""
    ph_range: Tuple[float, float] = (5.0, 7.0)
    viscosity_range: Tuple[int, int] = (1000, 10000)
    max_active_concentration: float = 10.0
    requires_preservative: bool = True
    max_cost_per_100ml: float = 5.0

class StabilityPredictor:
    """Predict formulation stability based on ingredient interactions"""
    
    def __init__(self):
        # Simplified stability matrix - in practice this would be ML-based
        self.stability_factors = {
            ('vitamin_c_l_ascorbic_acid', 'low_ph'): 0.9,
            ('hyaluronic_acid', 'high_humidity'): 0.8,
            ('retinol', 'light_exposure'): 0.3,
            ('niacinamide', 'wide_ph_range'): 0.9,
        }
    
    def predict_stability(self, formulation, conditions=None):
        """Predict stability score for a formulation"""
        conditions = conditions or ['normal_storage']
        base_stability = 0.7
        
        # Check for known stability factors
        for ingredient in formulation.ingredients:
            ingredient_name = ingredient.name
            for condition in conditions:
                key = (ingredient_name, condition)
                if key in self.stability_factors:
                    base_stability *= self.stability_factors[key]
        
        # Bonus for antioxidant systems
        has_vitamin_c = any(ing.name == 'vitamin_c_l_ascorbic_acid' for ing in formulation.ingredients)
        has_vitamin_e = any(ing.name == 'vitamin_e_tocopherol' for ing in formulation.ingredients)
        if has_vitamin_c and has_vitamin_e:
            base_stability *= 1.2
        
        return min(1.0, base_stability)

class RegulatoryChecker:
    """Check regulatory compliance for formulations"""
    
    def __init__(self):
        self.concentration_limits = {
            'salicylic_acid': {'EU': 2.0, 'US': 2.0},
            'retinol': {'EU': None, 'US': None},  # No specific limits
            'vitamin_c_l_ascorbic_acid': {'EU': 20.0, 'US': 20.0},
            'niacinamide': {'EU': 10.0, 'US': 10.0},
            'phenoxyethanol': {'EU': 1.0, 'US': None}
        }
        
        self.allergen_list = [
            'fragrance', 'essential_oils', 'lanolin', 'formaldehyde_releasers'
        ]
    
    def check_concentration_limits(self, ingredient, concentration, region='EU'):
        """Check if ingredient concentration meets regulatory limits"""
        ingredient_name = ingredient.name
        if ingredient_name in self.concentration_limits:
            limit = self.concentration_limits[ingredient_name].get(region)
            if limit is not None:
                return concentration <= limit
        return True  # No known limit
    
    def check_allergen_labeling(self, formulation):
        """Check if formulation contains potential allergens"""
        allergens_present = []
        for ingredient in formulation.ingredients:
            if any(allergen in ingredient.name for allergen in self.allergen_list):
                allergens_present.append(ingredient.name)
        return allergens_present

class FormulationOptimizer:
    """Optimize formulations based on multiple criteria"""
    
    def __init__(self):
        self.stability_predictor = StabilityPredictor()
        self.regulatory_checker = RegulatoryChecker()
        
        # Simplified cost database (per 100ml formulation)
        self.ingredient_costs = {
            'hyaluronic_acid': 2.50,
            'niacinamide': 0.75,
            'vitamin_c_l_ascorbic_acid': 3.00,
            'vitamin_e_tocopherol': 1.20,
            'glycerin': 0.15,
            'phenoxyethanol': 0.25,
            'cetyl_alcohol': 0.40,
            'salicylic_acid': 1.80,
        }
    
    def calculate_formulation_cost(self, formulation, concentrations=None):
        """Calculate estimated cost per 100ml"""
        concentrations = concentrations or {}
        total_cost = 0.0
        
        for ingredient in formulation.ingredients:
            base_cost = self.ingredient_costs.get(ingredient.name, 1.0)
            concentration = concentrations.get(ingredient.name, 1.0)  # Default 1%
            ingredient_cost = base_cost * (concentration / 100)
            total_cost += ingredient_cost
        
        return total_cost
    
    def optimize_formulation(self, formulation, constraints: FormulationConstraints):
        """Optimize formulation based on constraints"""
        results = {
            'formulation': formulation,
            'stability_score': self.stability_predictor.predict_stability(formulation),
            'estimated_cost': self.calculate_formulation_cost(formulation),
            'regulatory_compliance': True,
            'optimization_suggestions': []
        }
        
        # Check regulatory compliance
        allergens = self.regulatory_checker.check_allergen_labeling(formulation)
        if allergens:
            results['allergens_present'] = allergens
            results['optimization_suggestions'].append(
                f"Label allergens: {', '.join(allergens)}"
            )
        
        # Check cost constraints
        if results['estimated_cost'] > constraints.max_cost_per_100ml:
            results['optimization_suggestions'].append(
                f"Reduce cost from ${results['estimated_cost']:.2f} to ${constraints.max_cost_per_100ml:.2f}"
            )
        
        # Check for preservative requirement
        has_preservative = any(isinstance(ing, PRESERVATIVE) for ing in formulation.ingredients)
        if constraints.requires_preservative and not has_preservative:
            results['optimization_suggestions'].append("Add preservative system")
            results['regulatory_compliance'] = False
        
        return results

def main():
    """Main example demonstrating advanced cosmetic chemistry analysis"""
    
    print("=== Advanced Cosmetic Chemistry Analysis ===\n")
    
    # 1. Create Advanced Ingredient Library
    print("1. Creating Advanced Ingredient Library:")
    print("-" * 50)
    
    # Advanced actives
    retinol = ACTIVE_INGREDIENT('retinol', {
        'concentration_limit': '1%',
        'ph_stability': '5.5-6.5',
        'light_sensitive': True,
        'function': 'anti_aging'
    })
    
    salicylic_acid = ACTIVE_INGREDIENT('salicylic_acid', {
        'concentration_limit': '2%',
        'ph_stability': '3.0-4.0',
        'function': 'exfoliating_bha'
    })
    
    # Specialized base ingredients
    sodium_hyaluronate = HUMECTANT('sodium_hyaluronate', {
        'molecular_weight': 'low',
        'penetration': 'enhanced',
        'function': 'deep_hydration'
    })
    
    cocamidopropyl_betaine = SURFACTANT('cocamidopropyl_betaine', {
        'type': 'amphoteric',
        'gentleness': 'high',
        'foaming': 'excellent'
    })
    
    xanthan_gum = THICKENER('xanthan_gum', {
        'type': 'natural_polymer',
        'viscosity_efficiency': 'high',
        'shear_thinning': True
    })
    
    zinc_oxide = UV_FILTER('zinc_oxide', {
        'type': 'mineral',
        'spf_contribution': 'high',
        'broad_spectrum': True
    })
    
    print(f"Retinol: {retinol}")
    print(f"Salicylic Acid: {salicylic_acid}")
    print(f"Sodium Hyaluronate: {sodium_hyaluronate}")
    print(f"Cocamidopropyl Betaine: {cocamidopropyl_betaine}")
    print(f"Xanthan Gum: {xanthan_gum}")
    print(f"Zinc Oxide: {zinc_oxide}")
    
    # 2. Create Complex Formulations
    print("\n2. Creating Complex Formulations:")
    print("-" * 50)
    
    # Anti-aging night serum
    night_serum = SKINCARE_FORMULATION('anti_aging_night_serum')
    night_serum.ingredients = [retinol, sodium_hyaluronate, 
                              ANTIOXIDANT('vitamin_e_tocopherol'),
                              PRESERVATIVE('phenoxyethanol')]
    
    # Exfoliating cleanser
    exfoliating_cleanser = SKINCARE_FORMULATION('bha_exfoliating_cleanser')
    exfoliating_cleanser.ingredients = [salicylic_acid, cocamidopropyl_betaine,
                                       xanthan_gum, PRESERVATIVE('phenoxyethanol')]
    
    # Broad spectrum sunscreen
    sunscreen = SKINCARE_FORMULATION('mineral_sunscreen_spf30')
    sunscreen.ingredients = [zinc_oxide, EMULSIFIER('cetyl_alcohol'),
                           HUMECTANT('glycerin'), PRESERVATIVE('phenoxyethanol')]
    
    print(f"Night Serum: {night_serum}")
    print(f"Exfoliating Cleanser: {exfoliating_cleanser}")
    print(f"Sunscreen: {sunscreen}")
    
    # 3. Property Modeling
    print("\n3. Property Modeling:")
    print("-" * 50)
    
    # Assign properties to formulations
    night_serum_ph = PH_PROPERTY(night_serum, 6.0)
    night_serum_viscosity = VISCOSITY_PROPERTY(night_serum, 2000)
    night_serum_stability = STABILITY_PROPERTY(night_serum, 0.85)
    
    cleanser_ph = PH_PROPERTY(exfoliating_cleanser, 3.8)
    cleanser_viscosity = VISCOSITY_PROPERTY(exfoliating_cleanser, 8000)
    
    sunscreen_ph = PH_PROPERTY(sunscreen, 6.5)
    sunscreen_viscosity = VISCOSITY_PROPERTY(sunscreen, 15000)
    
    print(f"Night Serum pH: {night_serum_ph.ph_value}")
    print(f"Night Serum Viscosity: {night_serum_viscosity.viscosity_cp} cP")
    print(f"Night Serum Stability: {night_serum_stability.stability_score}")
    print(f"Cleanser pH: {cleanser_ph.ph_value}")
    print(f"Sunscreen Viscosity: {sunscreen_viscosity.viscosity_cp} cP")
    
    # 4. Stability Analysis
    print("\n4. Stability Analysis:")
    print("-" * 50)
    
    stability_predictor = StabilityPredictor()
    
    formulations = [night_serum, exfoliating_cleanser, sunscreen]
    conditions = ['normal_storage', 'high_temperature', 'light_exposure']
    
    for formulation in formulations:
        for condition in conditions:
            stability = stability_predictor.predict_stability(formulation, [condition])
            print(f"{formulation.name} - {condition}: {stability:.2f}")
    
    # 5. Regulatory Compliance Check
    print("\n5. Regulatory Compliance Analysis:")
    print("-" * 50)
    
    regulatory_checker = RegulatoryChecker()
    
    # Check concentration limits (example concentrations)
    test_concentrations = [
        (retinol, 0.5, 'EU'),
        (salicylic_acid, 2.0, 'EU'),
        (PRESERVATIVE('phenoxyethanol'), 0.8, 'EU')
    ]
    
    for ingredient, concentration, region in test_concentrations:
        compliant = regulatory_checker.check_concentration_limits(ingredient, concentration, region)
        status = "✓ Compliant" if compliant else "✗ Exceeds limit"
        print(f"{ingredient.name} at {concentration}% ({region}): {status}")
    
    # Check allergen labeling
    for formulation in formulations:
        allergens = regulatory_checker.check_allergen_labeling(formulation)
        if allergens:
            print(f"{formulation.name} contains potential allergens: {allergens}")
        else:
            print(f"{formulation.name}: No known allergens detected")
    
    # 6. Formulation Optimization
    print("\n6. Formulation Optimization:")
    print("-" * 50)
    
    optimizer = FormulationOptimizer()
    constraints = FormulationConstraints(
        ph_range=(5.5, 7.0),
        viscosity_range=(1000, 12000),
        max_active_concentration=8.0,
        requires_preservative=True,
        max_cost_per_100ml=4.0
    )
    
    for formulation in formulations:
        results = optimizer.optimize_formulation(formulation, constraints)
        
        print(f"\nOptimization Results for {formulation.name}:")
        print(f"  Stability Score: {results['stability_score']:.2f}")
        print(f"  Estimated Cost: ${results['estimated_cost']:.2f}/100ml")
        print(f"  Regulatory Compliant: {results['regulatory_compliance']}")
        
        if results['optimization_suggestions']:
            print(f"  Suggestions:")
            for suggestion in results['optimization_suggestions']:
                print(f"    • {suggestion}")
        else:
            print(f"  ✓ Formulation appears optimal")
        
        if 'allergens_present' in results:
            print(f"  ⚠ Allergens to label: {', '.join(results['allergens_present'])}")
    
    # 7. Advanced Interaction Analysis
    print("\n7. Advanced Interaction Analysis:")
    print("-" * 50)
    
    # Define complex interaction network
    interactions = [
        SYNERGY_LINK(retinol, ANTIOXIDANT('vitamin_e_tocopherol'), strength=0.8),
        INCOMPATIBILITY_LINK(retinol, salicylic_acid, strength=0.9),
        COMPATIBILITY_LINK(sodium_hyaluronate, HUMECTANT('glycerin'), strength=0.9),
        SYNERGY_LINK(zinc_oxide, ANTIOXIDANT('vitamin_e_tocopherol'), strength=0.7)
    ]
    
    print("Interaction Network:")
    for interaction in interactions:
        interaction_type = interaction.__class__.__name__.replace('_LINK', '').lower()
        print(f"  {interaction.atom1.name} ↔ {interaction.atom2.name}: "
              f"{interaction_type.capitalize()} (strength: {interaction.strength})")
    
    # 8. Predictive Modeling Example
    print("\n8. Predictive Modeling Example:")
    print("-" * 50)
    
    # Simulate machine learning predictions (simplified)
    def predict_consumer_acceptance(formulation):
        """Simulate ML-based consumer acceptance prediction"""
        # Simplified scoring based on ingredient types
        score = 0.5  # Base score
        
        for ingredient in formulation.ingredients:
            if isinstance(ingredient, ACTIVE_INGREDIENT):
                score += 0.15  # Consumers like active ingredients
            if isinstance(ingredient, HUMECTANT):
                score += 0.10  # Hydration improves acceptance
            if isinstance(ingredient, SURFACTANT):
                score -= 0.05  # Can be harsh
        
        return min(1.0, score)
    
    print("Consumer Acceptance Predictions:")
    for formulation in formulations:
        acceptance = predict_consumer_acceptance(formulation)
        rating = "High" if acceptance > 0.7 else "Medium" if acceptance > 0.5 else "Low"
        print(f"  {formulation.name}: {acceptance:.2f} ({rating})")
    
    # 9. Quality Control Integration
    print("\n9. Quality Control Integration:")
    print("-" * 50)
    
    def generate_qc_report(formulation, properties):
        """Generate quality control report"""
        print(f"\nQC Report for {formulation.name}:")
        print(f"  Batch ID: QC-{formulation.name.upper()}-001")
        
        for prop in properties:
            if isinstance(prop, PH_PROPERTY):
                status = "✓ PASS" if 5.0 <= prop.ph_value <= 7.5 else "✗ FAIL"
                print(f"  pH: {prop.ph_value} {status}")
            elif isinstance(prop, VISCOSITY_PROPERTY):
                status = "✓ PASS" if 1000 <= prop.viscosity_cp <= 20000 else "✗ FAIL"
                print(f"  Viscosity: {prop.viscosity_cp} cP {status}")
    
    # Generate QC reports
    night_serum_props = [night_serum_ph, night_serum_viscosity, night_serum_stability]
    cleanser_props = [cleanser_ph, cleanser_viscosity]
    
    generate_qc_report(night_serum, night_serum_props)
    generate_qc_report(exfoliating_cleanser, cleanser_props)
    
    print("\n=== Advanced Analysis Complete ===")
    print("\nThis example demonstrated:")
    print("• Advanced ingredient library and specialized atom types")
    print("• Complex formulation creation and property modeling")
    print("• Stability prediction using interaction matrices")
    print("• Regulatory compliance checking and concentration limits")
    print("• Multi-objective formulation optimization")
    print("• Advanced interaction network analysis")
    print("• Machine learning integration for predictive modeling")
    print("• Quality control reporting and batch analysis")
    print("\nThe framework provides a comprehensive foundation for")
    print("computational cosmetic chemistry and formulation optimization!")

if __name__ == "__main__":
    main()