#!/usr/bin/env python3
"""
🧬 Hypergredient Framework Architecture

Revolutionary Formulation Design System that abstracts ingredients into
functional hypergredient classes with multi-objective optimization.

Author: ONNX Runtime Cosmeceutical Optimization Team
"""

import json
import math
import random
from typing import Dict, List, Tuple, Set, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import copy

# Import existing cosmetic chemistry framework
from cosmetic_chemistry_example import *


# Core Hypergredient Classes
HYPERGREDIENT_DATABASE = {
    "H.CT": "Cellular Turnover Agents",
    "H.CS": "Collagen Synthesis Promoters", 
    "H.AO": "Antioxidant Systems",
    "H.BR": "Barrier Repair Complex",
    "H.ML": "Melanin Modulators",
    "H.HY": "Hydration Systems",
    "H.AI": "Anti-Inflammatory Agents",
    "H.MB": "Microbiome Balancers",
    "H.SE": "Sebum Regulators",
    "H.PD": "Penetration/Delivery Enhancers"
}


@dataclass
class HypergredientProperties:
    """Properties defining a hypergredient ingredient"""
    name: str
    inci_name: str
    hypergredient_class: str
    primary_function: str 
    secondary_functions: List[str] = field(default_factory=list)
    efficacy_score: float = 0.0  # 0-10 scale
    bioavailability: float = 0.0  # 0-1 scale (percentage as decimal)
    pH_min: float = 4.0
    pH_max: float = 8.0
    stability_conditions: Dict[str, Any] = field(default_factory=dict)
    incompatibilities: List[str] = field(default_factory=list)
    synergies: List[str] = field(default_factory=list)
    cost_per_gram: float = 0.0  # ZAR
    clinical_evidence: str = "None"
    safety_score: float = 5.0  # 1-10 scale (10 = safest)


class HypergredientDatabase:
    """Dynamic database of hypergredients with optimization capabilities"""
    
    def __init__(self):
        self.ingredients = {}
        self.interaction_matrix = {}
        self._initialize_database()
        self._initialize_interactions()
    
    def _initialize_database(self):
        """Initialize the hypergredient database with comprehensive ingredient data"""
        
        # H.CT - Cellular Turnover Agents
        self.ingredients.update({
            "tretinoin": HypergredientProperties(
                name="Tretinoin",
                inci_name="Tretinoin", 
                hypergredient_class="H.CT",
                primary_function="cellular_turnover",
                secondary_functions=["anti_aging", "acne_treatment"],
                efficacy_score=10.0,
                bioavailability=0.85,
                pH_min=5.5, pH_max=6.5,
                stability_conditions={"uv_sensitive": True},
                incompatibilities=["benzoyl_peroxide"],
                cost_per_gram=15.00,
                clinical_evidence="Strong",
                safety_score=6.0
            ),
            "bakuchiol": HypergredientProperties(
                name="Bakuchiol",
                inci_name="Bakuchiol",
                hypergredient_class="H.CT", 
                primary_function="cellular_turnover",
                secondary_functions=["antioxidant"],
                efficacy_score=7.0,
                bioavailability=0.70,
                pH_min=4.0, pH_max=9.0,
                stability_conditions={"stable": True},
                synergies=["vitamin_c", "niacinamide"],
                cost_per_gram=240.00,
                clinical_evidence="Moderate",
                safety_score=9.0
            ),
            "retinol": HypergredientProperties(
                name="Retinol",
                inci_name="Retinol",
                hypergredient_class="H.CT",
                primary_function="cellular_turnover", 
                secondary_functions=["anti_aging"],
                efficacy_score=8.0,
                bioavailability=0.60,
                pH_min=5.5, pH_max=6.5,
                stability_conditions={"oxygen_sensitive": True},
                incompatibilities=["aha", "bha"],
                cost_per_gram=180.00,
                clinical_evidence="Strong",
                safety_score=7.0
            ),
            "glycolic_acid": HypergredientProperties(
                name="Glycolic Acid",
                inci_name="Glycolic Acid",
                hypergredient_class="H.CT",
                primary_function="cellular_turnover",
                secondary_functions=["exfoliation"],
                efficacy_score=6.0,
                bioavailability=0.90,
                pH_min=3.5, pH_max=4.5,
                stability_conditions={"stable": True},
                incompatibilities=["retinoids"],
                cost_per_gram=45.00,
                clinical_evidence="Strong",
                safety_score=7.0
            )
        })
        
        # H.CS - Collagen Synthesis Promoters  
        self.ingredients.update({
            "matrixyl_3000": HypergredientProperties(
                name="Matrixyl 3000",
                inci_name="Palmitoyl Tripeptide-1, Palmitoyl Tetrapeptide-7",
                hypergredient_class="H.CS",
                primary_function="collagen_synthesis",
                secondary_functions=["skin_repair"],
                efficacy_score=9.0,
                bioavailability=0.75,
                pH_min=5.0, pH_max=7.0,
                stability_conditions={"stable": True},
                synergies=["vitamin_c"],
                cost_per_gram=120.00,
                clinical_evidence="Strong",
                safety_score=9.0
            ),
            "vitamin_c_l_aa": HypergredientProperties(
                name="Vitamin C (L-Ascorbic Acid)",
                inci_name="L-Ascorbic Acid",
                hypergredient_class="H.CS",
                primary_function="collagen_synthesis",
                secondary_functions=["antioxidant", "brightening"],
                efficacy_score=8.0,
                bioavailability=0.85,
                pH_min=3.0, pH_max=4.0,
                stability_conditions={"oxidation_sensitive": True},
                incompatibilities=["copper_peptides"],
                synergies=["vitamin_e", "ferulic_acid"],
                cost_per_gram=85.00,
                clinical_evidence="Strong", 
                safety_score=8.0
            ),
            "copper_peptides": HypergredientProperties(
                name="Copper Peptides",
                inci_name="Copper Tripeptide-1",
                hypergredient_class="H.CS",
                primary_function="collagen_synthesis",
                secondary_functions=["skin_remodeling"],
                efficacy_score=8.0,
                bioavailability=0.70,
                pH_min=6.0, pH_max=8.0,
                stability_conditions={"stable": True},
                incompatibilities=["vitamin_c"],
                cost_per_gram=390.00,
                clinical_evidence="Strong",
                safety_score=8.0
            )
        })
        
        # H.AO - Antioxidant Systems
        self.ingredients.update({
            "astaxanthin": HypergredientProperties(
                name="Astaxanthin",
                inci_name="Astaxanthin",
                hypergredient_class="H.AO",
                primary_function="antioxidant",
                secondary_functions=["anti_inflammatory"],
                efficacy_score=9.0,
                bioavailability=0.65,
                pH_min=5.0, pH_max=8.0,
                stability_conditions={"light_sensitive": True},
                synergies=["vitamin_e"],
                cost_per_gram=360.00,
                clinical_evidence="Strong",
                safety_score=9.0
            ),
            "vitamin_e": HypergredientProperties(
                name="Vitamin E",
                inci_name="Tocopherol",
                hypergredient_class="H.AO",
                primary_function="antioxidant",
                secondary_functions=["moisturizing"],
                efficacy_score=7.0,
                bioavailability=0.80,
                pH_min=4.0, pH_max=9.0,
                stability_conditions={"stable": True},
                synergies=["vitamin_c", "astaxanthin"],
                cost_per_gram=50.00,
                clinical_evidence="Strong",
                safety_score=9.0
            ),
            "resveratrol": HypergredientProperties(
                name="Resveratrol",
                inci_name="Resveratrol",
                hypergredient_class="H.AO",
                primary_function="antioxidant",
                secondary_functions=["anti_aging"],
                efficacy_score=8.0,
                bioavailability=0.60,
                pH_min=5.0, pH_max=7.0,
                stability_conditions={"moderate": True},
                synergies=["ferulic_acid"],
                cost_per_gram=190.00,
                clinical_evidence="Moderate",
                safety_score=8.0
            )
        })
        
        # H.HY - Hydration Systems
        self.ingredients.update({
            "hyaluronic_acid": HypergredientProperties(
                name="Hyaluronic Acid",
                inci_name="Sodium Hyaluronate",
                hypergredient_class="H.HY",
                primary_function="hydration",
                secondary_functions=["plumping"],
                efficacy_score=9.0,
                bioavailability=0.90,
                pH_min=4.0, pH_max=8.0,
                stability_conditions={"stable": True},
                synergies=["glycerin", "ceramides"],
                cost_per_gram=250.00,
                clinical_evidence="Strong",
                safety_score=10.0
            ),
            "glycerin": HypergredientProperties(
                name="Glycerin",
                inci_name="Glycerin",
                hypergredient_class="H.HY",
                primary_function="hydration",
                secondary_functions=["humectant"],
                efficacy_score=7.0,
                bioavailability=0.95,
                pH_min=4.0, pH_max=9.0,
                stability_conditions={"stable": True},
                synergies=["hyaluronic_acid"],
                cost_per_gram=15.00,
                clinical_evidence="Strong",
                safety_score=10.0
            )
        })
        
        # H.BR - Barrier Repair Complex
        self.ingredients.update({
            "ceramides": HypergredientProperties(
                name="Ceramides",
                inci_name="Ceramide NP",
                hypergredient_class="H.BR",
                primary_function="barrier_repair",
                secondary_functions=["moisturizing"],
                efficacy_score=8.0,
                bioavailability=0.75,
                pH_min=5.0, pH_max=8.0,
                stability_conditions={"stable": True},
                synergies=["cholesterol", "hyaluronic_acid"],
                cost_per_gram=180.00,
                clinical_evidence="Strong",
                safety_score=10.0
            ),
            "niacinamide": HypergredientProperties(
                name="Niacinamide",
                inci_name="Niacinamide",
                hypergredient_class="H.BR",
                primary_function="barrier_repair",
                secondary_functions=["sebum_regulation", "brightening"],
                efficacy_score=8.0,
                bioavailability=0.85,
                pH_min=5.0, pH_max=8.0,
                stability_conditions={"stable": True},
                synergies=["hyaluronic_acid", "ceramides"],
                cost_per_gram=75.00,
                clinical_evidence="Strong",
                safety_score=9.0
            )
        })
    
    def _initialize_interactions(self):
        """Initialize the interaction matrix for hypergredient synergies and antagonisms"""
        self.interaction_matrix = {
            ("H.CT", "H.CS"): 1.5,  # Cellular turnover + collagen synthesis = positive synergy
            ("H.CT", "H.AO"): 0.8,  # Potential oxidation issues
            ("H.CS", "H.AO"): 2.0,  # Strong synergy for anti-aging
            ("H.BR", "H.HY"): 2.5,  # Excellent barrier + hydration synergy
            ("H.AO", "H.AI"): 1.8,  # Good anti-inflammatory synergy
            ("H.CT", "H.BR"): 0.6,  # Potential irritation when barrier compromised
        }
    
    def get_by_class(self, hypergredient_class: str) -> List[HypergredientProperties]:
        """Get all ingredients in a specific hypergredient class"""
        return [ingredient for ingredient in self.ingredients.values() 
                if ingredient.hypergredient_class == hypergredient_class]
    
    def get_by_function(self, function: str) -> List[HypergredientProperties]:
        """Get ingredients by primary or secondary function"""
        results = []
        for ingredient in self.ingredients.values():
            if (ingredient.primary_function == function or 
                function in ingredient.secondary_functions):
                results.append(ingredient)
        return results
    
    def calculate_ingredient_score(self, ingredient: HypergredientProperties,
                                 objective_weights: Dict[str, float],
                                 formulation_context: List[str] = None) -> float:
        """Calculate multi-objective score for an ingredient"""
        formulation_context = formulation_context or []
        
        # Base scores
        efficacy = ingredient.efficacy_score / 10.0
        safety = ingredient.safety_score / 10.0  
        stability = 1.0 if ingredient.stability_conditions.get("stable") else 0.7
        cost_efficiency = 1.0 / (1.0 + ingredient.cost_per_gram / 100.0)  # Normalize cost
        
        # Check for synergies with existing formulation
        synergy_bonus = 0.0
        for existing_ingredient in formulation_context:
            if existing_ingredient in ingredient.synergies:
                synergy_bonus += 0.1
        
        # Check for incompatibilities  
        incompatibility_penalty = 0.0
        for existing_ingredient in formulation_context:
            if existing_ingredient in ingredient.incompatibilities:
                incompatibility_penalty -= 0.3
        
        # Calculate weighted score
        weighted_score = (
            efficacy * objective_weights.get('efficacy', 0.35) +
            safety * objective_weights.get('safety', 0.25) +
            stability * objective_weights.get('stability', 0.20) +
            cost_efficiency * objective_weights.get('cost', 0.15) +
            synergy_bonus * objective_weights.get('synergy', 0.05)
        )
        
        return max(0.0, weighted_score + incompatibility_penalty)


@dataclass
class FormulationRequest:
    """Request parameters for formulation optimization"""
    target_concerns: List[str]
    skin_type: str = "normal"
    budget: float = 1500.0  # ZAR
    preferences: List[str] = field(default_factory=list)
    exclude_ingredients: List[str] = field(default_factory=list)


@dataclass 
class OptimalFormulation:
    """Result of hypergredient optimization"""
    selected_hypergredients: Dict[str, Dict[str, Any]]
    total_score: float
    predicted_efficacy: float
    stability_months: int
    cost_per_50ml: float
    safety_profile: str
    synergy_score: float


class HypergredientFormulator:
    """Multi-objective formulation optimizer using hypergredients"""
    
    def __init__(self):
        self.database = HypergredientDatabase()
        
        # Default objective weights
        self.default_objectives = {
            'efficacy': 0.35,
            'safety': 0.25, 
            'stability': 0.20,
            'cost': 0.15,
            'synergy': 0.05
        }
        
        # Concern to hypergredient class mapping
        self.concern_mapping = {
            'wrinkles': ['H.CT', 'H.CS'],
            'fine_lines': ['H.CT', 'H.CS'],
            'firmness': ['H.CS'],
            'brightness': ['H.ML', 'H.AO'],
            'hyperpigmentation': ['H.ML'],
            'dark_spots': ['H.ML'],
            'dryness': ['H.HY', 'H.BR'],
            'dehydration': ['H.HY'],
            'sensitivity': ['H.AI', 'H.BR'],
            'acne': ['H.CT', 'H.SE'],
            'oily_skin': ['H.SE'],
            'dullness': ['H.CT', 'H.AO'],
            'anti_aging': ['H.CT', 'H.CS', 'H.AO']
        }
    
    def map_concern_to_hypergredient(self, concern: str) -> List[str]:
        """Map skin concern to relevant hypergredient classes"""
        return self.concern_mapping.get(concern.lower(), ['H.HY'])  # Default to hydration
    
    def optimize_formulation(self, request: FormulationRequest) -> OptimalFormulation:
        """Generate optimal formulation using hypergredient framework"""
        
        selected_hypergredients = {}
        selected_ingredients = []
        total_cost = 0.0
        
        # Process each concern
        for concern in request.target_concerns:
            hypergredient_classes = self.map_concern_to_hypergredient(concern)
            
            for hg_class in hypergredient_classes:
                if hg_class in selected_hypergredients:
                    continue  # Already selected for this class
                
                # Get candidates for this class
                candidates = self.database.get_by_class(hg_class)
                
                # Filter out excluded ingredients
                candidates = [c for c in candidates 
                            if c.name.lower() not in [e.lower() for e in request.exclude_ingredients]]
                
                if not candidates:
                    continue
                
                # Score each candidate
                scored_candidates = []
                for candidate in candidates:
                    score = self.database.calculate_ingredient_score(
                        candidate, 
                        self.default_objectives,
                        selected_ingredients
                    )
                    scored_candidates.append((candidate, score))
                
                # Select best candidate
                best_candidate, best_score = max(scored_candidates, key=lambda x: x[1])
                
                # Determine optimal percentage based on efficacy and safety
                percentage = self._calculate_optimal_percentage(best_candidate, request)
                
                selected_hypergredients[hg_class] = {
                    'selection': best_candidate.name,
                    'ingredient': best_candidate,
                    'percentage': percentage,
                    'score': best_score,
                    'reasoning': self._generate_reasoning(best_candidate, best_score)
                }
                
                selected_ingredients.append(best_candidate.name.lower())
                total_cost += best_candidate.cost_per_gram * (percentage / 100.0) * 0.5  # 50ml cost
        
        # Calculate overall metrics
        if selected_hypergredients:
            total_score = sum(hg['score'] for hg in selected_hypergredients.values()) / len(selected_hypergredients)
            synergy_score = self._calculate_synergy_score(selected_hypergredients)
            predicted_efficacy = self._predict_efficacy(selected_hypergredients, request.target_concerns)
            stability_months = self._estimate_stability(selected_hypergredients)
            safety_profile = self._assess_safety_profile(selected_hypergredients)
        else:
            # Fallback for empty formulation
            total_score = 0.0
            synergy_score = 5.0
            predicted_efficacy = 0.0
            stability_months = 12
            safety_profile = "Unknown - no ingredients selected"
        
        return OptimalFormulation(
            selected_hypergredients=selected_hypergredients,
            total_score=total_score,
            predicted_efficacy=predicted_efficacy,
            stability_months=stability_months,
            cost_per_50ml=total_cost,
            safety_profile=safety_profile,
            synergy_score=synergy_score
        )
    
    def _calculate_optimal_percentage(self, ingredient: HypergredientProperties, 
                                    request: FormulationRequest) -> float:
        """Calculate optimal usage percentage for an ingredient"""
        # Base percentage based on efficacy and safety balance
        base_percentage = (ingredient.efficacy_score / 10.0) * (ingredient.safety_score / 10.0) * 5.0
        
        # Adjust for preferences
        if 'gentle' in request.preferences:
            base_percentage *= 0.7
        if 'potent' in request.preferences:
            base_percentage *= 1.3
            
        # Class-specific adjustments
        class_limits = {
            'H.CT': 2.0,  # Cellular turnover actives
            'H.CS': 5.0,  # Collagen promoters
            'H.AO': 1.0,  # Antioxidants
            'H.BR': 5.0,  # Barrier repair
            'H.ML': 3.0,  # Melanin modulators
            'H.HY': 2.0,  # Hydration (concentrated)
            'H.AI': 2.0,  # Anti-inflammatory
            'H.MB': 1.0,  # Microbiome
            'H.SE': 3.0,  # Sebum regulators
            'H.PD': 1.0   # Penetration enhancers
        }
        
        max_percentage = class_limits.get(ingredient.hypergredient_class, 3.0)
        return min(base_percentage, max_percentage)
    
    def _generate_reasoning(self, ingredient: HypergredientProperties, score: float) -> str:
        """Generate reasoning for ingredient selection"""
        reasons = []
        
        if ingredient.safety_score >= 9.0:
            reasons.append("high safety profile")
        if ingredient.stability_conditions.get("stable"):
            reasons.append("excellent stability")
        if ingredient.cost_per_gram < 100:
            reasons.append("cost-effective")
        if len(ingredient.synergies) > 2:
            reasons.append("broad compatibility")
        if ingredient.bioavailability > 0.8:
            reasons.append("high bioavailability")
        
        return ", ".join(reasons) if reasons else "balanced performance profile"
    
    def _calculate_synergy_score(self, selected_hypergredients: Dict) -> float:
        """Calculate overall synergy score of the formulation"""
        if len(selected_hypergredients) < 2:
            return 5.0
        
        synergy_sum = 0.0
        interaction_count = 0
        
        classes = list(selected_hypergredients.keys())
        for i, class1 in enumerate(classes):
            for class2 in classes[i+1:]:
                interaction_key = tuple(sorted([class1, class2]))
                if interaction_key in self.database.interaction_matrix:
                    synergy_sum += self.database.interaction_matrix[interaction_key]
                    interaction_count += 1
                else:
                    synergy_sum += 1.0  # Neutral interaction
                    interaction_count += 1
        
        average_synergy = synergy_sum / interaction_count if interaction_count > 0 else 1.0
        return min(10.0, average_synergy * 5.0)  # Scale to 0-10
    
    def _predict_efficacy(self, selected_hypergredients: Dict, concerns: List[str]) -> float:
        """Predict formulation efficacy based on ingredients and synergies"""
        base_efficacy = 0.0
        for hg_data in selected_hypergredients.values():
            ingredient = hg_data['ingredient']
            base_efficacy += ingredient.efficacy_score * ingredient.bioavailability
        
        base_efficacy /= len(selected_hypergredients)
        
        # Boost for targeted concerns
        concern_boost = min(0.2, len(concerns) * 0.05)  # Up to 20% boost
        
        # Synergy boost
        synergy_boost = (self._calculate_synergy_score(selected_hypergredients) - 5.0) / 50.0
        
        final_efficacy = (base_efficacy + concern_boost + synergy_boost) / 10.0
        return min(1.0, max(0.0, final_efficacy))
    
    def _estimate_stability(self, selected_hypergredients: Dict) -> int:
        """Estimate formulation stability in months"""
        min_stability = 24  # Base stability
        
        for hg_data in selected_hypergredients.values():
            ingredient = hg_data['ingredient']
            if ingredient.stability_conditions.get("light_sensitive"):
                min_stability = min(min_stability, 12)
            elif ingredient.stability_conditions.get("oxygen_sensitive"):
                min_stability = min(min_stability, 18)
            elif not ingredient.stability_conditions.get("stable", True):
                min_stability = min(min_stability, 15)
        
        return min_stability
    
    def _assess_safety_profile(self, selected_hypergredients: Dict) -> str:
        """Assess overall safety profile of formulation"""
        avg_safety = sum(hg['ingredient'].safety_score for hg in selected_hypergredients.values()) / len(selected_hypergredients)
        
        if avg_safety >= 9.0:
            return "Excellent - suitable for sensitive skin"
        elif avg_safety >= 7.0:
            return "Good - suitable for most skin types"
        elif avg_safety >= 5.0:
            return "Moderate - patch test recommended"
        else:
            return "Caution - professional consultation advised"


def check_compatibility(ingredient_a: HypergredientProperties, 
                       ingredient_b: HypergredientProperties) -> Dict[str, Any]:
    """Real-time compatibility checker between two hypergredients"""
    
    # Check direct incompatibilities
    incompatible = (ingredient_a.name.lower() in [i.lower() for i in ingredient_b.incompatibilities] or
                   ingredient_b.name.lower() in [i.lower() for i in ingredient_a.incompatibilities])
    
    # Check synergies - use both full name and simple name matching
    synergistic = False
    
    # Create simple name mappings for lookup
    simple_name_a = ingredient_a.name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
    simple_name_b = ingredient_b.name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
    
    # Check if ingredient names or simple names are in synergy lists
    for synergy in ingredient_b.synergies:
        if (ingredient_a.name.lower() in synergy.lower() or 
            synergy.lower() in ingredient_a.name.lower() or
            simple_name_a in synergy.lower() or
            synergy.lower() in simple_name_a):
            synergistic = True
            break
    
    if not synergistic:
        for synergy in ingredient_a.synergies:
            if (ingredient_b.name.lower() in synergy.lower() or 
                synergy.lower() in ingredient_b.name.lower() or
                simple_name_b in synergy.lower() or
                synergy.lower() in simple_name_b):
                synergistic = True
                break
    
    # pH compatibility
    ph_overlap = not (ingredient_a.pH_max < ingredient_b.pH_min or ingredient_b.pH_max < ingredient_a.pH_min)
    ph_overlap_range = (max(ingredient_a.pH_min, ingredient_b.pH_min), 
                       min(ingredient_a.pH_max, ingredient_b.pH_max))
    
    # Calculate overall compatibility score
    base_score = 1.0
    if incompatible:
        base_score = 0.2
    elif synergistic:
        base_score = 1.8
    
    ph_factor = 1.0 if ph_overlap else 0.5
    
    final_score = base_score * ph_factor
    
    return {
        'score': final_score,
        'ph_overlap': ph_overlap,
        'ph_range': ph_overlap_range if ph_overlap else None,
        'synergistic': synergistic,
        'incompatible': incompatible,
        'recommendations': _generate_compatibility_recommendations(ingredient_a, ingredient_b, final_score)
    }


def _generate_compatibility_recommendations(ingredient_a: HypergredientProperties,
                                          ingredient_b: HypergredientProperties, 
                                          score: float) -> List[str]:
    """Generate recommendations based on compatibility analysis"""
    recommendations = []
    
    if score < 0.5:
        recommendations.append("❌ Avoid combining - use in separate routines")
        recommendations.append("🕐 Consider alternating days or AM/PM split")
    elif score > 1.5:
        recommendations.append("✅ Excellent combination - expect synergistic benefits")
        recommendations.append("💡 Consider increasing concentrations for enhanced effect")
    elif score < 1.0:
        recommendations.append("⚠️ Use with caution - monitor for irritation")
        recommendations.append("🧪 Consider lower concentrations initially")
    else:
        recommendations.append("✅ Compatible combination")
    
    # pH-specific recommendations
    if ingredient_a.pH_max < ingredient_b.pH_min or ingredient_b.pH_max < ingredient_a.pH_min:
        recommendations.append("🧪 pH adjustment needed for optimal stability")
    
    return recommendations


# Example usage functions
def generate_anti_aging_formulation() -> OptimalFormulation:
    """Generate example anti-aging formulation using hypergredient framework"""
    formulator = HypergredientFormulator()
    
    request = FormulationRequest(
        target_concerns=['wrinkles', 'firmness', 'brightness'],
        skin_type='normal_to_dry',
        budget=1500.0,
        preferences=['gentle', 'stable']
    )
    
    return formulator.optimize_formulation(request)


def demonstrate_compatibility_checking():
    """Demonstrate real-time compatibility checking"""
    db = HypergredientDatabase()
    
    # Test some combinations
    retinol = db.ingredients['retinol']
    vitamin_c = db.ingredients['vitamin_c_l_aa']
    bakuchiol = db.ingredients['bakuchiol']
    niacinamide = db.ingredients['niacinamide']
    
    combinations = [
        (retinol, vitamin_c, "Retinol + Vitamin C"),
        (bakuchiol, niacinamide, "Bakuchiol + Niacinamide"),
        (vitamin_c, niacinamide, "Vitamin C + Niacinamide")
    ]
    
    print("\n=== HYPERGREDIENT COMPATIBILITY ANALYSIS ===")
    for ing_a, ing_b, name in combinations:
        result = check_compatibility(ing_a, ing_b)
        print(f"\n{name}:")
        print(f"  Compatibility Score: {result['score']:.2f}/2.0")
        print(f"  pH Compatible: {result['ph_overlap']}")
        if result['ph_range']:
            print(f"  Optimal pH Range: {result['ph_range'][0]:.1f} - {result['ph_range'][1]:.1f}")
        print("  Recommendations:")
        for rec in result['recommendations']:
            print(f"    {rec}")


if __name__ == "__main__":
    print("🧬 HYPERGREDIENT FRAMEWORK DEMONSTRATION\n")
    
    # 1. Generate optimal formulation
    print("1. OPTIMAL ANTI-AGING FORMULATION")
    print("=" * 50)
    
    formulation = generate_anti_aging_formulation()
    
    print("\nSELECTED HYPERGREDIENTS:")
    for hg_class, data in formulation.selected_hypergredients.items():
        print(f"• {HYPERGREDIENT_DATABASE[hg_class]} ({hg_class}):")
        print(f"  Selection: {data['selection']} ({data['percentage']:.1f}%)")
        print(f"  Score: {data['score']:.2f}/1.0")
        print(f"  Reasoning: {data['reasoning']}")
        print()
    
    print(f"FORMULATION METRICS:")
    print(f"  Overall Score: {formulation.total_score:.2f}/1.0")
    print(f"  Synergy Score: {formulation.synergy_score:.1f}/10.0")
    print(f"  Predicted Efficacy: {formulation.predicted_efficacy*100:.0f}% improvement")
    print(f"  Stability: {formulation.stability_months} months")
    print(f"  Cost: R{formulation.cost_per_50ml:.2f}/50ml")
    print(f"  Safety Profile: {formulation.safety_profile}")
    
    # 2. Compatibility demonstrations
    demonstrate_compatibility_checking()
    
    print(f"\n🚀 Hypergredient Framework successfully demonstrated!")
    print("This system transforms formulation from art to science! 🧬")