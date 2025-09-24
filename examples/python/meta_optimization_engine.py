#!/usr/bin/env python3
"""
🎯 Meta-Optimization Engine for Universal Formulation Generation

This module implements a comprehensive meta-optimization strategy that generates
optimal formulations for every possible condition and treatment combination.
It pre-computes optimal solutions across the entire space of skin concerns,
types, conditions, and treatment goals.

Key Features:
- Universal condition/treatment taxonomy
- Multi-dimensional optimization space mapping  
- Pre-computed formulation matrices for instant retrieval
- Dynamic optimization updates based on new data
- Performance benchmarking and validation

Author: ONNX Runtime Cosmeceutical Optimization Team
"""

import json
import math
import time
import itertools
import concurrent.futures
from typing import Dict, List, Tuple, Set, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, OrderedDict
import copy

# Import existing frameworks
from hypergredient_framework import *
from hypergredient_advanced import *
from moses_formulation_optimizer import *


@dataclass
class ConditionProfile:
    """Comprehensive profile for a skin condition"""
    condition_id: str
    name: str
    category: str  # acne, aging, pigmentation, sensitivity, barrier, etc.
    severity_levels: List[str] = field(default_factory=lambda: ["mild", "moderate", "severe"])
    target_hypergredient_classes: List[str] = field(default_factory=list)
    contraindicated_classes: List[str] = field(default_factory=list)
    priority_functions: List[str] = field(default_factory=list)
    typical_duration_weeks: int = 12
    clinical_markers: List[str] = field(default_factory=list)


@dataclass
class TreatmentProtocol:
    """Treatment protocol specification"""
    protocol_id: str
    name: str
    target_conditions: List[str]
    treatment_phases: List[Dict[str, Any]] = field(default_factory=list)
    contraindications: List[str] = field(default_factory=list)
    expected_timeline_weeks: int = 12
    maintenance_required: bool = True


@dataclass
class OptimizationSpace:
    """Defines the complete optimization space dimensions"""
    conditions: List[ConditionProfile]
    treatments: List[TreatmentProtocol] 
    skin_types: List[str]
    age_ranges: List[str]
    sensitivities: List[str]
    budgets: List[float]
    preferences: List[List[str]]


@dataclass
class PrecomputedFormulation:
    """Pre-computed optimal formulation with metadata"""
    formulation: OptimalFormulation
    optimization_key: str
    conditions_addressed: List[str]
    confidence_score: float
    computation_time: float
    last_updated: float
    validation_score: float = 0.0


class MetaOptimizationEngine:
    """
    Universal meta-optimization engine that pre-computes optimal formulations
    for every possible combination of conditions and treatments
    """
    
    def __init__(self):
        self.formulator = HypergredientFormulator()
        self.ai_engine = HypergredientAI()
        self.evolution_engine = None
        
        # Initialize optimization space
        self.optimization_space = self._initialize_optimization_space()
        
        # Pre-computed formulation cache
        self.formulation_matrix: Dict[str, PrecomputedFormulation] = {}
        
        # Performance tracking
        self.optimization_stats = {
            'total_combinations': 0,
            'formulations_computed': 0,
            'average_computation_time': 0.0,
            'cache_hit_rate': 0.0,
            'last_full_optimization': None
        }
        
        # Optimization parameters
        self.max_concurrent_optimizations = 4
        self.confidence_threshold = 0.7
        self.recomputation_interval_hours = 24
    
    def _initialize_optimization_space(self) -> OptimizationSpace:
        """Initialize the complete optimization space"""
        
        # Define all possible skin conditions
        conditions = [
            ConditionProfile(
                condition_id="acne_comedonal",
                name="Comedonal Acne",
                category="acne",
                target_hypergredient_classes=["H.SE", "H.CT", "H.AI"],
                contraindicated_classes=["H.BR"],  # May clog pores
                priority_functions=["sebum_regulation", "cellular_turnover", "anti_inflammatory"],
                clinical_markers=["blackheads", "whiteheads", "enlarged_pores"]
            ),
            ConditionProfile(
                condition_id="acne_inflammatory", 
                name="Inflammatory Acne",
                category="acne",
                target_hypergredient_classes=["H.AI", "H.SE", "H.MB"],
                priority_functions=["anti_inflammatory", "microbiome_balance", "sebum_regulation"],
                clinical_markers=["papules", "pustules", "inflammation"]
            ),
            ConditionProfile(
                condition_id="aging_photoaging",
                name="Photoaging",
                category="aging", 
                target_hypergredient_classes=["H.AO", "H.CS", "H.CT"],
                priority_functions=["antioxidant", "collagen_synthesis", "cellular_turnover"],
                clinical_markers=["wrinkles", "age_spots", "texture_changes"]
            ),
            ConditionProfile(
                condition_id="aging_chronological",
                name="Chronological Aging",
                category="aging",
                target_hypergredient_classes=["H.CS", "H.HY", "H.BR"],
                priority_functions=["collagen_synthesis", "hydration", "barrier_repair"],
                clinical_markers=["fine_lines", "loss_of_firmness", "dryness"]
            ),
            ConditionProfile(
                condition_id="hyperpigmentation_melasma",
                name="Melasma",
                category="pigmentation",
                target_hypergredient_classes=["H.ML", "H.CT", "H.AO"],
                contraindicated_classes=["H.AI"],  # Some may worsen melasma
                priority_functions=["melanin_modulation", "cellular_turnover", "antioxidant"],
                clinical_markers=["symmetric_patches", "hormonal_triggers"]
            ),
            ConditionProfile(
                condition_id="hyperpigmentation_pih",
                name="Post-Inflammatory Hyperpigmentation",
                category="pigmentation",
                target_hypergredient_classes=["H.ML", "H.CT", "H.AI"],
                priority_functions=["melanin_modulation", "cellular_turnover", "anti_inflammatory"],
                clinical_markers=["dark_spots", "uneven_tone"]
            ),
            ConditionProfile(
                condition_id="barrier_disruption",
                name="Barrier Dysfunction",
                category="barrier",
                target_hypergredient_classes=["H.BR", "H.HY", "H.AI"],
                contraindicated_classes=["H.CT"],  # May further disrupt barrier
                priority_functions=["barrier_repair", "hydration", "anti_inflammatory"],
                clinical_markers=["dryness", "sensitivity", "irritation"]
            ),
            ConditionProfile(
                condition_id="rosacea_papulopustular",
                name="Papulopustular Rosacea",
                category="sensitivity",
                target_hypergredient_classes=["H.AI", "H.BR", "H.MB"],
                contraindicated_classes=["H.CT", "H.AO"],  # May trigger flares
                priority_functions=["anti_inflammatory", "barrier_repair", "microbiome_balance"],
                clinical_markers=["persistent_redness", "papules", "pustules"]
            ),
            ConditionProfile(
                condition_id="dehydration",
                name="Transepidermal Water Loss",
                category="hydration",
                target_hypergredient_classes=["H.HY", "H.BR", "H.PD"],
                priority_functions=["hydration", "barrier_repair", "penetration_enhancement"],
                clinical_markers=["tightness", "flaking", "dullness"]
            ),
            ConditionProfile(
                condition_id="sensitivity_general",
                name="General Sensitivity",
                category="sensitivity",
                target_hypergredient_classes=["H.AI", "H.BR"],
                contraindicated_classes=["H.CT", "H.SE"],
                priority_functions=["anti_inflammatory", "barrier_repair"],
                clinical_markers=["reactivity", "stinging", "burning"]
            )
        ]
        
        # Define treatment protocols
        treatments = [
            TreatmentProtocol(
                protocol_id="maintenance_daily",
                name="Daily Maintenance Protocol",
                target_conditions=["aging_chronological", "barrier_disruption"],
                treatment_phases=[{"phase": "maintenance", "duration_weeks": 52}],
                expected_timeline_weeks=52
            ),
            TreatmentProtocol(
                protocol_id="corrective_intensive", 
                name="Intensive Corrective Protocol",
                target_conditions=["hyperpigmentation_melasma", "aging_photoaging"],
                treatment_phases=[
                    {"phase": "preparation", "duration_weeks": 2},
                    {"phase": "active", "duration_weeks": 12},
                    {"phase": "maintenance", "duration_weeks": 12}
                ],
                expected_timeline_weeks=26
            ),
            TreatmentProtocol(
                protocol_id="acne_clearing",
                name="Acne Clearing Protocol", 
                target_conditions=["acne_comedonal", "acne_inflammatory"],
                treatment_phases=[
                    {"phase": "clearing", "duration_weeks": 8},
                    {"phase": "maintenance", "duration_weeks": 16}
                ],
                expected_timeline_weeks=24
            ),
            TreatmentProtocol(
                protocol_id="sensitivity_soothing",
                name="Sensitivity Soothing Protocol",
                target_conditions=["rosacea_papulopustular", "sensitivity_general"],
                treatment_phases=[
                    {"phase": "calming", "duration_weeks": 4},
                    {"phase": "strengthening", "duration_weeks": 8}
                ],
                expected_timeline_weeks=12
            ),
            TreatmentProtocol(
                protocol_id="barrier_restoration",
                name="Barrier Restoration Protocol",
                target_conditions=["barrier_disruption", "dehydration"],
                treatment_phases=[
                    {"phase": "repair", "duration_weeks": 6},
                    {"phase": "strengthen", "duration_weeks": 6}
                ],
                expected_timeline_weeks=12
            )
        ]
        
        return OptimizationSpace(
            conditions=conditions,
            treatments=treatments,
            skin_types=["oily", "dry", "combination", "normal", "sensitive"],
            age_ranges=["teens", "twenties", "thirties", "forties", "fifties_plus"],
            sensitivities=["none", "mild", "moderate", "severe"],
            budgets=[500.0, 1000.0, 1500.0, 2500.0, 5000.0],
            preferences=[
                ["gentle"],
                ["effective"],
                ["stable"],
                ["gentle", "stable"],
                ["effective", "stable"],
                ["gentle", "effective"],
                ["gentle", "effective", "stable"]
            ]
        )
    
    def generate_universal_formulation_matrix(self) -> Dict[str, PrecomputedFormulation]:
        """
        Generate optimal formulations for every possible combination of 
        conditions, treatments, and parameters
        """
        print("🎯 Starting Universal Meta-Optimization...")
        start_time = time.time()
        
        # Calculate total combinations
        total_combinations = self._calculate_total_combinations()
        self.optimization_stats['total_combinations'] = total_combinations
        
        print(f"📊 Optimization Space: {total_combinations:,} total combinations")
        print(f"🏭 Using {self.max_concurrent_optimizations} concurrent workers")
        
        # Generate all combination keys
        combination_keys = self._generate_all_combination_keys()
        
        # Process combinations in batches with concurrent optimization
        batch_size = 100
        total_batches = math.ceil(len(combination_keys) / batch_size)
        
        formulations_computed = 0
        total_computation_time = 0.0
        
        for batch_idx in range(total_batches):
            batch_start = batch_idx * batch_size  
            batch_end = min((batch_idx + 1) * batch_size, len(combination_keys))
            batch_keys = combination_keys[batch_start:batch_end]
            
            print(f"🔄 Processing batch {batch_idx + 1}/{total_batches} ({len(batch_keys)} combinations)")
            
            # Process batch with concurrent futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_concurrent_optimizations) as executor:
                future_to_key = {
                    executor.submit(self._optimize_single_combination, key): key 
                    for key in batch_keys
                }
                
                for future in concurrent.futures.as_completed(future_to_key):
                    key = future_to_key[future]
                    try:
                        precomputed_formulation = future.result()
                        if precomputed_formulation:
                            self.formulation_matrix[key] = precomputed_formulation
                            formulations_computed += 1
                            total_computation_time += precomputed_formulation.computation_time
                    except Exception as e:
                        print(f"⚠️ Error optimizing {key}: {e}")
            
            # Progress update
            progress = ((batch_idx + 1) / total_batches) * 100
            print(f"📈 Progress: {progress:.1f}% ({formulations_computed:,} formulations computed)")
        
        # Update statistics
        self.optimization_stats.update({
            'formulations_computed': formulations_computed,
            'average_computation_time': total_computation_time / formulations_computed if formulations_computed > 0 else 0.0,
            'last_full_optimization': time.time()
        })
        
        total_time = time.time() - start_time
        print(f"✅ Universal Meta-Optimization Complete!")
        print(f"⏱️ Total Time: {total_time:.1f}s")
        print(f"🎯 Formulations Generated: {formulations_computed:,}")
        print(f"💾 Cache Size: {len(self.formulation_matrix):,} entries")
        
        return self.formulation_matrix
    
    def _calculate_total_combinations(self) -> int:
        """Calculate total number of possible combinations"""
        space = self.optimization_space
        
        # Combinations = conditions × treatments × skin_types × age_ranges × sensitivities × budgets × preferences
        # We'll also consider condition combinations (pairs, triples, etc.)
        
        condition_combinations = 0
        for r in range(1, min(4, len(space.conditions) + 1)):  # Up to 3 conditions together
            condition_combinations += math.comb(len(space.conditions), r)
        
        total = (condition_combinations * 
                len(space.treatments) *
                len(space.skin_types) * 
                len(space.age_ranges) *
                len(space.sensitivities) *
                len(space.budgets) *
                len(space.preferences))
        
        # Update optimization stats when calculating
        self.optimization_stats['total_combinations'] = total
        return total
    
    def _generate_all_combination_keys(self) -> List[str]:
        """Generate all possible combination keys for optimization"""
        keys = []
        space = self.optimization_space
        
        # Generate condition combinations (1-3 conditions)
        condition_combinations = []
        for r in range(1, min(4, len(space.conditions) + 1)):
            for combo in itertools.combinations(space.conditions, r):
                condition_combinations.append([c.condition_id for c in combo])
        
        # Generate all parameter combinations
        for conditions in condition_combinations:
            for treatment in space.treatments:
                for skin_type in space.skin_types:
                    for age_range in space.age_ranges:
                        for sensitivity in space.sensitivities:
                            for budget in space.budgets:
                                for preferences in space.preferences:
                                    key = self._generate_combination_key(
                                        conditions, treatment.protocol_id, skin_type,
                                        age_range, sensitivity, budget, preferences
                                    )
                                    keys.append(key)
        
        return keys
    
    def _generate_combination_key(self, conditions: List[str], treatment_id: str,
                                 skin_type: str, age_range: str, sensitivity: str,
                                 budget: float, preferences: List[str]) -> str:
        """Generate unique key for parameter combination"""
        return f"{'+'.join(sorted(conditions))}|{treatment_id}|{skin_type}|{age_range}|{sensitivity}|{int(budget)}|{'+'.join(sorted(preferences))}"
    
    def _optimize_single_combination(self, combination_key: str) -> Optional[PrecomputedFormulation]:
        """Optimize a single parameter combination"""
        start_time = time.time()
        
        try:
            # Parse combination key
            parts = combination_key.split('|')
            conditions = parts[0].split('+')
            treatment_id = parts[1]
            skin_type = parts[2]
            age_range = parts[3]
            sensitivity = parts[4]
            budget = float(parts[5])
            preferences = parts[6].split('+') if parts[6] else []
            
            # Convert conditions to target concerns for formulation request
            target_concerns = self._conditions_to_concerns(conditions)
            
            # Adjust parameters based on age, sensitivity, etc.
            adjusted_budget = self._adjust_budget_for_demographics(budget, age_range, sensitivity)
            adjusted_preferences = self._adjust_preferences(preferences, sensitivity, age_range)
            
            # Create formulation request
            request = FormulationRequest(
                target_concerns=target_concerns,
                skin_type=skin_type,
                budget=adjusted_budget,
                preferences=adjusted_preferences,
                exclude_ingredients=self._get_contraindicated_ingredients(conditions, sensitivity)
            )
            
            # Optimize formulation
            formulation = self.formulator.optimize_formulation(request)
            
            # Calculate confidence score
            confidence = self._calculate_confidence_score(formulation, conditions, treatment_id)
            
            computation_time = time.time() - start_time
            
            return PrecomputedFormulation(
                formulation=formulation,
                optimization_key=combination_key,
                conditions_addressed=conditions,
                confidence_score=confidence,
                computation_time=computation_time,
                last_updated=time.time()
            )
            
        except Exception as e:
            # Log error but don't fail entire optimization
            return None
    
    def _conditions_to_concerns(self, condition_ids: List[str]) -> List[str]:
        """Convert condition IDs to formulation concerns"""
        condition_map = {
            "acne_comedonal": ["acne", "blackheads"],
            "acne_inflammatory": ["acne", "inflammation"],
            "aging_photoaging": ["wrinkles", "age_spots", "firmness"],
            "aging_chronological": ["wrinkles", "firmness", "dryness"],
            "hyperpigmentation_melasma": ["melasma", "hyperpigmentation"],
            "hyperpigmentation_pih": ["dark_spots", "uneven_tone"],
            "barrier_disruption": ["dryness", "sensitivity"],
            "rosacea_papulopustular": ["redness", "sensitivity"],
            "dehydration": ["dryness", "hydration"],
            "sensitivity_general": ["sensitivity", "irritation"]
        }
        
        concerns = []
        for condition_id in condition_ids:
            concerns.extend(condition_map.get(condition_id, [condition_id]))
        
        return list(set(concerns))  # Remove duplicates
    
    def _adjust_budget_for_demographics(self, base_budget: float, age_range: str, sensitivity: str) -> float:
        """Adjust budget based on demographic factors"""
        budget = base_budget
        
        # Age-based adjustments
        age_multipliers = {
            "teens": 0.7,
            "twenties": 0.9,
            "thirties": 1.0,
            "forties": 1.2,
            "fifties_plus": 1.4
        }
        budget *= age_multipliers.get(age_range, 1.0)
        
        # Sensitivity-based adjustments (sensitive skin often requires premium ingredients)
        sensitivity_multipliers = {
            "none": 1.0,
            "mild": 1.1,
            "moderate": 1.3,
            "severe": 1.5
        }
        budget *= sensitivity_multipliers.get(sensitivity, 1.0)
        
        return budget
    
    def _adjust_preferences(self, base_preferences: List[str], sensitivity: str, age_range: str) -> List[str]:
        """Adjust preferences based on demographics"""
        preferences = base_preferences.copy()
        
        # Always add gentle for sensitive skin
        if sensitivity in ["moderate", "severe"] and "gentle" not in preferences:
            preferences.append("gentle")
        
        # Prefer stability for mature skin
        if age_range in ["forties", "fifties_plus"] and "stable" not in preferences:
            preferences.append("stable")
        
        return preferences
    
    def _get_contraindicated_ingredients(self, condition_ids: List[str], sensitivity: str) -> List[str]:
        """Get list of contraindicated ingredients for conditions"""
        contraindicated = []
        
        # Find conditions and their contraindications
        for condition_id in condition_ids:
            condition = next((c for c in self.optimization_space.conditions if c.condition_id == condition_id), None)
            if condition:
                # Convert hypergredient classes to specific ingredients that should be avoided
                for hg_class in condition.contraindicated_classes:
                    class_ingredients = self.formulator.database.get_by_class(hg_class)
                    contraindicated.extend([ing.name.lower().replace(' ', '_') for ing in class_ingredients])
        
        # Add sensitivity-based contraindications
        if sensitivity in ["moderate", "severe"]:
            contraindicated.extend(["tretinoin", "glycolic_acid", "salicylic_acid"])
        
        return list(set(contraindicated))
    
    def _calculate_confidence_score(self, formulation: OptimalFormulation, 
                                   conditions: List[str], treatment_id: str) -> float:
        """Calculate confidence score for the formulation"""
        base_confidence = formulation.total_score
        
        # Boost confidence if formulation addresses multiple conditions effectively
        if len(conditions) > 1:
            base_confidence *= 1.1
        
        # Adjust based on formulation quality metrics
        if formulation.predicted_efficacy > 0.8:
            base_confidence *= 1.2
        if formulation.safety_profile.startswith("Excellent"):
            base_confidence *= 1.1
        
        return min(1.0, base_confidence)
    
    def get_optimal_formulation(self, conditions: List[str], treatment_protocol: str = None,
                               skin_type: str = "normal", age_range: str = "thirties",
                               sensitivity: str = "none", budget: float = 1500.0,
                               preferences: List[str] = None) -> Optional[PrecomputedFormulation]:
        """
        Instantly retrieve optimal formulation for given parameters
        """
        if preferences is None:
            preferences = ["stable"]
        
        # Generate key for lookup
        treatment_id = treatment_protocol or "maintenance_daily"
        key = self._generate_combination_key(
            conditions, treatment_id, skin_type, age_range, 
            sensitivity, budget, preferences
        )
        
        # Try exact match first
        if key in self.formulation_matrix:
            self.optimization_stats['cache_hit_rate'] += 1
            return self.formulation_matrix[key]
        
        # Try fuzzy matching for similar combinations
        similar_formulation = self._find_similar_formulation(key)
        if similar_formulation:
            return similar_formulation
        
        # If no pre-computed formulation exists, compute on-demand
        print(f"🔄 Computing formulation on-demand for: {key}")
        computed_formulation = self._optimize_single_combination(key)
        
        # Cache the result for future use
        if computed_formulation:
            self.formulation_matrix[key] = computed_formulation
            self.optimization_stats['formulations_computed'] += 1
        
        return computed_formulation
    
    def _find_similar_formulation(self, target_key: str) -> Optional[PrecomputedFormulation]:
        """Find most similar pre-computed formulation"""
        target_parts = target_key.split('|')
        best_match = None
        best_similarity = 0.0
        
        for key, formulation in self.formulation_matrix.items():
            similarity = self._calculate_key_similarity(target_key, key)
            if similarity > best_similarity and similarity >= 0.8:  # 80% similarity threshold
                best_similarity = similarity
                best_match = formulation
        
        return best_match
    
    def _calculate_key_similarity(self, key1: str, key2: str) -> float:
        """Calculate similarity between two combination keys"""
        parts1 = key1.split('|')
        parts2 = key2.split('|')
        
        matches = 0
        total_parts = len(parts1)
        
        for i, (p1, p2) in enumerate(zip(parts1, parts2)):
            if i == 0:  # Conditions - check overlap
                conds1 = set(p1.split('+'))
                conds2 = set(p2.split('+'))
                overlap = len(conds1.intersection(conds2))
                total_conds = len(conds1.union(conds2))
                matches += overlap / total_conds if total_conds > 0 else 0
            elif i == 5:  # Budget - allow 20% variance
                budget1, budget2 = float(p1), float(p2)
                if abs(budget1 - budget2) / max(budget1, budget2) <= 0.2:
                    matches += 1
            else:  # Exact match for other parts
                if p1 == p2:
                    matches += 1
        
        return matches / total_parts
    
    def benchmark_performance(self) -> Dict[str, Any]:
        """Benchmark the meta-optimization engine performance"""
        print("🏃‍♂️ Running Performance Benchmarks...")
        
        # Ensure we have some formulations in the matrix for testing
        if len(self.formulation_matrix) == 0:
            print("  No cached formulations found, generating test data...")
            test_conditions = ["acne_comedonal", "aging_photoaging", "dehydration"]
            for condition in test_conditions:
                formulation = self.get_optimal_formulation([condition])
                # The get_optimal_formulation method should add it to the matrix if successful
        
        # Test retrieval speed
        test_keys = list(self.formulation_matrix.keys())[:min(100, len(self.formulation_matrix))]  # Test available formulations
        
        if len(test_keys) == 0:
            return {
                'cache_size': 0,
                'average_retrieval_time_ms': 0.0,
                'max_retrieval_time_ms': 0.0,
                'average_on_demand_time_ms': 0.0,
                'speedup_factor': 1.0,
                'memory_efficiency_mb': 0.0,
                'optimization_stats': self.optimization_stats,
                'note': 'No formulations available for benchmarking'
            }
        
        retrieval_times = []
        for key in test_keys:
            start_time = time.time()
            formulation = self.formulation_matrix.get(key)
            retrieval_time = time.time() - start_time
            retrieval_times.append(retrieval_time)
        
        # Test on-demand optimization speed
        on_demand_times = []
        test_conditions = [["acne_comedonal"], ["aging_photoaging"], ["hyperpigmentation_pih"]]
        
        for conditions in test_conditions:
            start_time = time.time()
            formulation = self.get_optimal_formulation(conditions)
            on_demand_time = time.time() - start_time
            on_demand_times.append(on_demand_time)
        
        avg_retrieval = sum(retrieval_times) / len(retrieval_times) if retrieval_times else 0.0
        avg_on_demand = sum(on_demand_times) / len(on_demand_times) if on_demand_times else 0.0
        
        return {
            'cache_size': len(self.formulation_matrix),
            'average_retrieval_time_ms': avg_retrieval * 1000,
            'max_retrieval_time_ms': max(retrieval_times) * 1000 if retrieval_times else 0.0,
            'average_on_demand_time_ms': avg_on_demand * 1000,
            'speedup_factor': avg_on_demand / avg_retrieval if avg_retrieval > 0 else 1.0,
            'memory_efficiency_mb': len(str(self.formulation_matrix).encode('utf-8')) / (1024 * 1024),
            'optimization_stats': self.optimization_stats
        }
    
    def export_formulation_matrix(self, filepath: str) -> None:
        """Export the complete formulation matrix to file"""
        export_data = {
            'metadata': {
                'export_time': time.time(),
                'total_formulations': len(self.formulation_matrix),
                'optimization_stats': self.optimization_stats
            },
            'formulations': {}
        }
        
        # Convert formulations to serializable format
        for key, precomputed in self.formulation_matrix.items():
            # Convert selected_hypergredients to serializable format
            serializable_hypergredients = {}
            for hg_class, data in precomputed.formulation.selected_hypergredients.items():
                if isinstance(data, dict):
                    # If data is already a dict, extract serializable fields
                    ingredient = data.get('ingredient')
                    if hasattr(ingredient, 'name'):
                        ingredient_name = ingredient.name
                        ingredient_class = ingredient.hypergredient_class
                    else:
                        ingredient_name = str(ingredient)
                        ingredient_class = hg_class
                    
                    serializable_hypergredients[hg_class] = {
                        'ingredient_name': ingredient_name,
                        'ingredient_class': ingredient_class,
                        'percentage': data.get('percentage', 0.0),
                        'reasoning': data.get('reasoning', '')
                    }
                else:
                    # Handle other data formats
                    serializable_hypergredients[hg_class] = {
                        'ingredient_name': str(data),
                        'ingredient_class': hg_class,
                        'percentage': 0.0,
                        'reasoning': ''
                    }
            
            export_data['formulations'][key] = {
                'optimization_key': precomputed.optimization_key,
                'conditions_addressed': precomputed.conditions_addressed,
                'confidence_score': precomputed.confidence_score,
                'computation_time': precomputed.computation_time,
                'last_updated': precomputed.last_updated,
                'formulation': {
                    'selected_hypergredients': serializable_hypergredients,
                    'total_score': precomputed.formulation.total_score,
                    'predicted_efficacy': precomputed.formulation.predicted_efficacy,
                    'stability_months': precomputed.formulation.stability_months,
                    'cost_per_50ml': precomputed.formulation.cost_per_50ml,
                    'safety_profile': precomputed.formulation.safety_profile,
                    'synergy_score': precomputed.formulation.synergy_score
                }
            }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"💾 Formulation matrix exported to {filepath}")


# Example usage and demonstration functions
def demonstrate_meta_optimization():
    """Demonstrate the meta-optimization engine capabilities"""
    print("🎯 Meta-Optimization Engine Demonstration")
    print("=" * 50)
    
    # Initialize engine
    engine = MetaOptimizationEngine()
    
    # Show optimization space
    space = engine.optimization_space
    print(f"📊 Optimization Space:")
    print(f"  Conditions: {len(space.conditions)}")
    print(f"  Treatments: {len(space.treatments)}")  
    print(f"  Total Combinations: {engine._calculate_total_combinations():,}")
    
    # Generate subset of formulations for demonstration
    print(f"\n🔄 Generating sample formulations...")
    
    # Test specific formulation retrievals
    test_cases = [
        {
            'conditions': ['acne_comedonal'],
            'treatment': 'acne_clearing',
            'skin_type': 'oily',
            'description': 'Oily skin with comedonal acne'
        },
        {
            'conditions': ['aging_photoaging', 'hyperpigmentation_pih'],
            'treatment': 'corrective_intensive', 
            'skin_type': 'normal',
            'age_range': 'forties',
            'description': 'Mature skin with photoaging and PIH'
        },
        {
            'conditions': ['sensitivity_general'],
            'treatment': 'sensitivity_soothing',
            'skin_type': 'sensitive',
            'sensitivity': 'moderate',
            'description': 'Sensitive skin needing gentle care'
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 Test Case {i}: {test_case['description']}")
        
        formulation = engine.get_optimal_formulation(
            conditions=test_case['conditions'],
            treatment_protocol=test_case['treatment'],
            skin_type=test_case['skin_type'],
            age_range=test_case.get('age_range', 'thirties'),
            sensitivity=test_case.get('sensitivity', 'none')
        )
        
        if formulation:
            print(f"  ✅ Optimal formulation generated")
            print(f"  💯 Confidence: {formulation.confidence_score:.1%}")
            print(f"  🧬 Ingredients: {len(formulation.formulation.selected_hypergredients)}")
            print(f"  💰 Cost: R{formulation.formulation.cost_per_50ml:.2f}")
            print(f"  ⚡ Efficacy: {formulation.formulation.predicted_efficacy:.1%}")
        else:
            print(f"  ❌ Failed to generate formulation")


if __name__ == "__main__":
    demonstrate_meta_optimization()