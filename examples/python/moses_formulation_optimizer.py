#!/usr/bin/env python3
"""
MOSES-Inspired Evolutionary Formulation Optimizer

This module implements MOSES (Meta-Optimizing Semantic Evolutionary Search)
concepts adapted for cosmeceutical formulation optimization, including:

- Population-based evolutionary search for optimal formulations
- Semantic-aware crossover and mutation operations
- Multi-objective fitness evaluation
- Multiscale skin model integration
- Constraint satisfaction for regulatory compliance

Author: ONNX Runtime Cosmeceutical Optimization Team
"""

import random
import math
from typing import Dict, List, Tuple, Set, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import copy

# Import OpenCog-inspired components
from opencog_cosmeceutical_optimizer import (
    AtomSpace, CognitiveAtom, CognitiveLink, TruthValue,
    SkinLayer, DeliveryMechanism, TherapeuticVector,
    INCIParser, ECANAttentionModule, PLNReasoningEngine
)


@dataclass
class FormulationGenome:
    """Genetic representation of a cosmeceutical formulation"""
    ingredients: Dict[str, float]  # ingredient_name -> concentration (%)
    delivery_systems: Dict[str, str] = field(default_factory=dict)  # ingredient -> delivery_type
    ph_target: float = 6.0
    viscosity_target: float = 5000.0  # cP
    stability_enhancers: Set[str] = field(default_factory=set)
    
    def __post_init__(self):
        # Ensure concentrations sum to reasonable total
        total = sum(self.ingredients.values())
        if total > 100.0:
            # Normalize concentrations
            factor = 95.0 / total  # Leave 5% for water/base
            self.ingredients = {k: v * factor for k, v in self.ingredients.items()}
    
    def mutate(self, mutation_rate: float = 0.1) -> 'FormulationGenome':
        """Create mutated copy of genome"""
        new_genome = copy.deepcopy(self)
        
        for ingredient in list(new_genome.ingredients.keys()):
            if random.random() < mutation_rate:
                # Mutate concentration
                current = new_genome.ingredients[ingredient]
                mutation = random.gauss(0, current * 0.2)  # 20% std dev
                new_conc = max(0.01, min(20.0, current + mutation))
                new_genome.ingredients[ingredient] = new_conc
        
        # Possibly add new ingredient
        if random.random() < mutation_rate * 0.5:
            available_ingredients = [
                'ceramides', 'peptides', 'alpha_arbutin', 'kojic_acid',
                'azelaic_acid', 'bakuchiol', 'centella_asiatica'
            ]
            new_ingredient = random.choice(available_ingredients)
            if new_ingredient not in new_genome.ingredients:
                new_genome.ingredients[new_ingredient] = random.uniform(0.1, 2.0)
        
        # Mutate other properties
        if random.random() < mutation_rate:
            new_genome.ph_target += random.gauss(0, 0.3)
            new_genome.ph_target = max(4.5, min(8.0, new_genome.ph_target))
        
        if random.random() < mutation_rate:
            new_genome.viscosity_target *= random.uniform(0.8, 1.2)
            new_genome.viscosity_target = max(100, min(50000, new_genome.viscosity_target))
        
        return new_genome
    
    def crossover(self, other: 'FormulationGenome') -> Tuple['FormulationGenome', 'FormulationGenome']:
        """Semantic-aware crossover operation"""
        child1 = FormulationGenome({})
        child2 = FormulationGenome({})
        
        # Combine ingredients from both parents
        all_ingredients = set(self.ingredients.keys()) | set(other.ingredients.keys())
        
        for ingredient in all_ingredients:
            conc1 = self.ingredients.get(ingredient, 0)
            conc2 = other.ingredients.get(ingredient, 0)
            
            # Semantic crossover: average concentrations with some variation
            if conc1 > 0 and conc2 > 0:
                avg = (conc1 + conc2) / 2
                child1.ingredients[ingredient] = avg + random.gauss(0, avg * 0.1)
                child2.ingredients[ingredient] = avg + random.gauss(0, avg * 0.1)
            elif conc1 > 0:
                if random.random() < 0.7:  # 70% chance to inherit
                    child1.ingredients[ingredient] = conc1 * random.uniform(0.8, 1.2)
                if random.random() < 0.3:  # 30% chance for other child
                    child2.ingredients[ingredient] = conc1 * random.uniform(0.5, 1.0)
            elif conc2 > 0:
                if random.random() < 0.7:
                    child2.ingredients[ingredient] = conc2 * random.uniform(0.8, 1.2)
                if random.random() < 0.3:
                    child1.ingredients[ingredient] = conc2 * random.uniform(0.5, 1.0)
        
        # Remove very low concentrations
        child1.ingredients = {k: v for k, v in child1.ingredients.items() if v > 0.01}
        child2.ingredients = {k: v for k, v in child2.ingredients.items() if v > 0.01}
        
        # Inherit other properties
        child1.ph_target = (self.ph_target + other.ph_target) / 2
        child2.ph_target = (self.ph_target + other.ph_target) / 2
        child1.viscosity_target = math.sqrt(self.viscosity_target * other.viscosity_target)
        child2.viscosity_target = math.sqrt(self.viscosity_target * other.viscosity_target)
        
        return child1, child2


@dataclass 
class MultiscaleSkinModel:
    """Multiscale skin model for therapeutic targeting"""
    
    def __init__(self):
        self.layers = {
            SkinLayer.STRATUM_CORNEUM: {
                'thickness_um': 10,
                'ph': 5.5,
                'lipid_content': 0.15,
                'water_content': 0.15,
                'permeability': 0.1
            },
            SkinLayer.EPIDERMIS: {
                'thickness_um': 100,
                'ph': 6.8,
                'lipid_content': 0.05,
                'water_content': 0.70,
                'permeability': 0.3
            },
            SkinLayer.DERMIS_PAPILLARY: {
                'thickness_um': 200,
                'ph': 7.2,
                'lipid_content': 0.02,
                'water_content': 0.75,
                'permeability': 0.7
            },
            SkinLayer.DERMIS_RETICULAR: {
                'thickness_um': 1800,
                'ph': 7.4,
                'lipid_content': 0.01,
                'water_content': 0.70,
                'permeability': 0.9
            }
        }
        
        self.therapeutic_vectors = {
            'anti_aging': TherapeuticVector(
                name='anti_aging',
                target_condition='aging',
                mechanism_of_action='collagen_stimulation',
                optimal_concentration_range=(0.1, 2.0),
                target_layers=[SkinLayer.DERMIS_PAPILLARY, SkinLayer.DERMIS_RETICULAR],
                synergistic_ingredients=['vitamin_c', 'retinol', 'peptides']
            ),
            'barrier_repair': TherapeuticVector(
                name='barrier_repair',
                target_condition='barrier_dysfunction',
                mechanism_of_action='lipid_restoration',
                optimal_concentration_range=(1.0, 10.0),
                target_layers=[SkinLayer.STRATUM_CORNEUM, SkinLayer.EPIDERMIS],
                synergistic_ingredients=['ceramides', 'niacinamide', 'cholesterol']
            ),
            'hydration': TherapeuticVector(
                name='hydration',
                target_condition='dehydration',
                mechanism_of_action='water_retention',
                optimal_concentration_range=(0.5, 15.0),
                target_layers=[SkinLayer.STRATUM_CORNEUM, SkinLayer.EPIDERMIS],
                synergistic_ingredients=['hyaluronic_acid', 'glycerin', 'sodium_pca']
            )
        }
    
    def calculate_penetration_profile(self, ingredient: str, concentration: float,
                                    delivery_system: Optional[DeliveryMechanism] = None) -> Dict[SkinLayer, float]:
        """Calculate ingredient penetration across skin layers"""
        profile = {}
        remaining_conc = concentration
        
        for layer in [SkinLayer.STRATUM_CORNEUM, SkinLayer.EPIDERMIS, 
                     SkinLayer.DERMIS_PAPILLARY, SkinLayer.DERMIS_RETICULAR]:
            
            layer_props = self.layers[layer]
            base_penetration = layer_props['permeability']
            
            # Adjust for delivery system
            if delivery_system:
                penetration_factor = delivery_system.compatibility_score(ingredient, layer)
                base_penetration *= penetration_factor
            
            # Adjust for ingredient properties
            penetration_factor = self._get_ingredient_penetration_factor(ingredient, layer)
            actual_penetration = base_penetration * penetration_factor
            
            penetrated_amount = remaining_conc * actual_penetration
            profile[layer] = penetrated_amount
            remaining_conc -= penetrated_amount
            
            if remaining_conc <= 0.01:
                break
        
        return profile
    
    def _get_ingredient_penetration_factor(self, ingredient: str, layer: SkinLayer) -> float:
        """Get ingredient-specific penetration factor for layer"""
        # Simplified model - in practice would use molecular properties
        factors = {
            'retinol': {SkinLayer.STRATUM_CORNEUM: 0.8, SkinLayer.EPIDERMIS: 1.2},
            'niacinamide': {SkinLayer.STRATUM_CORNEUM: 1.1, SkinLayer.EPIDERMIS: 1.3},
            'hyaluronic_acid': {SkinLayer.STRATUM_CORNEUM: 1.5, SkinLayer.EPIDERMIS: 0.3},
            'vitamin_c': {SkinLayer.EPIDERMIS: 1.2, SkinLayer.DERMIS_PAPILLARY: 1.1},
        }
        
        return factors.get(ingredient, {}).get(layer, 1.0)
    
    def evaluate_therapeutic_efficacy(self, formulation: FormulationGenome) -> Dict[str, float]:
        """Evaluate therapeutic efficacy across all vectors"""
        efficacy_scores = {}
        
        for vector_name, vector in self.therapeutic_vectors.items():
            total_efficacy = 0.0
            active_ingredients = []
            
            # Find ingredients targeting this vector
            for ingredient, concentration in formulation.ingredients.items():
                if ingredient in vector.synergistic_ingredients:
                    active_ingredients.append((ingredient, concentration))
            
            if not active_ingredients:
                efficacy_scores[vector_name] = 0.0
                continue
            
            # Calculate layer-specific efficacy
            for ingredient, concentration in active_ingredients:
                penetration_profile = self.calculate_penetration_profile(ingredient, concentration)
                
                for layer, layer_conc in penetration_profile.items():
                    layer_efficacy = vector.efficacy_score(layer_conc, layer)
                    total_efficacy += layer_efficacy
            
            # Apply synergy bonuses
            if len(active_ingredients) > 1:
                synergy_bonus = min(0.5, (len(active_ingredients) - 1) * 0.2)
                total_efficacy *= (1.0 + synergy_bonus)
            
            efficacy_scores[vector_name] = min(1.0, total_efficacy / len(vector.target_layers))
        
        return efficacy_scores


@dataclass
class FitnessScore:
    """Multi-objective fitness score for formulation evaluation"""
    efficacy: float = 0.0
    stability: float = 0.0
    safety: float = 0.0
    cost: float = 0.0
    regulatory_compliance: float = 0.0
    consumer_acceptance: float = 0.0
    
    def overall_fitness(self, weights: Dict[str, float] = None) -> float:
        """Calculate weighted overall fitness"""
        if weights is None:
            weights = {
                'efficacy': 0.3,
                'stability': 0.2,
                'safety': 0.2,
                'cost': 0.1,
                'regulatory_compliance': 0.15,
                'consumer_acceptance': 0.05
            }
        
        return (
            self.efficacy * weights['efficacy'] +
            self.stability * weights['stability'] +
            self.safety * weights['safety'] +
            (1.0 - self.cost) * weights['cost'] +  # Lower cost is better
            self.regulatory_compliance * weights['regulatory_compliance'] +
            self.consumer_acceptance * weights['consumer_acceptance']
        )


class MOSESFormulationOptimizer:
    """MOSES-inspired evolutionary optimizer for cosmeceutical formulations"""
    
    def __init__(self, atomspace: AtomSpace, skin_model: MultiscaleSkinModel):
        self.atomspace = atomspace
        self.skin_model = skin_model
        self.population_size = 50
        self.elite_size = 10
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
        self.max_generations = 100
        
        # Initialize components
        self.reasoning_engine = PLNReasoningEngine(atomspace)
        self.inci_parser = INCIParser()
        
        # Regulatory constraints
        self.regulatory_limits = {
            'retinol': 0.3,
            'salicylic_acid': 2.0,
            'glycolic_acid': 10.0,
            'niacinamide': 12.0,
            'vitamin_c': 20.0,
        }
    
    def initialize_population(self, base_ingredients: List[str]) -> List[FormulationGenome]:
        """Initialize random population of formulations"""
        population = []
        
        for _ in range(self.population_size):
            # Random subset of ingredients
            num_ingredients = random.randint(3, min(8, len(base_ingredients)))
            selected = random.sample(base_ingredients, num_ingredients)
            
            # Random concentrations
            ingredients = {}
            for ingredient in selected:
                max_conc = self.regulatory_limits.get(ingredient, 10.0)
                ingredients[ingredient] = random.uniform(0.1, min(5.0, max_conc))
            
            # Random formulation properties
            genome = FormulationGenome(
                ingredients=ingredients,
                ph_target=random.uniform(5.0, 7.5),
                viscosity_target=random.uniform(1000, 20000)
            )
            
            population.append(genome)
        
        return population
    
    def evaluate_fitness(self, genome: FormulationGenome) -> FitnessScore:
        """Evaluate multi-objective fitness of formulation"""
        fitness = FitnessScore()
        
        # 1. Therapeutic Efficacy
        efficacy_scores = self.skin_model.evaluate_therapeutic_efficacy(genome)
        fitness.efficacy = sum(efficacy_scores.values()) / max(1, len(efficacy_scores))
        
        # 2. Stability Assessment
        fitness.stability = self._assess_stability(genome)
        
        # 3. Safety Evaluation
        fitness.safety = self._assess_safety(genome)
        
        # 4. Cost Estimation
        fitness.cost = self._estimate_cost(genome)
        
        # 5. Regulatory Compliance
        fitness.regulatory_compliance = self._check_regulatory_compliance(genome)
        
        # 6. Consumer Acceptance Prediction
        fitness.consumer_acceptance = self._predict_consumer_acceptance(genome)
        
        return fitness
    
    def _assess_stability(self, genome: FormulationGenome) -> float:
        """Assess formulation stability"""
        stability_score = 0.8  # Base stability
        
        # Check for incompatible combinations
        ingredients = list(genome.ingredients.keys())
        for i, ing1 in enumerate(ingredients):
            for ing2 in ingredients[i+1:]:
                compatibility = self.reasoning_engine.reason_about_compatibility(ing1, ing2)
                if compatibility.strength < 0.3:  # Incompatible
                    stability_score -= 0.2
        
        # pH compatibility
        if 5.0 <= genome.ph_target <= 7.0:
            stability_score += 0.1
        
        return max(0.0, min(1.0, stability_score))
    
    def _assess_safety(self, genome: FormulationGenome) -> float:
        """Assess formulation safety"""
        safety_score = 1.0
        
        # Check for over-concentration of actives
        active_total = sum(conc for ingredient, conc in genome.ingredients.items() 
                          if ingredient in ['retinol', 'salicylic_acid', 'glycolic_acid'])
        
        if active_total > 5.0:  # Too many actives
            safety_score -= 0.3
        
        # Check individual ingredient limits
        for ingredient, conc in genome.ingredients.items():
            limit = self.regulatory_limits.get(ingredient, 20.0)
            if conc > limit:
                safety_score -= 0.5
        
        return max(0.0, safety_score)
    
    def _estimate_cost(self, genome: FormulationGenome) -> float:
        """Estimate formulation cost (normalized 0-1, higher = more expensive)"""
        # Simplified cost model
        ingredient_costs = {
            'retinol': 50.0,  # $/kg
            'niacinamide': 15.0,
            'hyaluronic_acid': 100.0,
            'vitamin_c': 25.0,
            'glycerin': 2.0,
            'ceramides': 200.0,
            'peptides': 500.0,
        }
        
        total_cost = 0.0
        for ingredient, conc in genome.ingredients.items():
            unit_cost = ingredient_costs.get(ingredient, 10.0)
            total_cost += (conc / 100.0) * unit_cost
        
        # Normalize to 0-1 scale (assuming max reasonable cost is $50/kg formulation)
        return min(1.0, total_cost / 50.0)
    
    def _check_regulatory_compliance(self, genome: FormulationGenome) -> float:
        """Check regulatory compliance"""
        compliance_score = 1.0
        
        for ingredient, conc in genome.ingredients.items():
            limit = self.regulatory_limits.get(ingredient)
            if limit and conc > limit:
                compliance_score = 0.0  # Non-compliant
                break
        
        return compliance_score
    
    def _predict_consumer_acceptance(self, genome: FormulationGenome) -> float:
        """Predict consumer acceptance based on formulation properties"""
        acceptance = 0.5  # Base acceptance
        
        # pH preference (consumers prefer neutral)
        if 6.0 <= genome.ph_target <= 7.0:
            acceptance += 0.2
        elif genome.ph_target < 5.0 or genome.ph_target > 8.0:
            acceptance -= 0.2
        
        # Viscosity preference
        if 2000 <= genome.viscosity_target <= 10000:
            acceptance += 0.1
        
        # Ingredient complexity (simpler is often preferred)
        if len(genome.ingredients) <= 6:
            acceptance += 0.1
        elif len(genome.ingredients) > 10:
            acceptance -= 0.2
        
        return max(0.0, min(1.0, acceptance))
    
    def selection(self, population: List[FormulationGenome], 
                 fitness_scores: List[FitnessScore]) -> List[FormulationGenome]:
        """Tournament selection for breeding"""
        selected = []
        tournament_size = 3
        
        # Keep elite
        elite_indices = sorted(range(len(fitness_scores)), 
                             key=lambda i: fitness_scores[i].overall_fitness(), 
                             reverse=True)[:self.elite_size]
        selected.extend([population[i] for i in elite_indices])
        
        # Tournament selection for rest
        while len(selected) < self.population_size:
            tournament_indices = random.sample(range(len(population)), tournament_size)
            tournament_fitness = [fitness_scores[i].overall_fitness() for i in tournament_indices]
            winner_idx = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
            selected.append(population[winner_idx])
        
        return selected
    
    def optimize(self, base_ingredients: List[str], target_vectors: List[str] = None) -> Tuple[FormulationGenome, FitnessScore]:
        """Run MOSES-inspired evolutionary optimization"""
        print(f"Starting optimization with {len(base_ingredients)} base ingredients...")
        
        # Initialize population
        population = self.initialize_population(base_ingredients)
        best_genome = None
        best_fitness = None
        
        for generation in range(self.max_generations):
            # Evaluate fitness
            fitness_scores = [self.evaluate_fitness(genome) for genome in population]
            
            # Track best
            current_best_idx = max(range(len(fitness_scores)), 
                                 key=lambda i: fitness_scores[i].overall_fitness())
            
            if best_fitness is None or fitness_scores[current_best_idx].overall_fitness() > best_fitness.overall_fitness():
                best_genome = copy.deepcopy(population[current_best_idx])
                best_fitness = fitness_scores[current_best_idx]
            
            if generation % 10 == 0:
                avg_fitness = sum(f.overall_fitness() for f in fitness_scores) / len(fitness_scores)
                print(f"Generation {generation}: Best={best_fitness.overall_fitness():.3f}, Avg={avg_fitness:.3f}")
            
            # Selection
            selected = self.selection(population, fitness_scores)
            
            # Create next generation
            next_population = []
            
            # Keep elite
            next_population.extend(selected[:self.elite_size])
            
            # Crossover and mutation
            while len(next_population) < self.population_size:
                if random.random() < self.crossover_rate and len(selected) >= 2:
                    parent1, parent2 = random.sample(selected, 2)
                    child1, child2 = parent1.crossover(parent2)
                    
                    if random.random() < self.mutation_rate:
                        child1 = child1.mutate(self.mutation_rate)
                    if random.random() < self.mutation_rate:
                        child2 = child2.mutate(self.mutation_rate)
                    
                    next_population.extend([child1, child2])
                else:
                    # Mutation only
                    parent = random.choice(selected)
                    child = parent.mutate(self.mutation_rate)
                    next_population.append(child)
            
            population = next_population[:self.population_size]
        
        return best_genome, best_fitness


if __name__ == "__main__":
    print("=== MOSES-Inspired Formulation Optimizer ===\n")
    
    # Initialize system
    atomspace = AtomSpace()
    skin_model = MultiscaleSkinModel()
    optimizer = MOSESFormulationOptimizer(atomspace, skin_model)
    
    # Define available ingredients
    base_ingredients = [
        'retinol', 'niacinamide', 'hyaluronic_acid', 'vitamin_c', 'vitamin_e',
        'glycerin', 'ceramides', 'peptides', 'alpha_arbutin', 'azelaic_acid'
    ]
    
    print("Available ingredients:", base_ingredients)
    print(f"Population size: {optimizer.population_size}")
    print(f"Max generations: {optimizer.max_generations}\n")
    
    # Run optimization
    best_formulation, best_fitness = optimizer.optimize(base_ingredients)
    
    print("\n=== Optimization Results ===")
    print(f"Best Overall Fitness: {best_fitness.overall_fitness():.3f}")
    print(f"  Efficacy: {best_fitness.efficacy:.3f}")
    print(f"  Stability: {best_fitness.stability:.3f}")
    print(f"  Safety: {best_fitness.safety:.3f}")
    print(f"  Cost: {best_fitness.cost:.3f}")
    print(f"  Regulatory: {best_fitness.regulatory_compliance:.3f}")
    print(f"  Consumer: {best_fitness.consumer_acceptance:.3f}")
    
    print(f"\nOptimal Formulation:")
    print(f"  pH Target: {best_formulation.ph_target:.1f}")
    print(f"  Viscosity Target: {best_formulation.viscosity_target:.0f} cP")
    print(f"  Ingredients:")
    for ingredient, conc in sorted(best_formulation.ingredients.items(), key=lambda x: x[1], reverse=True):
        print(f"    {ingredient}: {conc:.2f}%")
    
    # Analyze therapeutic efficacy
    print(f"\nTherapeutic Efficacy Analysis:")
    efficacy_scores = skin_model.evaluate_therapeutic_efficacy(best_formulation)
    for vector, score in efficacy_scores.items():
        print(f"  {vector}: {score:.3f}")
    
    print("\n=== MOSES Optimization Complete ===")