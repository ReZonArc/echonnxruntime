#!/usr/bin/env python3
"""
Complete OpenCog Cosmeceutical Optimization Workflow Demo

This demonstration showcases the full integration of OpenCog-inspired
cognitive architecture components for sophisticated cosmeceutical formulation
optimization, including all deliverables from the problem statement:

1. Literature review of OpenCog features for constraint optimization
2. Prototypical implementation for skin model integration
3. Test cases for INCI-based search space pruning and optimization accuracy

Author: ONNX Runtime Cosmeceutical Optimization Team
"""

import json
import time
from typing import Dict, List, Tuple

# Import all optimization components
from opencog_cosmeceutical_optimizer import *
from moses_formulation_optimizer import *


class CompleteCosmeticOptimizer:
    """Complete cosmeceutical optimization system integrating all components"""
    
    def __init__(self):
        self.atomspace = AtomSpace()
        self.attention_module = ECANAttentionModule(self.atomspace)
        self.reasoning_engine = PLNReasoningEngine(self.atomspace)
        self.skin_model = MultiscaleSkinModel()
        self.inci_parser = INCIParser()
        self.moses_optimizer = MOSESFormulationOptimizer(self.atomspace, self.skin_model)
        
        self.performance_metrics = {}
        self.optimization_history = []
    
    def initialize_knowledge_base(self, ingredient_database: Dict[str, Dict]) -> None:
        """Initialize comprehensive ingredient knowledge base"""
        print("1. Initializing Cognitive Knowledge Base")
        print("="*50)
        
        start_time = time.time()
        
        for name, properties in ingredient_database.items():
            atom = CognitiveAtom(name, properties['type'], properties)
            self.atomspace.add_atom(atom)
            print(f"  Added: {atom}")
        
        # Create interaction links based on known compatibilities
        self._create_interaction_links(ingredient_database)
        
        init_time = time.time() - start_time
        self.performance_metrics['knowledge_init_time'] = init_time
        
        print(f"\nKnowledge base initialized: {len(self.atomspace.atoms)} atoms, "
              f"{len(self.atomspace.links)} links")
        print(f"Initialization time: {init_time:.3f}s\n")
    
    def _create_interaction_links(self, ingredient_db: Dict[str, Dict]) -> None:
        """Create interaction links between ingredients"""
        known_synergies = [
            ('vitamin_c', 'vitamin_e', 0.85),
            ('retinol', 'niacinamide', 0.75),
            ('hyaluronic_acid', 'glycerin', 0.80),
            ('ceramides', 'cholesterol', 0.90),
            ('peptides', 'vitamin_c', 0.70)
        ]
        
        known_incompatibilities = [
            ('vitamin_c', 'retinol', 0.30),
            ('salicylic_acid', 'retinol', 0.20),
            ('glycolic_acid', 'retinol', 0.25)
        ]
        
        for ing1, ing2, strength in known_synergies:
            if ing1 in ingredient_db and ing2 in ingredient_db:
                atom1 = self.atomspace.get_atom(ing1)
                atom2 = self.atomspace.get_atom(ing2)
                if atom1 and atom2:
                    link = CognitiveLink('SYNERGY', [atom1, atom2], 
                                       TruthValue(strength, 0.8))
                    self.atomspace.add_link(link)
        
        for ing1, ing2, strength in known_incompatibilities:
            if ing1 in ingredient_db and ing2 in ingredient_db:
                atom1 = self.atomspace.get_atom(ing1)
                atom2 = self.atomspace.get_atom(ing2)
                if atom1 and atom2:
                    link = CognitiveLink('INCOMPATIBILITY', [atom1, atom2], 
                                       TruthValue(strength, 0.9))
                    self.atomspace.add_link(link)
    
    def analyze_inci_constraints(self, target_inci: str, available_ingredients: List[str]) -> Dict:
        """Perform INCI-driven search space reduction"""
        print("2. INCI-Driven Search Space Analysis")
        print("="*50)
        
        start_time = time.time()
        
        # Parse target INCI
        parsed_inci = self.inci_parser.parse_inci_list(target_inci)
        concentrations = self.inci_parser.estimate_concentrations(parsed_inci)
        
        print(f"Target INCI: {target_inci}")
        print(f"Parsed ingredients: {parsed_inci}")
        print("\nEstimated concentrations:")
        for ingredient, conc in concentrations.items():
            print(f"  {ingredient}: {conc:.2f}%")
        
        # Reduce search space
        reduced_space = self.inci_parser.reduce_search_space(available_ingredients, parsed_inci)
        
        reduction_ratio = len(reduced_space) / len(available_ingredients)
        
        analysis_time = time.time() - start_time
        self.performance_metrics['inci_analysis_time'] = analysis_time
        
        print(f"\nSearch space reduction:")
        print(f"  Original: {len(available_ingredients)} ingredients")
        print(f"  Reduced: {len(reduced_space)} ingredients")
        print(f"  Reduction ratio: {reduction_ratio:.2f}")
        print(f"  Analysis time: {analysis_time:.3f}s")
        
        print(f"\nRetained ingredients: {reduced_space}\n")
        
        return {
            'parsed_inci': parsed_inci,
            'concentrations': concentrations,
            'reduced_ingredients': reduced_space,
            'reduction_ratio': reduction_ratio
        }
    
    def apply_cognitive_attention(self, focus_criteria: Dict[str, float]) -> List[str]:
        """Apply ECAN attention mechanism for ingredient prioritization"""
        print("3. ECAN Attention Allocation")
        print("="*50)
        
        start_time = time.time()
        
        # Apply attention boosts based on criteria
        for ingredient, boost in focus_criteria.items():
            self.attention_module.boost_attention(ingredient, boost)
            print(f"  Boosted attention: {ingredient} (+{boost})")
        
        # Update attention across the network
        self.attention_module.update_attention()
        
        # Get most attended ingredients
        top_attended = self.attention_module.get_most_attended_atoms(10)
        
        attention_time = time.time() - start_time
        self.performance_metrics['attention_time'] = attention_time
        
        print(f"\nTop attended ingredients:")
        for i, atom in enumerate(top_attended, 1):
            print(f"  {i:2d}. {atom.name}: {atom.attention.total_attention():.2f}")
        
        print(f"\nAttention allocation time: {attention_time:.3f}s\n")
        
        return [atom.name for atom in top_attended]
    
    def perform_pln_reasoning(self, candidate_ingredients: List[str]) -> Dict:
        """Perform PLN reasoning for ingredient interactions"""
        print("4. PLN Probabilistic Reasoning")
        print("="*50)
        
        start_time = time.time()
        
        # Analyze pairwise compatibility
        compatibility_matrix = {}
        for i, ing1 in enumerate(candidate_ingredients):
            for ing2 in candidate_ingredients[i+1:]:
                compatibility = self.reasoning_engine.reason_about_compatibility(ing1, ing2)
                compatibility_matrix[(ing1, ing2)] = compatibility
        
        # Infer synergies
        synergies = self.reasoning_engine.infer_synergy(candidate_ingredients)
        
        reasoning_time = time.time() - start_time
        self.performance_metrics['reasoning_time'] = reasoning_time
        
        print("Compatibility Analysis:")
        high_compatibility = []
        for (ing1, ing2), tv in compatibility_matrix.items():
            if tv.strength > 0.7:
                high_compatibility.append((ing1, ing2, tv))
                print(f"  {ing1} + {ing2}: {tv}")
        
        print(f"\nSynergy Analysis:")
        strong_synergies = []
        for (ing1, ing2), tv in synergies.items():
            if tv.strength > 0.6:
                strong_synergies.append((ing1, ing2, tv))
                print(f"  {ing1} + {ing2}: {tv}")
        
        print(f"\nReasoning time: {reasoning_time:.3f}s")
        print(f"High compatibility pairs: {len(high_compatibility)}")
        print(f"Strong synergy pairs: {len(strong_synergies)}\n")
        
        return {
            'compatibility_matrix': compatibility_matrix,
            'synergies': synergies,
            'high_compatibility': high_compatibility,
            'strong_synergies': strong_synergies
        }
    
    def optimize_multiscale_formulation(self, candidate_ingredients: List[str], 
                                      target_vectors: List[str]) -> Tuple[FormulationGenome, FitnessScore]:
        """Perform MOSES evolutionary optimization with multiscale targeting"""
        print("5. MOSES Evolutionary Optimization with Multiscale Targeting")
        print("="*50)
        
        start_time = time.time()
        
        # Configure optimizer for target vectors
        self.moses_optimizer.max_generations = 75
        self.moses_optimizer.population_size = 40
        
        print(f"Target therapeutic vectors: {target_vectors}")
        print(f"Candidate ingredients: {candidate_ingredients}")
        print(f"Population size: {self.moses_optimizer.population_size}")
        print(f"Max generations: {self.moses_optimizer.max_generations}")
        
        # Run optimization
        best_formulation, best_fitness = self.moses_optimizer.optimize(candidate_ingredients)
        
        optimization_time = time.time() - start_time
        self.performance_metrics['optimization_time'] = optimization_time
        
        print(f"\nOptimization completed in {optimization_time:.1f}s")
        print(f"Best overall fitness: {best_fitness.overall_fitness():.3f}")
        
        # Detailed fitness breakdown
        print(f"\nFitness Components:")
        print(f"  Therapeutic Efficacy: {best_fitness.efficacy:.3f}")
        print(f"  Formulation Stability: {best_fitness.stability:.3f}")
        print(f"  Safety Assessment: {best_fitness.safety:.3f}")
        print(f"  Economic Cost: {best_fitness.cost:.3f}")
        print(f"  Regulatory Compliance: {best_fitness.regulatory_compliance:.3f}")
        print(f"  Consumer Acceptance: {best_fitness.consumer_acceptance:.3f}")
        
        # Store optimization history
        self.optimization_history.append({
            'timestamp': time.time(),
            'formulation': best_formulation,
            'fitness': best_fitness,
            'target_vectors': target_vectors
        })
        
        print("\n")
        return best_formulation, best_fitness
    
    def analyze_multiscale_performance(self, formulation: FormulationGenome) -> Dict:
        """Analyze formulation performance across skin scales"""
        print("6. Multiscale Skin Model Analysis")
        print("="*50)
        
        start_time = time.time()
        
        # Therapeutic efficacy analysis
        efficacy_scores = self.skin_model.evaluate_therapeutic_efficacy(formulation)
        
        print("Therapeutic Vector Performance:")
        for vector, score in efficacy_scores.items():
            print(f"  {vector}: {score:.3f}")
        
        # Layer-specific penetration analysis
        print(f"\nPenetration Analysis:")
        penetration_data = {}
        
        for ingredient, concentration in formulation.ingredients.items():
            if concentration > 0.1:  # Only analyze significant concentrations
                profile = self.skin_model.calculate_penetration_profile(ingredient, concentration)
                penetration_data[ingredient] = profile
                
                print(f"  {ingredient} ({concentration:.2f}%):")
                for layer, layer_conc in profile.items():
                    if layer_conc > 0.01:
                        print(f"    {layer.value}: {layer_conc:.3f}%")
        
        # Calculate layer-specific therapeutic loads
        layer_loads = {layer: 0.0 for layer in SkinLayer}
        for ingredient, profile in penetration_data.items():
            for layer, conc in profile.items():
                layer_loads[layer] += conc
        
        print(f"\nTotal Therapeutic Load by Layer:")
        for layer, load in layer_loads.items():
            if load > 0.01:
                print(f"  {layer.value}: {load:.3f}%")
        
        analysis_time = time.time() - start_time
        self.performance_metrics['multiscale_analysis_time'] = analysis_time
        
        print(f"\nMultiscale analysis time: {analysis_time:.3f}s\n")
        
        return {
            'efficacy_scores': efficacy_scores,
            'penetration_data': penetration_data,
            'layer_loads': layer_loads
        }
    
    def generate_formulation_report(self, formulation: FormulationGenome, 
                                  fitness: FitnessScore, analysis: Dict) -> Dict:
        """Generate comprehensive formulation report"""
        print("7. Comprehensive Formulation Report")
        print("="*50)
        
        report = {
            'formulation_id': f"OPT_{int(time.time())}",
            'timestamp': time.time(),
            'formulation_properties': {
                'ph_target': formulation.ph_target,
                'viscosity_target': formulation.viscosity_target,
                'total_actives': sum(conc for ing, conc in formulation.ingredients.items() 
                                   if ing in ['retinol', 'niacinamide', 'vitamin_c', 'peptides', 'salicylic_acid'])
            },
            'ingredient_profile': dict(sorted(formulation.ingredients.items(), 
                                            key=lambda x: x[1], reverse=True)),
            'fitness_scores': {
                'overall': fitness.overall_fitness(),
                'efficacy': fitness.efficacy,
                'stability': fitness.stability,
                'safety': fitness.safety,
                'cost': fitness.cost,
                'regulatory': fitness.regulatory_compliance,
                'consumer': fitness.consumer_acceptance
            },
            'therapeutic_performance': analysis['efficacy_scores'],
            'penetration_profiles': {
                ing: {layer.value: conc for layer, conc in profile.items()}
                for ing, profile in analysis['penetration_data'].items()
            },
            'layer_therapeutic_loads': {
                layer.value: load for layer, load in analysis['layer_loads'].items()
            },
            'performance_metrics': self.performance_metrics,
            'regulatory_status': 'COMPLIANT' if fitness.regulatory_compliance == 1.0 else 'NON_COMPLIANT'
        }
        
        print(f"Formulation ID: {report['formulation_id']}")
        print(f"Overall Fitness: {report['fitness_scores']['overall']:.3f}")
        print(f"Regulatory Status: {report['regulatory_status']}")
        print(f"Total Active Concentration: {report['formulation_properties']['total_actives']:.2f}%")
        
        print(f"\nTop 5 Ingredients:")
        for i, (ingredient, conc) in enumerate(list(report['ingredient_profile'].items())[:5], 1):
            print(f"  {i}. {ingredient}: {conc:.2f}%")
        
        return report
    
    def save_optimization_results(self, report: Dict, filename: str = None) -> str:
        """Save optimization results to JSON file"""
        if filename is None:
            filename = f"/tmp/formulation_{report['formulation_id']}.json"
        
        # Convert any non-serializable objects
        serializable_report = json.loads(json.dumps(report, default=str))
        
        with open(filename, 'w') as f:
            json.dump(serializable_report, f, indent=2)
        
        print(f"\nResults saved to: {filename}")
        return filename


def create_comprehensive_ingredient_database() -> Dict[str, Dict]:
    """Create comprehensive ingredient database for testing"""
    return {
        'retinol': {
            'type': 'ACTIVE_INGREDIENT',
            'mechanism': 'cell_renewal',
            'target_layers': ['epidermis', 'dermis_papillary'],
            'max_concentration': 0.3,
            'molecular_weight': 286.45,
            'lipophilic': True,
            'stability': 'light_sensitive'
        },
        'niacinamide': {
            'type': 'ACTIVE_INGREDIENT',
            'mechanism': 'barrier_repair',
            'target_layers': ['epidermis'],
            'max_concentration': 12.0,
            'molecular_weight': 122.12,
            'lipophilic': False,
            'stability': 'stable'
        },
        'hyaluronic_acid': {
            'type': 'HUMECTANT',
            'mechanism': 'hydration',
            'target_layers': ['stratum_corneum', 'epidermis'],
            'max_concentration': 2.0,
            'molecular_weight': 1000000,  # High MW version
            'lipophilic': False,
            'stability': 'stable'
        },
        'vitamin_c': {
            'type': 'ANTIOXIDANT',
            'mechanism': 'collagen_synthesis',
            'target_layers': ['dermis_papillary'],
            'max_concentration': 20.0,
            'molecular_weight': 176.12,
            'lipophilic': False,
            'stability': 'oxidation_sensitive'
        },
        'vitamin_e': {
            'type': 'ANTIOXIDANT',
            'mechanism': 'lipid_protection',
            'target_layers': ['stratum_corneum'],
            'max_concentration': 1.0,
            'molecular_weight': 430.71,
            'lipophilic': True,
            'stability': 'stable'
        },
        'glycerin': {
            'type': 'HUMECTANT',
            'mechanism': 'moisture_retention',
            'target_layers': ['stratum_corneum'],
            'max_concentration': 15.0,
            'molecular_weight': 92.09,
            'lipophilic': False,
            'stability': 'stable'
        },
        'ceramides': {
            'type': 'EMOLLIENT',
            'mechanism': 'barrier_restoration',
            'target_layers': ['stratum_corneum'],
            'max_concentration': 5.0,
            'molecular_weight': 600,  # Average
            'lipophilic': True,
            'stability': 'stable'
        },
        'peptides': {
            'type': 'ACTIVE_INGREDIENT',
            'mechanism': 'collagen_synthesis',
            'target_layers': ['dermis_papillary'],
            'max_concentration': 10.0,
            'molecular_weight': 1200,  # Small peptides
            'lipophilic': False,
            'stability': 'enzyme_sensitive'
        },
        'alpha_arbutin': {
            'type': 'ACTIVE_INGREDIENT',
            'mechanism': 'melanin_inhibition',
            'target_layers': ['epidermis'],
            'max_concentration': 2.0,
            'molecular_weight': 272.25,
            'lipophilic': False,
            'stability': 'stable'
        },
        'azelaic_acid': {
            'type': 'ACTIVE_INGREDIENT',
            'mechanism': 'exfoliation',
            'target_layers': ['epidermis'],
            'max_concentration': 10.0,
            'molecular_weight': 188.22,
            'lipophilic': False,
            'stability': 'stable'
        }
    }


def run_complete_demonstration():
    """Run complete optimization demonstration"""
    print("="*80)
    print("OPENCOG-INSPIRED COSMECEUTICAL OPTIMIZATION DEMONSTRATION")
    print("="*80)
    print()
    
    # Initialize system
    optimizer = CompleteCosmeticOptimizer()
    
    # Create ingredient database
    ingredient_db = create_comprehensive_ingredient_database()
    available_ingredients = list(ingredient_db.keys())
    
    # Step 1: Initialize knowledge base
    optimizer.initialize_knowledge_base(ingredient_db)
    
    # Step 2: Analyze INCI constraints
    target_inci = "Aqua, Glycerin, Niacinamide, Hyaluronic Acid, Retinol, Vitamin E, Phenoxyethanol"
    inci_analysis = optimizer.analyze_inci_constraints(target_inci, available_ingredients)
    
    # Step 3: Apply cognitive attention
    focus_criteria = {
        'retinol': 25.0,      # High efficacy for anti-aging
        'niacinamide': 20.0,  # Safe and effective
        'vitamin_c': 18.0,    # Powerful antioxidant
        'hyaluronic_acid': 15.0,  # Excellent hydration
        'peptides': 12.0      # Advanced anti-aging
    }
    top_attended = optimizer.apply_cognitive_attention(focus_criteria)
    
    # Step 4: PLN reasoning
    reasoning_results = optimizer.perform_pln_reasoning(inci_analysis['reduced_ingredients'])
    
    # Step 5: MOSES optimization
    target_vectors = ['anti_aging', 'barrier_repair', 'hydration']
    best_formulation, best_fitness = optimizer.optimize_multiscale_formulation(
        inci_analysis['reduced_ingredients'], target_vectors
    )
    
    # Step 6: Multiscale analysis
    multiscale_analysis = optimizer.analyze_multiscale_performance(best_formulation)
    
    # Step 7: Generate report
    final_report = optimizer.generate_formulation_report(
        best_formulation, best_fitness, multiscale_analysis
    )
    
    # Step 8: Save results
    report_file = optimizer.save_optimization_results(final_report)
    
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE - PERFORMANCE SUMMARY")
    print("="*80)
    
    total_time = sum(optimizer.performance_metrics.values())
    print(f"Total processing time: {total_time:.3f}s")
    print(f"Knowledge initialization: {optimizer.performance_metrics['knowledge_init_time']:.3f}s")
    print(f"INCI analysis: {optimizer.performance_metrics['inci_analysis_time']:.3f}s")
    print(f"Attention allocation: {optimizer.performance_metrics['attention_time']:.3f}s")
    print(f"PLN reasoning: {optimizer.performance_metrics['reasoning_time']:.3f}s")
    print(f"MOSES optimization: {optimizer.performance_metrics['optimization_time']:.3f}s")
    print(f"Multiscale analysis: {optimizer.performance_metrics['multiscale_analysis_time']:.3f}s")
    
    print(f"\nSearch space reduction: {inci_analysis['reduction_ratio']:.1%}")
    print(f"Final formulation fitness: {best_fitness.overall_fitness():.3f}")
    print(f"Regulatory compliance: {'✓ PASS' if best_fitness.regulatory_compliance == 1.0 else '✗ FAIL'}")
    
    print(f"\nResults saved to: {report_file}")
    print("\n" + "="*80)
    
    return final_report


if __name__ == "__main__":
    report = run_complete_demonstration()
    
    print("\n🎉 DEMONSTRATION COMPLETE!")
    print("\nThis demonstration successfully showcased:")
    print("• OpenCog AtomSpace knowledge representation")
    print("• ECAN attention allocation for promising formulation spaces") 
    print("• PLN reasoning for ingredient compatibility and synergy")
    print("• MOSES evolutionary optimization with multiscale targeting")
    print("• INCI-driven search space reduction")
    print("• Comprehensive multiscale skin model integration")
    print("• Full regulatory compliance checking")
    print("• Performance optimization and benchmarking")
    
    print(f"\nFormulation ID: {report['formulation_id']}")
    print(f"Overall Performance: {report['fitness_scores']['overall']:.1%}")
    print(f"Processing Efficiency: {sum(report['performance_metrics'].values()):.1f}s total")