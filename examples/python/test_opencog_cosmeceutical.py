#!/usr/bin/env python3
"""
Test Suite for OpenCog-Inspired Cosmeceutical Optimization

Comprehensive tests for the cognitive architecture components including:
- AtomSpace knowledge representation
- ECAN attention allocation
- PLN reasoning engine
- MOSES-like evolutionary optimization
- Multiscale skin modeling
- INCI parsing and analysis

Author: ONNX Runtime Cosmeceutical Optimization Team
"""

import unittest
import sys
import os
from typing import Dict, List

# Import modules to test
from opencog_cosmeceutical_optimizer import (
    AtomSpace, CognitiveAtom, CognitiveLink, TruthValue, AttentionValue,
    ECANAttentionModule, PLNReasoningEngine, INCIParser,
    SkinLayer, DeliveryMechanism, TherapeuticVector
)

from moses_formulation_optimizer import (
    FormulationGenome, MultiscaleSkinModel, FitnessScore, MOSESFormulationOptimizer
)


class TestAtomSpace(unittest.TestCase):
    """Test AtomSpace knowledge representation"""
    
    def setUp(self):
        self.atomspace = AtomSpace()
    
    def test_atom_creation_and_retrieval(self):
        """Test creating and retrieving atoms"""
        atom = CognitiveAtom('test_ingredient', 'ACTIVE_INGREDIENT')
        added_atom = self.atomspace.add_atom(atom)
        
        self.assertEqual(added_atom.name, 'test_ingredient')
        self.assertEqual(added_atom.atom_type, 'ACTIVE_INGREDIENT')
        
        retrieved = self.atomspace.get_atom('test_ingredient')
        self.assertEqual(retrieved, added_atom)
    
    def test_link_creation(self):
        """Test creating links between atoms"""
        atom1 = CognitiveAtom('ingredient1', 'ACTIVE_INGREDIENT')
        atom2 = CognitiveAtom('ingredient2', 'HUMECTANT')
        
        self.atomspace.add_atom(atom1)
        self.atomspace.add_atom(atom2)
        
        link = CognitiveLink('COMPATIBILITY', [atom1, atom2])
        added_link = self.atomspace.add_link(link)
        
        self.assertEqual(added_link.link_type, 'COMPATIBILITY')
        self.assertEqual(len(added_link.atoms), 2)
    
    def test_atom_type_filtering(self):
        """Test filtering atoms by type"""
        atom1 = CognitiveAtom('active1', 'ACTIVE_INGREDIENT')
        atom2 = CognitiveAtom('active2', 'ACTIVE_INGREDIENT')
        atom3 = CognitiveAtom('humectant1', 'HUMECTANT')
        
        self.atomspace.add_atom(atom1)
        self.atomspace.add_atom(atom2)
        self.atomspace.add_atom(atom3)
        
        actives = self.atomspace.get_atoms_by_type('ACTIVE_INGREDIENT')
        humectants = self.atomspace.get_atoms_by_type('HUMECTANT')
        
        self.assertEqual(len(actives), 2)
        self.assertEqual(len(humectants), 1)
    
    def test_time_advancement(self):
        """Test time advancement and attention decay"""
        atom = CognitiveAtom('test', 'ACTIVE_INGREDIENT')
        atom.attention.sti = 10.0
        self.atomspace.add_atom(atom)
        
        initial_sti = atom.attention.sti
        self.atomspace.advance_time()
        
        # STI should decay slightly
        self.assertLess(atom.attention.sti, initial_sti)


class TestTruthValues(unittest.TestCase):
    """Test PLN truth value operations"""
    
    def test_truth_value_creation(self):
        """Test creating truth values with proper bounds"""
        tv = TruthValue(0.8, 0.9)
        self.assertEqual(tv.strength, 0.8)
        self.assertEqual(tv.confidence, 0.9)
        
        # Test bounds
        tv_bounded = TruthValue(1.5, -0.1)
        self.assertEqual(tv_bounded.strength, 1.0)
        self.assertEqual(tv_bounded.confidence, 0.0)
    
    def test_revision_rule(self):
        """Test PLN revision rule for combining evidence"""
        tv1 = TruthValue(0.8, 0.7)
        tv2 = TruthValue(0.6, 0.8)
        
        revised = tv1.revision_rule(tv2)
        
        # Should combine evidence
        self.assertGreater(revised.confidence, max(tv1.confidence, tv2.confidence))
        self.assertGreater(revised.strength, 0.0)
        self.assertLessEqual(revised.strength, 1.0)


class TestAttentionMechanism(unittest.TestCase):
    """Test ECAN attention allocation"""
    
    def setUp(self):
        self.atomspace = AtomSpace()
        self.attention_module = ECANAttentionModule(self.atomspace)
    
    def test_attention_spreading(self):
        """Test attention spreading between connected atoms"""
        atom1 = CognitiveAtom('source', 'ACTIVE_INGREDIENT')
        atom2 = CognitiveAtom('target', 'HUMECTANT')
        
        atom1.attention.sti = 80.0  # Above focus boundary
        atom2.attention.sti = 10.0
        
        self.atomspace.add_atom(atom1)
        self.atomspace.add_atom(atom2)
        
        # Create connection
        link = CognitiveLink('COMPATIBILITY', [atom1, atom2], TruthValue(0.8, 0.7))
        self.atomspace.add_link(link)
        
        initial_target_sti = atom2.attention.sti
        self.attention_module.update_attention()
        
        # Target should receive some attention
        self.assertGreater(atom2.attention.sti, initial_target_sti)
    
    def test_attention_normalization(self):
        """Test attention budget normalization"""
        # Create many high-attention atoms
        for i in range(20):
            atom = CognitiveAtom(f'atom_{i}', 'ACTIVE_INGREDIENT')
            atom.attention.sti = 100.0
            self.atomspace.add_atom(atom)
        
        self.attention_module._normalize_attention()
        
        total_positive_sti = sum(max(0, atom.attention.sti) 
                               for atom in self.atomspace.atoms.values())
        
        # Should not exceed budget
        self.assertLessEqual(total_positive_sti, self.attention_module.attention_budget * 1.1)  # Small tolerance
    
    def test_most_attended_retrieval(self):
        """Test retrieving most attended atoms"""
        atoms = []
        for i in range(10):
            atom = CognitiveAtom(f'atom_{i}', 'ACTIVE_INGREDIENT')
            atom.attention.sti = i * 10.0  # Varying attention
            atoms.append(atom)
            self.atomspace.add_atom(atom)
        
        top_5 = self.attention_module.get_most_attended_atoms(5)
        
        self.assertEqual(len(top_5), 5)
        # Should be in descending order of attention
        for i in range(4):
            self.assertGreaterEqual(
                top_5[i].attention.total_attention(),
                top_5[i+1].attention.total_attention()
            )


class TestPLNReasoning(unittest.TestCase):
    """Test PLN reasoning engine"""
    
    def setUp(self):
        self.atomspace = AtomSpace()
        self.reasoning_engine = PLNReasoningEngine(self.atomspace)
    
    def test_compatibility_reasoning(self):
        """Test ingredient compatibility reasoning"""
        atom1 = CognitiveAtom('retinol', 'ACTIVE_INGREDIENT', 
                             {'mechanism_of_action': 'cell_renewal'})
        atom2 = CognitiveAtom('niacinamide', 'ACTIVE_INGREDIENT',
                             {'mechanism_of_action': 'barrier_repair'})
        
        self.atomspace.add_atom(atom1)
        self.atomspace.add_atom(atom2)
        
        compatibility = self.reasoning_engine.reason_about_compatibility('retinol', 'niacinamide')
        
        self.assertIsInstance(compatibility, TruthValue)
        self.assertGreaterEqual(compatibility.strength, 0.0)
        self.assertLessEqual(compatibility.strength, 1.0)
    
    def test_synergy_inference(self):
        """Test synergy inference between ingredients"""
        # Create antioxidant pair
        atom1 = CognitiveAtom('vitamin_c', 'ANTIOXIDANT')
        atom2 = CognitiveAtom('vitamin_e', 'ANTIOXIDANT')
        
        self.atomspace.add_atom(atom1)
        self.atomspace.add_atom(atom2)
        
        synergies = self.reasoning_engine.infer_synergy(['vitamin_c', 'vitamin_e'])
        
        self.assertIn(('vitamin_c', 'vitamin_e'), synergies)
        synergy_tv = synergies[('vitamin_c', 'vitamin_e')]
        self.assertGreater(synergy_tv.strength, 0.6)  # Should detect antioxidant synergy


class TestINCIParser(unittest.TestCase):
    """Test INCI parsing and analysis"""
    
    def setUp(self):
        self.parser = INCIParser()
    
    def test_inci_parsing(self):
        """Test parsing INCI strings"""
        inci_string = "Aqua, Glycerin, Niacinamide, Hyaluronic Acid, Phenoxyethanol"
        parsed = self.parser.parse_inci_list(inci_string)
        
        expected = ['aqua', 'glycerin', 'niacinamide', 'hyaluronic_acid', 'phenoxyethanol']
        self.assertEqual(parsed, expected)
    
    def test_concentration_estimation(self):
        """Test concentration estimation from INCI ordering"""
        inci_list = ['aqua', 'glycerin', 'niacinamide', 'phenoxyethanol']
        concentrations = self.parser.estimate_concentrations(inci_list)
        
        # First ingredient should have highest concentration
        self.assertGreater(concentrations['aqua'], concentrations['glycerin'])
        self.assertGreater(concentrations['glycerin'], concentrations['niacinamide'])
        
        # Total should be reasonable
        total = sum(concentrations.values())
        self.assertLess(total, 100.0)
        self.assertGreater(total, 50.0)
    
    def test_regulatory_limits(self):
        """Test regulatory limit enforcement"""
        inci_list = ['aqua', 'retinol']
        concentrations = self.parser.estimate_concentrations(inci_list)
        
        # Retinol should not exceed regulatory limit
        self.assertLessEqual(concentrations['retinol'], self.parser.concentration_rules['retinol'])
    
    def test_subset_compatibility(self):
        """Test ingredient subset compatibility checking"""
        product_inci = ['aqua', 'glycerin', 'niacinamide', 'phenoxyethanol']
        subset1 = {'glycerin', 'niacinamide'}
        subset2 = {'retinol', 'salicylic_acid'}
        
        self.assertTrue(self.parser.check_subset_compatibility(product_inci, subset1))
        self.assertFalse(self.parser.check_subset_compatibility(product_inci, subset2))
    
    def test_search_space_reduction(self):
        """Test ingredient search space reduction"""
        all_ingredients = ['retinol', 'niacinamide', 'glycerin', 'salicylic_acid', 'ceramides']
        target_inci = ['aqua', 'niacinamide', 'glycerin', 'phenoxyethanol']
        
        reduced = self.parser.reduce_search_space(all_ingredients, target_inci)
        
        # Should include INCI ingredients that are available
        self.assertIn('niacinamide', reduced)
        self.assertIn('glycerin', reduced)
        
        # Length should be reduced
        self.assertLessEqual(len(reduced), len(all_ingredients))


class TestMultiscaleSkinModel(unittest.TestCase):
    """Test multiscale skin model"""
    
    def setUp(self):
        self.skin_model = MultiscaleSkinModel()
    
    def test_skin_layer_properties(self):
        """Test skin layer property definitions"""
        sc_props = self.skin_model.layers[SkinLayer.STRATUM_CORNEUM]
        
        self.assertIn('thickness_um', sc_props)
        self.assertIn('permeability', sc_props)
        self.assertGreater(sc_props['thickness_um'], 0)
        self.assertGreaterEqual(sc_props['permeability'], 0)
        self.assertLessEqual(sc_props['permeability'], 1)
    
    def test_therapeutic_vectors(self):
        """Test therapeutic vector definitions"""
        anti_aging = self.skin_model.therapeutic_vectors['anti_aging']
        
        self.assertEqual(anti_aging.name, 'anti_aging')
        self.assertIn('collagen_stimulation', anti_aging.mechanism_of_action)
        self.assertGreater(len(anti_aging.target_layers), 0)
        self.assertGreater(len(anti_aging.synergistic_ingredients), 0)
    
    def test_penetration_profile(self):
        """Test ingredient penetration profile calculation"""
        profile = self.skin_model.calculate_penetration_profile('retinol', 1.0)
        
        # Should have entries for skin layers
        self.assertIn(SkinLayer.STRATUM_CORNEUM, profile)
        self.assertIn(SkinLayer.EPIDERMIS, profile)
        
        # Concentrations should be positive and sum to less than input
        total_penetrated = sum(profile.values())
        self.assertGreater(total_penetrated, 0)
        self.assertLessEqual(total_penetrated, 1.0)
    
    def test_therapeutic_efficacy_evaluation(self):
        """Test therapeutic efficacy evaluation"""
        formulation = FormulationGenome({
            'retinol': 0.5,
            'vitamin_c': 10.0,
            'niacinamide': 5.0,
            'ceramides': 2.0
        })
        
        efficacy = self.skin_model.evaluate_therapeutic_efficacy(formulation)
        
        # Should have scores for all vectors
        self.assertIn('anti_aging', efficacy)
        self.assertIn('barrier_repair', efficacy)
        self.assertIn('hydration', efficacy)
        
        # Scores should be in valid range
        for score in efficacy.values():
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)


class TestFormulationGenome(unittest.TestCase):
    """Test formulation genome operations"""
    
    def test_genome_creation(self):
        """Test creating formulation genomes"""
        ingredients = {'niacinamide': 5.0, 'hyaluronic_acid': 1.0}
        genome = FormulationGenome(ingredients)
        
        self.assertEqual(genome.ingredients['niacinamide'], 5.0)
        self.assertEqual(genome.ph_target, 6.0)
        self.assertIsInstance(genome.stability_enhancers, set)
    
    def test_genome_normalization(self):
        """Test automatic concentration normalization"""
        # Create over-concentrated formulation
        ingredients = {'ingredient1': 50.0, 'ingredient2': 60.0}
        genome = FormulationGenome(ingredients)
        
        # Should be normalized
        total = sum(genome.ingredients.values())
        self.assertLessEqual(total, 100.0)
    
    def test_mutation(self):
        """Test genome mutation"""
        original = FormulationGenome({'niacinamide': 5.0, 'glycerin': 2.0})
        mutated = original.mutate(mutation_rate=1.0)  # Force mutations
        
        # Should be different genome
        self.assertNotEqual(original.ingredients, mutated.ingredients)
        
        # Should maintain reasonable concentrations
        for conc in mutated.ingredients.values():
            self.assertGreater(conc, 0)
            self.assertLessEqual(conc, 20.0)
    
    def test_crossover(self):
        """Test genome crossover"""
        parent1 = FormulationGenome({'niacinamide': 5.0, 'retinol': 0.3})
        parent2 = FormulationGenome({'glycerin': 3.0, 'vitamin_c': 10.0})
        
        child1, child2 = parent1.crossover(parent2)
        
        # Children should have some ingredients from both parents
        all_parent_ingredients = set(parent1.ingredients.keys()) | set(parent2.ingredients.keys())
        child1_ingredients = set(child1.ingredients.keys())
        child2_ingredients = set(child2.ingredients.keys())
        
        # At least one child should have some overlap with parent ingredients
        has_overlap = bool(child1_ingredients & all_parent_ingredients) or bool(child2_ingredients & all_parent_ingredients)
        self.assertTrue(has_overlap)
        
        # Both children should have some ingredients
        self.assertGreater(len(child1.ingredients), 0)
        self.assertGreater(len(child2.ingredients), 0)


class TestMOSESOptimizer(unittest.TestCase):
    """Test MOSES-inspired evolutionary optimizer"""
    
    def setUp(self):
        self.atomspace = AtomSpace()
        self.skin_model = MultiscaleSkinModel()
        self.optimizer = MOSESFormulationOptimizer(self.atomspace, self.skin_model)
    
    def test_population_initialization(self):
        """Test random population initialization"""
        base_ingredients = ['niacinamide', 'retinol', 'glycerin', 'vitamin_c']
        population = self.optimizer.initialize_population(base_ingredients)
        
        self.assertEqual(len(population), self.optimizer.population_size)
        
        # Each genome should have reasonable number of ingredients
        for genome in population:
            self.assertGreater(len(genome.ingredients), 0)
            self.assertLessEqual(len(genome.ingredients), len(base_ingredients))
    
    def test_fitness_evaluation(self):
        """Test multi-objective fitness evaluation"""
        genome = FormulationGenome({
            'niacinamide': 5.0,
            'hyaluronic_acid': 1.0,
            'glycerin': 3.0
        })
        
        fitness = self.optimizer.evaluate_fitness(genome)
        
        # Check all fitness components
        self.assertIsInstance(fitness.efficacy, float)
        self.assertIsInstance(fitness.stability, float)
        self.assertIsInstance(fitness.safety, float)
        self.assertIsInstance(fitness.cost, float)
        self.assertIsInstance(fitness.regulatory_compliance, float)
        self.assertIsInstance(fitness.consumer_acceptance, float)
        
        # All should be in [0, 1] range
        components = [fitness.efficacy, fitness.stability, fitness.safety, 
                     fitness.cost, fitness.regulatory_compliance, fitness.consumer_acceptance]
        for component in components:
            self.assertGreaterEqual(component, 0.0)
            self.assertLessEqual(component, 1.0)
    
    def test_fitness_overall_calculation(self):
        """Test overall fitness calculation"""
        fitness = FitnessScore(
            efficacy=0.8,
            stability=0.7,
            safety=0.9,
            cost=0.3,
            regulatory_compliance=1.0,
            consumer_acceptance=0.6
        )
        
        overall = fitness.overall_fitness()
        
        self.assertGreaterEqual(overall, 0.0)
        self.assertLessEqual(overall, 1.0)
        self.assertIsInstance(overall, float)
    
    def test_selection_mechanism(self):
        """Test tournament selection"""
        # Create population with known fitness values
        population = [
            FormulationGenome({'ingredient1': 1.0}),
            FormulationGenome({'ingredient2': 2.0}),
            FormulationGenome({'ingredient3': 3.0})
        ]
        
        fitness_scores = [
            FitnessScore(efficacy=0.5),  # Low fitness
            FitnessScore(efficacy=0.9),  # High fitness
            FitnessScore(efficacy=0.7)   # Medium fitness
        ]
        
        selected = self.optimizer.selection(population, fitness_scores)
        
        self.assertEqual(len(selected), self.optimizer.population_size)
        
        # Elite should be preserved (highest fitness individual)
        elite_genome = population[1]  # Index 1 has highest fitness
        self.assertIn(elite_genome.ingredients['ingredient2'], 
                     [g.ingredients.get('ingredient2', 0) for g in selected])


class TestIntegrationScenarios(unittest.TestCase):
    """Test integration scenarios combining multiple components"""
    
    def setUp(self):
        self.atomspace = AtomSpace()
        self.attention_module = ECANAttentionModule(self.atomspace)
        self.reasoning_engine = PLNReasoningEngine(self.atomspace)
        self.skin_model = MultiscaleSkinModel()
        self.inci_parser = INCIParser()
    
    def test_full_cognitive_pipeline(self):
        """Test complete cognitive processing pipeline"""
        # 1. Initialize knowledge base
        ingredients = [
            ('retinol', 'ACTIVE_INGREDIENT'),
            ('niacinamide', 'ACTIVE_INGREDIENT'),
            ('hyaluronic_acid', 'HUMECTANT'),
            ('vitamin_c', 'ANTIOXIDANT')
        ]
        
        for name, atom_type in ingredients:
            atom = CognitiveAtom(name, atom_type)
            self.atomspace.add_atom(atom)
        
        # 2. Reason about compatibility
        compatibility = self.reasoning_engine.reason_about_compatibility('retinol', 'niacinamide')
        self.assertIsInstance(compatibility, TruthValue)
        
        # 3. Update attention based on reasoning
        self.attention_module.boost_attention('retinol', 15.0)
        self.attention_module.update_attention()
        
        top_attended = self.attention_module.get_most_attended_atoms(2)
        self.assertEqual(len(top_attended), 2)
        
        # 4. Parse INCI and reduce search space
        inci_string = "aqua, glycerin, niacinamide, hyaluronic_acid, phenoxyethanol"
        parsed_inci = self.inci_parser.parse_inci_list(inci_string)
        
        all_ingredients = [name for name, _ in ingredients] + ['ceramides', 'peptides']
        reduced_space = self.inci_parser.reduce_search_space(all_ingredients, parsed_inci)
        
        self.assertIn('niacinamide', reduced_space)
        self.assertIn('hyaluronic_acid', reduced_space)
    
    def test_optimization_with_cognitive_guidance(self):
        """Test optimization guided by cognitive attention"""
        # Set up optimizer
        optimizer = MOSESFormulationOptimizer(self.atomspace, self.skin_model)
        
        # Create knowledge base
        ingredients = ['niacinamide', 'retinol', 'hyaluronic_acid', 'vitamin_c', 'glycerin']
        for ingredient in ingredients:
            atom = CognitiveAtom(ingredient, 'INGREDIENT')
            self.atomspace.add_atom(atom)
        
        # Boost attention for promising ingredients
        self.attention_module.boost_attention('niacinamide', 20.0)
        self.attention_module.boost_attention('hyaluronic_acid', 15.0)
        
        # Run short optimization
        optimizer.max_generations = 5  # Quick test
        best_formulation, best_fitness = optimizer.optimize(ingredients)
        
        self.assertIsInstance(best_formulation, FormulationGenome)
        self.assertIsInstance(best_fitness, FitnessScore)
        self.assertGreater(best_fitness.overall_fitness(), 0.0)
        
        # Check that high-attention ingredients are preferred
        high_attention_ingredients = ['niacinamide', 'hyaluronic_acid']
        formulation_ingredients = set(best_formulation.ingredients.keys())
        
        # At least one high-attention ingredient should be present
        self.assertTrue(any(ing in formulation_ingredients for ing in high_attention_ingredients))


def run_all_tests():
    """Run all test suites"""
    # Create test suite
    test_classes = [
        TestAtomSpace,
        TestTruthValues,
        TestAttentionMechanism,
        TestPLNReasoning,
        TestINCIParser,
        TestMultiscaleSkinModel,
        TestFormulationGenome,
        TestMOSESOptimizer,
        TestIntegrationScenarios
    ]
    
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Test Summary:")
    print(f"  Tests run: {result.testsRun}")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    print(f"  Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"{'='*60}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)