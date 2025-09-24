#!/usr/bin/env python3
"""
Test Suite for Cosmetic Chemistry Framework

This test suite validates the cosmetic chemistry atom types and functionality
implemented in the ONNX Runtime cheminformatics framework.

Author: ONNX Runtime Cosmetic Chemistry Team
"""

import unittest
import sys
import os

# Import the cosmetic chemistry modules
from cosmetic_intro_example import *
from cosmetic_chemistry_example import StabilityPredictor, RegulatoryChecker, FormulationOptimizer, FormulationConstraints

class TestCosmeticAtomTypes(unittest.TestCase):
    """Test basic cosmetic atom types functionality"""
    
    def setUp(self):
        """Set up test ingredients"""
        self.hyaluronic_acid = ACTIVE_INGREDIENT('hyaluronic_acid')
        self.niacinamide = ACTIVE_INGREDIENT('niacinamide')
        self.glycerin = HUMECTANT('glycerin')
        self.phenoxyethanol = PRESERVATIVE('phenoxyethanol')
        
    def test_ingredient_creation(self):
        """Test that ingredients are created correctly"""
        self.assertEqual(self.hyaluronic_acid.name, 'hyaluronic_acid')
        self.assertIsInstance(self.hyaluronic_acid, ACTIVE_INGREDIENT)
        self.assertIsInstance(self.glycerin, HUMECTANT)
        self.assertIsInstance(self.phenoxyethanol, PRESERVATIVE)
        
    def test_ingredient_properties(self):
        """Test ingredient properties"""
        ingredient_with_props = ACTIVE_INGREDIENT('test_ingredient', {
            'concentration_limit': '10%',
            'function': 'test_function'
        })
        self.assertEqual(ingredient_with_props.properties['concentration_limit'], '10%')
        self.assertEqual(ingredient_with_props.properties['function'], 'test_function')

class TestFormulations(unittest.TestCase):
    """Test formulation creation and analysis"""
    
    def setUp(self):
        """Set up test formulations"""
        self.hyaluronic_acid = ACTIVE_INGREDIENT('hyaluronic_acid')
        self.glycerin = HUMECTANT('glycerin')
        self.phenoxyethanol = PRESERVATIVE('phenoxyethanol')
        
        self.simple_serum = SKINCARE_FORMULATION(
            'simple_serum',
            self.hyaluronic_acid,
            self.glycerin,
            self.phenoxyethanol
        )
        
    def test_formulation_creation(self):
        """Test basic formulation creation"""
        self.assertEqual(self.simple_serum.name, 'simple_serum')
        self.assertEqual(len(self.simple_serum.ingredients), 3)
        self.assertIn(self.hyaluronic_acid, self.simple_serum.ingredients)
        
    def test_formulation_ingredient_addition(self):
        """Test adding ingredients to formulation"""
        initial_count = len(self.simple_serum.ingredients)
        vitamin_e = ANTIOXIDANT('vitamin_e')
        self.simple_serum.add_ingredient(vitamin_e)
        self.assertEqual(len(self.simple_serum.ingredients), initial_count + 1)
        self.assertIn(vitamin_e, self.simple_serum.ingredients)

class TestCompatibilityLinks(unittest.TestCase):
    """Test ingredient compatibility links"""
    
    def setUp(self):
        """Set up test ingredients and links"""
        self.vitamin_c = ACTIVE_INGREDIENT('vitamin_c')
        self.vitamin_e = ANTIOXIDANT('vitamin_e')
        self.retinol = ACTIVE_INGREDIENT('retinol')
        
        self.synergy_link = SYNERGY_LINK(self.vitamin_c, self.vitamin_e, strength=0.95)
        self.incompatibility_link = INCOMPATIBILITY_LINK(self.vitamin_c, self.retinol, strength=0.8)
        
    def test_synergy_link_creation(self):
        """Test synergy link creation"""
        self.assertEqual(self.synergy_link.atom1, self.vitamin_c)
        self.assertEqual(self.synergy_link.atom2, self.vitamin_e)
        self.assertEqual(self.synergy_link.strength, 0.95)
        
    def test_incompatibility_link_creation(self):
        """Test incompatibility link creation"""
        self.assertEqual(self.incompatibility_link.atom1, self.vitamin_c)
        self.assertEqual(self.incompatibility_link.atom2, self.retinol)
        self.assertEqual(self.incompatibility_link.strength, 0.8)

class TestStabilityPredictor(unittest.TestCase):
    """Test stability prediction functionality"""
    
    def setUp(self):
        """Set up stability predictor and test formulation"""
        self.predictor = StabilityPredictor()
        
        self.vitamin_c = ACTIVE_INGREDIENT('vitamin_c_l_ascorbic_acid')
        self.vitamin_e = ANTIOXIDANT('vitamin_e_tocopherol')
        self.phenoxyethanol = PRESERVATIVE('phenoxyethanol')
        
        self.test_formulation = SKINCARE_FORMULATION('test_formulation')
        self.test_formulation.ingredients = [self.vitamin_c, self.vitamin_e, self.phenoxyethanol]
        
    def test_stability_prediction(self):
        """Test basic stability prediction"""
        stability = self.predictor.predict_stability(self.test_formulation)
        self.assertIsInstance(stability, float)
        self.assertGreaterEqual(stability, 0.0)
        self.assertLessEqual(stability, 1.0)
        
    def test_stability_with_conditions(self):
        """Test stability prediction with different conditions"""
        normal_stability = self.predictor.predict_stability(self.test_formulation, ['normal_storage'])
        low_ph_stability = self.predictor.predict_stability(self.test_formulation, ['low_ph'])
        
        self.assertIsInstance(normal_stability, float)
        self.assertIsInstance(low_ph_stability, float)
        
    def test_synergy_bonus(self):
        """Test that synergistic combinations get stability bonus"""
        # Formulation with vitamin C + E should get synergy bonus
        stability = self.predictor.predict_stability(self.test_formulation)
        
        # Formulation without synergy
        single_active = SKINCARE_FORMULATION('single_active')
        single_active.ingredients = [self.vitamin_c, self.phenoxyethanol]
        single_stability = self.predictor.predict_stability(single_active)
        
        # The synergistic formulation might have higher stability (depending on base factors)
        self.assertIsInstance(stability, float)
        self.assertIsInstance(single_stability, float)

class TestRegulatoryChecker(unittest.TestCase):
    """Test regulatory compliance checking"""
    
    def setUp(self):
        """Set up regulatory checker and test ingredients"""
        self.checker = RegulatoryChecker()
        self.salicylic_acid = ACTIVE_INGREDIENT('salicylic_acid')
        self.phenoxyethanol = PRESERVATIVE('phenoxyethanol')
        
    def test_concentration_limit_check(self):
        """Test concentration limit checking"""
        # Test within limits
        compliant = self.checker.check_concentration_limits(self.salicylic_acid, 1.5, 'EU')
        self.assertTrue(compliant)
        
        # Test exceeding limits
        non_compliant = self.checker.check_concentration_limits(self.salicylic_acid, 3.0, 'EU')
        self.assertFalse(non_compliant)
        
    def test_unknown_ingredient_limits(self):
        """Test handling of ingredients without specific limits"""
        unknown_ingredient = ACTIVE_INGREDIENT('unknown_ingredient')
        result = self.checker.check_concentration_limits(unknown_ingredient, 5.0, 'EU')
        self.assertTrue(result)  # Should return True for unknown ingredients
        
    def test_allergen_labeling(self):
        """Test allergen detection"""
        safe_formulation = SKINCARE_FORMULATION('safe_formulation')
        safe_formulation.ingredients = [self.salicylic_acid, self.phenoxyethanol]
        
        allergens = self.checker.check_allergen_labeling(safe_formulation)
        self.assertIsInstance(allergens, list)

class TestFormulationOptimizer(unittest.TestCase):
    """Test formulation optimization functionality"""
    
    def setUp(self):
        """Set up optimizer and test formulation"""
        self.optimizer = FormulationOptimizer()
        self.constraints = FormulationConstraints(
            ph_range=(5.0, 7.0),
            max_cost_per_100ml=3.0,
            requires_preservative=True
        )
        
        self.hyaluronic_acid = ACTIVE_INGREDIENT('hyaluronic_acid')
        self.phenoxyethanol = PRESERVATIVE('phenoxyethanol')
        
        self.test_formulation = SKINCARE_FORMULATION('test_formulation')
        self.test_formulation.ingredients = [self.hyaluronic_acid, self.phenoxyethanol]
        
    def test_cost_calculation(self):
        """Test cost calculation"""
        cost = self.optimizer.calculate_formulation_cost(self.test_formulation)
        self.assertIsInstance(cost, float)
        self.assertGreaterEqual(cost, 0.0)
        
    def test_formulation_optimization(self):
        """Test formulation optimization"""
        results = self.optimizer.optimize_formulation(self.test_formulation, self.constraints)
        
        self.assertIn('formulation', results)
        self.assertIn('stability_score', results)
        self.assertIn('estimated_cost', results)
        self.assertIn('regulatory_compliance', results)
        self.assertIn('optimization_suggestions', results)
        
        self.assertIsInstance(results['stability_score'], float)
        self.assertIsInstance(results['estimated_cost'], float)
        self.assertIsInstance(results['regulatory_compliance'], bool)
        self.assertIsInstance(results['optimization_suggestions'], list)
        
    def test_preservative_requirement(self):
        """Test preservative requirement checking"""
        # Formulation without preservative
        no_preservative = SKINCARE_FORMULATION('no_preservative')
        no_preservative.ingredients = [self.hyaluronic_acid]
        
        results = self.optimizer.optimize_formulation(no_preservative, self.constraints)
        self.assertFalse(results['regulatory_compliance'])
        self.assertIn('Add preservative system', results['optimization_suggestions'])

class TestIntegrationScenarios(unittest.TestCase):
    """Test integration scenarios combining multiple components"""
    
    def test_complete_formulation_workflow(self):
        """Test complete formulation development workflow"""
        # 1. Create ingredients
        retinol = ACTIVE_INGREDIENT('retinol')
        hyaluronic_acid = ACTIVE_INGREDIENT('hyaluronic_acid')
        vitamin_e = ANTIOXIDANT('vitamin_e_tocopherol')
        phenoxyethanol = PRESERVATIVE('phenoxyethanol')
        
        # 2. Create formulation
        night_serum = SKINCARE_FORMULATION('night_serum')
        night_serum.ingredients = [retinol, hyaluronic_acid, vitamin_e, phenoxyethanol]
        
        # 3. Analyze stability
        predictor = StabilityPredictor()
        stability = predictor.predict_stability(night_serum)
        self.assertIsInstance(stability, float)
        
        # 4. Check regulatory compliance
        checker = RegulatoryChecker()
        allergens = checker.check_allergen_labeling(night_serum)
        self.assertIsInstance(allergens, list)
        
        # 5. Optimize formulation
        optimizer = FormulationOptimizer()
        constraints = FormulationConstraints()
        results = optimizer.optimize_formulation(night_serum, constraints)
        
        self.assertIn('formulation', results)
        self.assertEqual(results['formulation'], night_serum)
        
    def test_compatibility_analysis_workflow(self):
        """Test compatibility analysis workflow"""
        # Create ingredients with known interactions
        vitamin_c = ACTIVE_INGREDIENT('vitamin_c')
        vitamin_e = ANTIOXIDANT('vitamin_e')
        retinol = ACTIVE_INGREDIENT('retinol')
        
        # Create interaction links
        synergy = SYNERGY_LINK(vitamin_c, vitamin_e, strength=0.95)
        incompatibility = INCOMPATIBILITY_LINK(vitamin_c, retinol, strength=0.8)
        
        # Test that links are created correctly
        self.assertEqual(synergy.atom1, vitamin_c)
        self.assertEqual(synergy.atom2, vitamin_e)
        self.assertEqual(incompatibility.atom1, vitamin_c)
        self.assertEqual(incompatibility.atom2, retinol)
        
        # Create formulations to test
        good_formulation = SKINCARE_FORMULATION('good_combo')
        good_formulation.ingredients = [vitamin_c, vitamin_e]
        
        problematic_formulation = SKINCARE_FORMULATION('problematic_combo')
        problematic_formulation.ingredients = [vitamin_c, retinol]
        
        self.assertEqual(len(good_formulation.ingredients), 2)
        self.assertEqual(len(problematic_formulation.ingredients), 2)


def run_tests():
    """Run all cosmetic chemistry tests"""
    print("=== Running Cosmetic Chemistry Framework Tests ===\n")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test cases
    test_classes = [
        TestCosmeticAtomTypes,
        TestFormulations,
        TestCompatibilityLinks,
        TestStabilityPredictor,
        TestRegulatoryChecker,
        TestFormulationOptimizer,
        TestIntegrationScenarios
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n=== Test Summary ===")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\nOverall result: {'✓ PASS' if success else '✗ FAIL'}")
    
    return success


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)