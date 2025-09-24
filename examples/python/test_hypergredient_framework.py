#!/usr/bin/env python3
"""
Test Suite for Hypergredient Framework

This test suite validates the hypergredient framework functionality
including database operations, optimization algorithms, and compatibility checking.

Author: ONNX Runtime Cosmeceutical Optimization Team
"""

import unittest
import sys
import os

# Import the hypergredient framework
from hypergredient_framework import *


class TestHypergredientDatabase(unittest.TestCase):
    """Test hypergredient database functionality"""
    
    def setUp(self):
        """Set up test database"""
        self.database = HypergredientDatabase()
    
    def test_database_initialization(self):
        """Test that database initializes with expected ingredients"""
        self.assertGreater(len(self.database.ingredients), 10)
        self.assertIn('tretinoin', self.database.ingredients)
        self.assertIn('hyaluronic_acid', self.database.ingredients)
        
        # Test hypergredient classes are represented
        classes_found = set()
        for ingredient in self.database.ingredients.values():
            classes_found.add(ingredient.hypergredient_class)
        
        expected_classes = ['H.CT', 'H.CS', 'H.AO', 'H.HY', 'H.BR']
        for hg_class in expected_classes:
            self.assertIn(hg_class, classes_found)
    
    def test_get_by_class(self):
        """Test filtering ingredients by hypergredient class"""
        ct_ingredients = self.database.get_by_class('H.CT')
        self.assertGreater(len(ct_ingredients), 0)
        
        for ingredient in ct_ingredients:
            self.assertEqual(ingredient.hypergredient_class, 'H.CT')
    
    def test_get_by_function(self):
        """Test filtering ingredients by function"""
        hydration_ingredients = self.database.get_by_function('hydration')
        self.assertGreater(len(hydration_ingredients), 0)
        
        for ingredient in hydration_ingredients:
            self.assertTrue(
                ingredient.primary_function == 'hydration' or
                'hydration' in ingredient.secondary_functions
            )
    
    def test_ingredient_score_calculation(self):
        """Test ingredient scoring algorithm"""
        tretinoin = self.database.ingredients['tretinoin']
        
        # Test with default weights
        score = self.database.calculate_ingredient_score(
            tretinoin, 
            {'efficacy': 0.35, 'safety': 0.25, 'stability': 0.20, 'cost': 0.15, 'synergy': 0.05}
        )
        
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.5)  # Allow for synergy bonuses
    
    def test_synergy_bonus(self):
        """Test that synergy bonuses are calculated correctly"""
        vitamin_c = self.database.ingredients['vitamin_c_l_aa']
        
        # Score without synergies
        score_alone = self.database.calculate_ingredient_score(
            vitamin_c,
            {'efficacy': 0.35, 'safety': 0.25, 'stability': 0.20, 'cost': 0.15, 'synergy': 0.05}
        )
        
        # Score with synergistic ingredient in context
        score_with_synergy = self.database.calculate_ingredient_score(
            vitamin_c,
            {'efficacy': 0.35, 'safety': 0.25, 'stability': 0.20, 'cost': 0.15, 'synergy': 0.05},
            formulation_context=['vitamin_e']  # Vitamin C synergizes with Vitamin E
        )
        
        self.assertGreater(score_with_synergy, score_alone)
    
    def test_incompatibility_penalty(self):
        """Test that incompatibility penalties are applied"""
        vitamin_c = self.database.ingredients['vitamin_c_l_aa']
        
        # Score without incompatibilities
        score_alone = self.database.calculate_ingredient_score(
            vitamin_c,
            {'efficacy': 0.35, 'safety': 0.25, 'stability': 0.20, 'cost': 0.15, 'synergy': 0.05}
        )
        
        # Score with incompatible ingredient
        score_with_incompatibility = self.database.calculate_ingredient_score(
            vitamin_c,
            {'efficacy': 0.35, 'safety': 0.25, 'stability': 0.20, 'cost': 0.15, 'synergy': 0.05},
            formulation_context=['copper_peptides']  # Incompatible with Vitamin C
        )
        
        self.assertLess(score_with_incompatibility, score_alone)


class TestHypergredientFormulator(unittest.TestCase):
    """Test formulation optimization functionality"""
    
    def setUp(self):
        """Set up test formulator"""
        self.formulator = HypergredientFormulator()
    
    def test_concern_mapping(self):
        """Test mapping of skin concerns to hypergredient classes"""
        wrinkle_classes = self.formulator.map_concern_to_hypergredient('wrinkles')
        self.assertIn('H.CT', wrinkle_classes)
        self.assertIn('H.CS', wrinkle_classes)
        
        dryness_classes = self.formulator.map_concern_to_hypergredient('dryness')
        self.assertIn('H.HY', dryness_classes)
        self.assertIn('H.BR', dryness_classes)
    
    def test_formulation_optimization(self):
        """Test complete formulation optimization process"""
        request = FormulationRequest(
            target_concerns=['wrinkles', 'dryness'],
            skin_type='sensitive',
            budget=1000.0,
            preferences=['gentle']
        )
        
        result = self.formulator.optimize_formulation(request)
        
        # Validate result structure
        self.assertIsInstance(result, OptimalFormulation)
        self.assertGreater(len(result.selected_hypergredients), 0)
        self.assertIsInstance(result.total_score, float)
        self.assertIsInstance(result.predicted_efficacy, float)
        self.assertIsInstance(result.stability_months, int)
        self.assertIsInstance(result.cost_per_50ml, float)
        
        # Validate score ranges
        self.assertGreaterEqual(result.total_score, 0.0)
        self.assertLessEqual(result.total_score, 1.0)
        self.assertGreaterEqual(result.predicted_efficacy, 0.0)
        self.assertLessEqual(result.predicted_efficacy, 1.0)
        self.assertGreater(result.stability_months, 0)
        self.assertGreater(result.cost_per_50ml, 0.0)
    
    def test_ingredient_exclusion(self):
        """Test that excluded ingredients are not selected"""
        request = FormulationRequest(
            target_concerns=['wrinkles'],
            exclude_ingredients=['tretinoin']
        )
        
        result = self.formulator.optimize_formulation(request)
        
        # Check that tretinoin is not in the selected ingredients
        selected_ingredients = [hg['selection'] for hg in result.selected_hypergredients.values()]
        self.assertNotIn('Tretinoin', selected_ingredients)
    
    def test_budget_consideration(self):
        """Test that formulation considers budget constraints"""
        low_budget_request = FormulationRequest(
            target_concerns=['hydration'],
            budget=200.0  # Very low budget
        )
        
        high_budget_request = FormulationRequest(
            target_concerns=['hydration'],
            budget=5000.0  # High budget
        )
        
        low_result = self.formulator.optimize_formulation(low_budget_request)
        high_result = self.formulator.optimize_formulation(high_budget_request)
        
        # Both should complete successfully
        self.assertIsInstance(low_result, OptimalFormulation)
        self.assertIsInstance(high_result, OptimalFormulation)
    
    def test_optimal_percentage_calculation(self):
        """Test optimal percentage calculation"""
        vitamin_c = self.formulator.database.ingredients['vitamin_c_l_aa']
        
        gentle_request = FormulationRequest(
            target_concerns=['brightening'],
            preferences=['gentle']
        )
        
        potent_request = FormulationRequest(
            target_concerns=['brightening'],
            preferences=['potent']
        )
        
        gentle_percentage = self.formulator._calculate_optimal_percentage(vitamin_c, gentle_request)
        potent_percentage = self.formulator._calculate_optimal_percentage(vitamin_c, potent_request)
        
        # Potent should be higher than gentle
        self.assertGreater(potent_percentage, gentle_percentage)
        
        # Both should be reasonable percentages
        self.assertGreater(gentle_percentage, 0.0)
        self.assertLess(gentle_percentage, 10.0)
        self.assertGreater(potent_percentage, 0.0)
        self.assertLess(potent_percentage, 10.0)


class TestCompatibilityChecker(unittest.TestCase):
    """Test ingredient compatibility checking"""
    
    def setUp(self):
        """Set up test ingredients"""
        self.database = HypergredientDatabase()
        self.retinol = self.database.ingredients['retinol']
        self.vitamin_c = self.database.ingredients['vitamin_c_l_aa']
        self.bakuchiol = self.database.ingredients['bakuchiol']
        self.niacinamide = self.database.ingredients['niacinamide']
    
    def test_compatibility_scoring(self):
        """Test compatibility score calculation"""
        # Test incompatible pair (Vitamin C and Copper Peptides)
        vitamin_c = self.database.ingredients['vitamin_c_l_aa']
        copper_peptides = self.database.ingredients['copper_peptides']
        
        result = check_compatibility(vitamin_c, copper_peptides)
        
        self.assertIsInstance(result, dict)
        self.assertIn('score', result)
        self.assertIn('recommendations', result)
        self.assertLess(result['score'], 1.0)  # Should be low compatibility
    
    def test_synergistic_combination(self):
        """Test synergistic ingredient combinations"""
        # Test Vitamin C + Vitamin E (known synergy)
        vitamin_c = self.database.ingredients['vitamin_c_l_aa']
        vitamin_e = self.database.ingredients['vitamin_e']
        
        result = check_compatibility(vitamin_c, vitamin_e)
        
        self.assertGreater(result['score'], 1.0)  # Should be high compatibility
        self.assertTrue(result['synergistic'])
    
    def test_ph_compatibility(self):
        """Test pH compatibility checking"""
        result = check_compatibility(self.retinol, self.vitamin_c)
        
        self.assertIn('ph_overlap', result)
        self.assertIn('ph_range', result)
        
        # Retinol (pH 5.5-6.5) and Vitamin C (pH 3.0-4.0) should not overlap
        self.assertFalse(result['ph_overlap'])
    
    def test_recommendation_generation(self):
        """Test that appropriate recommendations are generated"""
        # Test incompatible combination
        incompatible_result = check_compatibility(self.vitamin_c, self.database.ingredients['copper_peptides'])
        
        self.assertIsInstance(incompatible_result['recommendations'], list)
        self.assertGreater(len(incompatible_result['recommendations']), 0)
        
        # Should contain warnings for incompatible combinations
        recommendations_text = ' '.join(incompatible_result['recommendations'])
        self.assertTrue(any(word in recommendations_text.lower() 
                           for word in ['avoid', 'caution', 'separate']))


class TestHypergredientProperties(unittest.TestCase):
    """Test hypergredient properties and data structures"""
    
    def test_hypergredient_properties_creation(self):
        """Test creation of hypergredient properties"""
        props = HypergredientProperties(
            name="Test Ingredient",
            inci_name="Test INCI",
            hypergredient_class="H.CT",
            primary_function="test_function",
            efficacy_score=8.0,
            safety_score=9.0,
            cost_per_gram=100.0
        )
        
        self.assertEqual(props.name, "Test Ingredient")
        self.assertEqual(props.hypergredient_class, "H.CT")
        self.assertEqual(props.efficacy_score, 8.0)
        self.assertEqual(props.safety_score, 9.0)
        self.assertIsInstance(props.secondary_functions, list)
        self.assertIsInstance(props.synergies, list)
    
    def test_formulation_request_creation(self):
        """Test formulation request data structure"""
        request = FormulationRequest(
            target_concerns=['wrinkles', 'dryness'],
            skin_type='sensitive',
            budget=1500.0,
            preferences=['gentle', 'stable']
        )
        
        self.assertEqual(len(request.target_concerns), 2)
        self.assertEqual(request.skin_type, 'sensitive')
        self.assertEqual(request.budget, 1500.0)
        self.assertIn('gentle', request.preferences)


class TestAdvancedFeatures(unittest.TestCase):
    """Test advanced hypergredient framework features"""
    
    def setUp(self):
        """Set up test components"""
        self.formulator = HypergredientFormulator()
    
    def test_multi_objective_optimization(self):
        """Test that multiple objectives are balanced"""
        # Create requests with different objective priorities
        efficacy_focused = FormulationRequest(
            target_concerns=['anti_aging'],
            preferences=['potent']
        )
        
        safety_focused = FormulationRequest(
            target_concerns=['anti_aging'],
            preferences=['gentle']
        )
        
        efficacy_result = self.formulator.optimize_formulation(efficacy_focused)
        safety_result = self.formulator.optimize_formulation(safety_focused)
        
        # Both should be valid results
        self.assertIsInstance(efficacy_result, OptimalFormulation)
        self.assertIsInstance(safety_result, OptimalFormulation)
    
    def test_synergy_score_calculation(self):
        """Test synergy score calculation for formulations"""
        # Create a formulation with known synergistic ingredients
        selected_hg = {
            'H.CS': {
                'ingredient': self.formulator.database.ingredients['vitamin_c_l_aa'],
                'selection': 'Vitamin C'
            },
            'H.AO': {
                'ingredient': self.formulator.database.ingredients['vitamin_e'],
                'selection': 'Vitamin E'
            }
        }
        
        synergy_score = self.formulator._calculate_synergy_score(selected_hg)
        
        self.assertIsInstance(synergy_score, float)
        self.assertGreaterEqual(synergy_score, 0.0)
        self.assertLessEqual(synergy_score, 10.0)
    
    def test_stability_estimation(self):
        """Test stability estimation based on ingredient properties"""
        # Create formulation with stable ingredients
        stable_hg = {
            'H.HY': {
                'ingredient': self.formulator.database.ingredients['hyaluronic_acid']
            }
        }
        
        # Create formulation with unstable ingredients
        unstable_hg = {
            'H.CT': {
                'ingredient': self.formulator.database.ingredients['retinol']  # Oxygen sensitive
            }
        }
        
        stable_months = self.formulator._estimate_stability(stable_hg)
        unstable_months = self.formulator._estimate_stability(unstable_hg)
        
        self.assertGreater(stable_months, unstable_months)
        self.assertGreater(stable_months, 0)
        self.assertGreater(unstable_months, 0)
    
    def test_safety_profile_assessment(self):
        """Test safety profile assessment"""
        # High-safety ingredients
        safe_hg = {
            'H.HY': {
                'ingredient': self.formulator.database.ingredients['hyaluronic_acid']  # Safety score 10
            }
        }
        
        # Lower-safety ingredients  
        caution_hg = {
            'H.CT': {
                'ingredient': self.formulator.database.ingredients['tretinoin']  # Safety score 6
            }
        }
        
        safe_profile = self.formulator._assess_safety_profile(safe_hg)
        caution_profile = self.formulator._assess_safety_profile(caution_hg)
        
        self.assertIsInstance(safe_profile, str)
        self.assertIsInstance(caution_profile, str)
        self.assertNotEqual(safe_profile, caution_profile)


class TestIntegrationScenarios(unittest.TestCase):
    """Test complete integration scenarios"""
    
    def test_complete_anti_aging_workflow(self):
        """Test complete anti-aging formulation workflow"""
        formulator = HypergredientFormulator()
        
        request = FormulationRequest(
            target_concerns=['wrinkles', 'firmness', 'brightness'],
            skin_type='normal',
            budget=2000.0,
            preferences=['stable']
        )
        
        result = formulator.optimize_formulation(request)
        
        # Validate complete result
        self.assertIsInstance(result, OptimalFormulation)
        self.assertGreater(len(result.selected_hypergredients), 2)  # Should address multiple concerns
        self.assertGreater(result.total_score, 0.5)  # Reasonable quality
        self.assertGreater(result.predicted_efficacy, 0.4)  # Reasonable efficacy
        self.assertLess(result.cost_per_50ml, request.budget)  # Within budget
    
    def test_sensitive_skin_formulation(self):
        """Test formulation for sensitive skin requirements"""
        formulator = HypergredientFormulator()
        
        request = FormulationRequest(
            target_concerns=['hydration', 'sensitivity'],
            skin_type='sensitive',
            preferences=['gentle']
        )
        
        result = formulator.optimize_formulation(request)
        
        # Should prioritize safety for sensitive skin
        avg_safety = 0.0
        for hg_data in result.selected_hypergredients.values():
            avg_safety += hg_data['ingredient'].safety_score
        avg_safety /= len(result.selected_hypergredients)
        
        self.assertGreater(avg_safety, 7.0)  # High safety requirement
    
    def test_budget_constrained_formulation(self):
        """Test formulation under budget constraints"""
        formulator = HypergredientFormulator()
        
        request = FormulationRequest(
            target_concerns=['hydration'],
            budget=300.0,  # Low budget
            preferences=['cost-effective']
        )
        
        result = formulator.optimize_formulation(request)
        
        # Should complete successfully even with low budget
        self.assertIsInstance(result, OptimalFormulation)
        self.assertGreater(len(result.selected_hypergredients), 0)
        
        # Cost should be reasonable
        self.assertLess(result.cost_per_50ml, 200.0)  # Should find cost-effective options


if __name__ == '__main__':
    # Set up test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestHypergredientDatabase,
        TestHypergredientFormulator,
        TestCompatibilityChecker,
        TestHypergredientProperties,
        TestAdvancedFeatures,
        TestIntegrationScenarios
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"HYPERGREDIENT FRAMEWORK TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print(f"\nFAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print(f"\nERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    if result.wasSuccessful():
        print(f"\n🎉 All tests passed! Hypergredient Framework is working correctly.")
    else:
        print(f"\n❌ Some tests failed. Please review the issues above.")