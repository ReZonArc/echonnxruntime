#!/usr/bin/env python3
"""
Test Suite for Meta-Optimization Engine

This comprehensive test suite validates the meta-optimization engine's
ability to generate optimal formulations for every possible condition
and treatment combination.

Author: ONNX Runtime Cosmeceutical Optimization Team
"""

import unittest
import sys
import time
import json
import tempfile
import os
from unittest.mock import patch, MagicMock

# Import the meta-optimization engine
from meta_optimization_engine import *


class TestConditionProfile(unittest.TestCase):
    """Test condition profile data structure"""
    
    def test_condition_profile_creation(self):
        """Test creation of condition profiles"""
        profile = ConditionProfile(
            condition_id="test_acne",
            name="Test Acne",
            category="acne",
            target_hypergredient_classes=["H.SE", "H.AI"],
            priority_functions=["sebum_regulation", "anti_inflammatory"]
        )
        
        self.assertEqual(profile.condition_id, "test_acne")
        self.assertEqual(profile.name, "Test Acne")
        self.assertEqual(profile.category, "acne")
        self.assertEqual(len(profile.target_hypergredient_classes), 2)
        self.assertEqual(len(profile.priority_functions), 2)
        self.assertEqual(profile.severity_levels, ["mild", "moderate", "severe"])


class TestTreatmentProtocol(unittest.TestCase):
    """Test treatment protocol data structure"""
    
    def test_treatment_protocol_creation(self):
        """Test creation of treatment protocols"""
        protocol = TreatmentProtocol(
            protocol_id="test_protocol",
            name="Test Protocol",
            target_conditions=["acne", "inflammation"],
            expected_timeline_weeks=12
        )
        
        self.assertEqual(protocol.protocol_id, "test_protocol")
        self.assertEqual(protocol.name, "Test Protocol")
        self.assertEqual(len(protocol.target_conditions), 2)
        self.assertEqual(protocol.expected_timeline_weeks, 12)
        self.assertTrue(protocol.maintenance_required)


class TestOptimizationSpace(unittest.TestCase):
    """Test optimization space initialization"""
    
    def setUp(self):
        """Set up test optimization space"""
        self.engine = MetaOptimizationEngine()
        self.space = self.engine.optimization_space
    
    def test_optimization_space_initialization(self):
        """Test that optimization space initializes correctly"""
        self.assertGreater(len(self.space.conditions), 5)
        self.assertGreater(len(self.space.treatments), 3)
        self.assertGreater(len(self.space.skin_types), 3)
        self.assertGreater(len(self.space.budgets), 3)
    
    def test_condition_categories(self):
        """Test that conditions cover major categories"""
        categories = set(condition.category for condition in self.space.conditions)
        expected_categories = {"acne", "aging", "pigmentation", "sensitivity", "barrier", "hydration"}
        
        for category in expected_categories:
            self.assertIn(category, categories)
    
    def test_treatment_protocols_coverage(self):
        """Test that treatment protocols cover major use cases"""
        protocol_names = [protocol.name for protocol in self.space.treatments]
        
        # Should have protocols for different treatment types
        self.assertTrue(any("maintenance" in name.lower() for name in protocol_names))
        self.assertTrue(any("acne" in name.lower() for name in protocol_names))
        self.assertTrue(any("sensitivity" in name.lower() for name in protocol_names))


class TestMetaOptimizationEngine(unittest.TestCase):
    """Test meta-optimization engine core functionality"""
    
    def setUp(self):
        """Set up test engine"""
        self.engine = MetaOptimizationEngine()
        # Reduce concurrency for testing
        self.engine.max_concurrent_optimizations = 2
    
    def test_engine_initialization(self):
        """Test that engine initializes correctly"""
        self.assertIsInstance(self.engine.formulator, HypergredientFormulator)
        self.assertIsInstance(self.engine.optimization_space, OptimizationSpace)
        self.assertEqual(len(self.engine.formulation_matrix), 0)
        self.assertEqual(self.engine.optimization_stats['formulations_computed'], 0)
    
    def test_combination_key_generation(self):
        """Test generation of unique combination keys"""
        key = self.engine._generate_combination_key(
            conditions=["acne_comedonal"],
            treatment_id="acne_clearing",
            skin_type="oily",
            age_range="twenties",
            sensitivity="none",
            budget=1500.0,
            preferences=["gentle"]
        )
        
        expected = "acne_comedonal|acne_clearing|oily|twenties|none|1500|gentle"
        self.assertEqual(key, expected)
    
    def test_combination_key_sorting(self):
        """Test that combination keys are consistently sorted"""
        key1 = self.engine._generate_combination_key(
            conditions=["acne_comedonal", "aging_photoaging"],
            treatment_id="test",
            skin_type="normal",
            age_range="thirties",
            sensitivity="none", 
            budget=1000.0,
            preferences=["gentle", "stable"]
        )
        
        key2 = self.engine._generate_combination_key(
            conditions=["aging_photoaging", "acne_comedonal"],
            treatment_id="test",
            skin_type="normal",
            age_range="thirties",
            sensitivity="none",
            budget=1000.0,
            preferences=["stable", "gentle"]
        )
        
        self.assertEqual(key1, key2)  # Should be identical due to sorting
    
    def test_total_combinations_calculation(self):
        """Test calculation of total combinations"""
        total = self.engine._calculate_total_combinations()
        self.assertGreater(total, 1000)  # Should be substantial number
        self.assertEqual(total, self.engine.optimization_stats['total_combinations'])
    
    def test_conditions_to_concerns_mapping(self):
        """Test conversion from conditions to formulation concerns"""
        concerns = self.engine._conditions_to_concerns(["acne_comedonal", "aging_photoaging"])
        
        self.assertIn("acne", concerns)
        self.assertIn("wrinkles", concerns)
        self.assertGreater(len(concerns), 2)
    
    def test_budget_adjustment_for_demographics(self):
        """Test budget adjustment based on demographics"""
        base_budget = 1000.0
        
        # Test age-based adjustments
        teen_budget = self.engine._adjust_budget_for_demographics(base_budget, "teens", "none")
        self.assertLess(teen_budget, base_budget)
        
        mature_budget = self.engine._adjust_budget_for_demographics(base_budget, "fifties_plus", "none")
        self.assertGreater(mature_budget, base_budget)
        
        # Test sensitivity-based adjustments
        sensitive_budget = self.engine._adjust_budget_for_demographics(base_budget, "thirties", "severe")
        self.assertGreater(sensitive_budget, base_budget)
    
    def test_preference_adjustment(self):
        """Test preference adjustment based on demographics"""
        base_prefs = ["effective"]
        
        # Sensitive skin should add gentle
        adjusted = self.engine._adjust_preferences(base_prefs, "moderate", "twenties")
        self.assertIn("gentle", adjusted)
        
        # Mature skin should add stable
        adjusted = self.engine._adjust_preferences(base_prefs, "none", "forties")
        self.assertIn("stable", adjusted)
    
    def test_contraindicated_ingredients(self):
        """Test generation of contraindicated ingredients list"""
        contraindicated = self.engine._get_contraindicated_ingredients(
            ["sensitivity_general"], "severe"
        )
        
        # Should include harsh ingredients for sensitive skin
        self.assertGreater(len(contraindicated), 0)
        # Common harsh ingredients should be excluded
        harsh_ingredients = ["tretinoin", "glycolic_acid", "salicylic_acid"]
        self.assertTrue(any(ing in contraindicated for ing in harsh_ingredients))


class TestSingleOptimization(unittest.TestCase):
    """Test single combination optimization"""
    
    def setUp(self):
        """Set up test engine"""
        self.engine = MetaOptimizationEngine()
    
    def test_single_combination_optimization(self):
        """Test optimization of a single parameter combination"""
        # Create a test combination key
        key = "acne_comedonal|acne_clearing|oily|twenties|none|1500|gentle"
        
        result = self.engine._optimize_single_combination(key)
        
        self.assertIsInstance(result, PrecomputedFormulation)
        self.assertEqual(result.optimization_key, key)
        self.assertIn("acne_comedonal", result.conditions_addressed)
        self.assertGreater(result.confidence_score, 0.0)
        self.assertLessEqual(result.confidence_score, 1.0)
        self.assertGreater(result.computation_time, 0.0)
    
    def test_confidence_score_calculation(self):
        """Test confidence score calculation"""
        # Create a mock formulation
        formulation = OptimalFormulation(
            selected_hypergredients={"H.SE": {"ingredient": "test", "percentage": 2.0}},
            total_score=0.85,
            predicted_efficacy=0.9,
            stability_months=12,
            cost_per_50ml=800.0,
            safety_profile="Excellent - suitable for sensitive skin",
            synergy_score=7.5
        )
        
        confidence = self.engine._calculate_confidence_score(
            formulation, ["acne_comedonal"], "acne_clearing"
        )
        
        self.assertGreater(confidence, 0.8)  # Should be high due to good metrics
        self.assertLessEqual(confidence, 1.0)


class TestFormulationRetrieval(unittest.TestCase):
    """Test formulation retrieval and caching"""
    
    def setUp(self):
        """Set up test engine with sample formulations"""
        self.engine = MetaOptimizationEngine()
        
        # Add a sample pre-computed formulation
        sample_formulation = PrecomputedFormulation(
            formulation=OptimalFormulation(
                selected_hypergredients={"H.SE": {"ingredient": "niacinamide", "percentage": 3.0}},
                total_score=0.8,
                predicted_efficacy=0.75,
                stability_months=18,
                cost_per_50ml=900.0,
                safety_profile="Good - suitable for most skin types",
                synergy_score=6.5
            ),
            optimization_key="acne_comedonal|acne_clearing|oily|twenties|none|1500|gentle",
            conditions_addressed=["acne_comedonal"],
            confidence_score=0.82,
            computation_time=0.15,
            last_updated=time.time()
        )
        
        self.engine.formulation_matrix[sample_formulation.optimization_key] = sample_formulation
    
    def test_exact_formulation_retrieval(self):
        """Test retrieval of exact pre-computed formulation"""
        result = self.engine.get_optimal_formulation(
            conditions=["acne_comedonal"],
            treatment_protocol="acne_clearing",
            skin_type="oily",
            age_range="twenties",
            sensitivity="none",
            budget=1500.0,
            preferences=["gentle"]
        )
        
        self.assertIsNotNone(result)
        self.assertEqual(result.conditions_addressed, ["acne_comedonal"])
        self.assertGreater(result.confidence_score, 0.8)
    
    def test_key_similarity_calculation(self):
        """Test similarity calculation between combination keys"""
        key1 = "acne_comedonal|acne_clearing|oily|twenties|none|1500|gentle"
        key2 = "acne_comedonal|acne_clearing|combination|twenties|none|1500|gentle"  # Similar but different skin type
        key3 = "aging_photoaging|corrective_intensive|dry|forties|mild|2500|stable"  # Very different
        
        similarity_similar = self.engine._calculate_key_similarity(key1, key2)
        similarity_different = self.engine._calculate_key_similarity(key1, key3)
        
        self.assertGreater(similarity_similar, 0.8)  # Should be high similarity
        self.assertLess(similarity_different, 0.3)   # Should be low similarity
    
    def test_on_demand_optimization(self):
        """Test on-demand optimization for uncached combinations"""
        # Request a combination that's not pre-computed
        result = self.engine.get_optimal_formulation(
            conditions=["dehydration"],
            treatment_protocol="barrier_restoration", 
            skin_type="dry",
            sensitivity="mild"
        )
        
        self.assertIsNotNone(result)
        self.assertIn("dehydration", result.conditions_addressed)


class TestSmallScaleMetaOptimization(unittest.TestCase):
    """Test meta-optimization with a small subset"""
    
    def setUp(self):
        """Set up engine with limited optimization space for testing"""
        self.engine = MetaOptimizationEngine()
        
        # Reduce optimization space for faster testing
        limited_space = OptimizationSpace(
            conditions=self.engine.optimization_space.conditions[:3],  # Only first 3 conditions
            treatments=self.engine.optimization_space.treatments[:2],   # Only first 2 treatments
            skin_types=["oily", "dry"],
            age_ranges=["twenties", "thirties"],
            sensitivities=["none", "mild"],
            budgets=[1000.0, 1500.0],
            preferences=[["gentle"], ["stable"]]
        )
        
        self.engine.optimization_space = limited_space
        self.engine.max_concurrent_optimizations = 1  # Single threaded for testing
    
    def test_small_scale_matrix_generation(self):
        """Test generation of formulation matrix with limited scope"""
        # Calculate expected combinations for our limited space
        expected_combinations = self.engine._calculate_total_combinations()
        self.assertLess(expected_combinations, 1000)  # Should be manageable for testing
        
        # Generate first batch of combinations only
        combination_keys = self.engine._generate_all_combination_keys()[:20]  # Just test 20 combinations
        
        formulations_computed = 0
        for key in combination_keys:
            result = self.engine._optimize_single_combination(key)
            if result:
                self.engine.formulation_matrix[key] = result
                formulations_computed += 1
        
        self.assertGreater(formulations_computed, 15)  # Should compute most successfully
        self.assertEqual(len(self.engine.formulation_matrix), formulations_computed)


class TestPerformanceBenchmarking(unittest.TestCase):
    """Test performance benchmarking functionality"""
    
    def setUp(self):
        """Set up engine with sample formulations for benchmarking"""
        self.engine = MetaOptimizationEngine()
        
        # Add several sample formulations for benchmarking
        for i in range(10):
            sample_formulation = PrecomputedFormulation(
                formulation=OptimalFormulation(
                    selected_hypergredients={"H.SE": {"ingredient": f"test_ingredient_{i}", "percentage": 2.0}},
                    total_score=0.8,
                    predicted_efficacy=0.75,
                    stability_months=12,
                    cost_per_50ml=1000.0,
                    safety_profile="Good",
                    synergy_score=6.0
                ),
                optimization_key=f"test_condition_{i}|test_treatment|normal|thirties|none|1500|stable",
                conditions_addressed=[f"test_condition_{i}"],
                confidence_score=0.8,
                computation_time=0.1,
                last_updated=time.time()
            )
            
            self.engine.formulation_matrix[sample_formulation.optimization_key] = sample_formulation
    
    def test_benchmark_execution(self):
        """Test that benchmark runs and returns valid metrics"""
        initial_cache_size = len(self.engine.formulation_matrix)
        benchmark_results = self.engine.benchmark_performance()
        
        # Validate benchmark results structure
        required_keys = [
            'cache_size', 'average_retrieval_time_ms', 'max_retrieval_time_ms',
            'average_on_demand_time_ms', 'speedup_factor', 'memory_efficiency_mb',
            'optimization_stats'
        ]
        
        for key in required_keys:
            self.assertIn(key, benchmark_results)
        
        # Validate benchmark values
        self.assertGreaterEqual(benchmark_results['cache_size'], initial_cache_size)  # Should be at least what we started with
        self.assertGreaterEqual(benchmark_results['average_retrieval_time_ms'], 0)
        self.assertGreaterEqual(benchmark_results['speedup_factor'], 1)  # Cache should be at least as fast as on-demand


class TestDataExportImport(unittest.TestCase):
    """Test formulation matrix export functionality"""
    
    def setUp(self):
        """Set up engine with sample data"""
        self.engine = MetaOptimizationEngine()
        
        # Add sample formulation
        sample_formulation = PrecomputedFormulation(
            formulation=OptimalFormulation(
                selected_hypergredients={"H.AO": {"ingredient": "vitamin_c", "percentage": 15.0}},
                total_score=0.9,
                predicted_efficacy=0.85,
                stability_months=6,
                cost_per_50ml=1200.0,
                safety_profile="Excellent",
                synergy_score=8.0
            ),
            optimization_key="aging_photoaging|corrective_intensive|normal|forties|none|2000|stable",
            conditions_addressed=["aging_photoaging"],
            confidence_score=0.88,
            computation_time=0.2,
            last_updated=time.time()
        )
        
        self.engine.formulation_matrix[sample_formulation.optimization_key] = sample_formulation
    
    def test_export_formulation_matrix(self):
        """Test export of formulation matrix to JSON file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Export the matrix
            self.engine.export_formulation_matrix(temp_path)
            
            # Verify file was created and contains expected data
            self.assertTrue(os.path.exists(temp_path))
            
            with open(temp_path, 'r') as f:
                exported_data = json.load(f)
            
            # Validate export structure
            self.assertIn('metadata', exported_data)
            self.assertIn('formulations', exported_data)
            self.assertEqual(exported_data['metadata']['total_formulations'], 1)
            
            # Validate formulation data
            formulation_keys = list(exported_data['formulations'].keys())
            self.assertEqual(len(formulation_keys), 1)
            
            exported_formulation = exported_data['formulations'][formulation_keys[0]]
            self.assertIn('optimization_key', exported_formulation)
            self.assertIn('conditions_addressed', exported_formulation)
            self.assertIn('formulation', exported_formulation)
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestIntegrationScenarios(unittest.TestCase):
    """Test complete integration scenarios"""
    
    def setUp(self):
        """Set up engine for integration testing"""
        self.engine = MetaOptimizationEngine()
    
    def test_complete_acne_treatment_workflow(self):
        """Test complete workflow for acne treatment optimization"""
        # Request formulation for teenage acne
        result = self.engine.get_optimal_formulation(
            conditions=["acne_comedonal", "acne_inflammatory"],
            treatment_protocol="acne_clearing",
            skin_type="oily",
            age_range="teens",
            sensitivity="none",
            budget=800.0,  # Lower budget for teens
            preferences=["gentle"]
        )
        
        self.assertIsNotNone(result)
        self.assertTrue(any("acne" in condition for condition in result.conditions_addressed))
        self.assertGreater(result.confidence_score, 0.5)
        
        # Formulation should be within budget (adjusted for demographics)
        self.assertLessEqual(result.formulation.cost_per_50ml, 800.0 * 1.1)  # Allow small buffer
    
    def test_complete_anti_aging_workflow(self):
        """Test complete workflow for anti-aging treatment"""
        result = self.engine.get_optimal_formulation(
            conditions=["aging_photoaging", "hyperpigmentation_pih"],
            treatment_protocol="corrective_intensive",
            skin_type="combination",
            age_range="forties",
            sensitivity="mild",
            budget=2500.0,
            preferences=["effective", "stable"]
        )
        
        self.assertIsNotNone(result)
        self.assertTrue(any("aging" in condition for condition in result.conditions_addressed))
        self.assertGreater(result.confidence_score, 0.6)
        
        # Should have multiple active ingredients for complex concerns
        self.assertGreater(len(result.formulation.selected_hypergredients), 2)
    
    def test_sensitive_skin_formulation_safety(self):
        """Test that sensitive skin formulations prioritize safety"""
        result = self.engine.get_optimal_formulation(
            conditions=["sensitivity_general", "barrier_disruption"],
            treatment_protocol="sensitivity_soothing",
            skin_type="sensitive",
            age_range="thirties",
            sensitivity="severe",
            preferences=["gentle"]
        )
        
        self.assertIsNotNone(result)
        
        # Safety profile should be excellent for severe sensitivity
        self.assertTrue(result.formulation.safety_profile.startswith("Excellent") or 
                       result.formulation.safety_profile.startswith("Good"))
        
        # Should have low percentage of active ingredients
        total_actives = sum(data.get('percentage', 0) for data in result.formulation.selected_hypergredients.values())
        self.assertLess(total_actives, 10.0)  # Conservative approach for sensitive skin


if __name__ == '__main__':
    # Create test suite with different verbosity levels
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    
    print("🧪 Running Meta-Optimization Engine Test Suite")
    print("=" * 50)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n📊 Test Results Summary:")
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success Rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%" if result.testsRun > 0 else "N/A")
    
    if result.failures:
        print(f"\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError: ')[-1].split('\\n')[0] if 'AssertionError:' in traceback else 'Unknown error'}")
    
    if result.errors:
        print(f"\n⚠️ Errors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('\\n')[-2] if '\\n' in traceback else traceback}")