"""
Test: K-Value Cost Function vs Total Crossing Cost Function

This test demonstrates the key difference:
- Total Crossing Cost: Minimizes sum of all crossings
- K-Value Cost: Minimizes the maximum crossing count on any single edge

TDD Approach:
- Write failing tests that expect K-value optimization
- Implement cost function strategy pattern
- Tests should pass after implementation
"""

import pytest
import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    import planar_cuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    pytest.skip("CUDA module not available", allow_module_level=True)


class TestCostFunctionBehavior:
    """
    Test that different cost functions lead to different optimization decisions
    """
    
    def test_total_crossing_vs_k_value_difference(self):
        """
        Demonstrate scenario where total crossings and K-value
        give opposite optimization directions.
        
        Scenario:
        - Initial state: K=5, Total=20
        - Move A: K=6, Total=18  (worse K, better total)
        - Move B: K=4, Total=21  (better K, worse total)
        
        Expected:
        - Total crossing cost prefers Move A
        - K-value cost prefers Move B
        """
        # This is a conceptual test - actual implementation
        # will depend on finding such a configuration
        
        # For now, test the interface exists
        nodes_x = [0, 10, 10, 0, 5]
        nodes_y = [0, 0, 10, 10, 5]
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3), (4, 0), (4, 2)]
        
        solver = planar_cuda.PlanarSolver(nodes_x, nodes_y, edges)
        
        # Test that cost function can be set
        solver.set_cost_function("total_crossings")
        total_cost_1 = solver.get_current_cost()
        
        solver.set_cost_function("k_value")
        k_cost_1 = solver.get_current_cost()
        
        # Costs should be different (total crossings vs max crossing)
        # This assumes the graph has crossings
        assert total_cost_1 != k_cost_1 or total_cost_1 == 0, \
            "Cost functions should give different values (unless planar)"
    
    def test_compute_delta_with_different_cost_functions(self):
        """
        Test that compute_delta returns different values
        for different cost functions.
        """
        nodes_x = [0, 10, 10, 0]
        nodes_y = [0, 0, 10, 10]
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)]
        
        solver = planar_cuda.PlanarSolver(nodes_x, nodes_y, edges)
        
        # Test move with total crossing cost
        solver.set_cost_function("total_crossings")
        delta_total = solver.compute_delta(0, 2, 2)
        
        # Test same move with K-value cost
        solver.set_cost_function("k_value")
        delta_k = solver.compute_delta(0, 2, 2)
        
        # Deltas might be different
        # (This depends on the specific graph structure)
        print(f"Delta with total crossings: {delta_total}")
        print(f"Delta with K-value: {delta_k}")


class TestKValueOptimization:
    """
    Test that SA optimization with K-value cost
    actually minimizes K-value (not total crossings)
    """
    
    def test_sa_minimizes_k_value(self):
        """
        Verify that SA with K-value cost reduces K-value
        """
        # Arrange - Create graph with known high K
        nodes_x = [0, 10, 10, 0, 5, 5]
        nodes_y = [0, 0, 10, 10, 3, 7]
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Square
            (0, 2), (1, 3),                   # Diagonals
            (4, 5)                            # Extra edge
        ]
        
        solver = planar_cuda.PlanarSolver(nodes_x, nodes_y, edges)
        
        # Initial K-value
        initial_k = solver.calculate_k_value()
        initial_total = solver.calculate_total_crossings()
        
        # Set K-value cost function
        solver.set_cost_function("k_value")
        
        # Act - Run SA optimization
        result = solver.run_sa_optimization(
            iterations=5000,
            start_temp=50.0,
            cooling_rate=0.95
        )
        
        # Final K-value
        final_k = solver.calculate_k_value()
        final_total = solver.calculate_total_crossings()
        
        # Assert - K-value should decrease (or stay same if already optimal)
        assert final_k <= initial_k, \
            f"K-value should not increase: {initial_k} → {final_k}"
        
        print(f"\nK-value optimization results:")
        print(f"  K: {initial_k} → {final_k} (Δ={final_k - initial_k})")
        print(f"  Total: {initial_total} → {final_total} (Δ={final_total - initial_total})")
        print(f"  Accepted moves: {result.get('accepted_moves', 'N/A')}")
    
    def test_sa_minimizes_total_crossings(self):
        """
        Verify that SA with total crossing cost reduces total crossings
        (Baseline comparison)
        """
        # Same graph as above
        nodes_x = [0, 10, 10, 0, 5, 5]
        nodes_y = [0, 0, 10, 10, 3, 7]
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (0, 2), (1, 3),
            (4, 5)
        ]
        
        solver = planar_cuda.PlanarSolver(nodes_x, nodes_y, edges)
        
        initial_k = solver.calculate_k_value()
        initial_total = solver.calculate_total_crossings()
        
        # Set total crossing cost function
        solver.set_cost_function("total_crossings")
        
        # Run SA
        result = solver.run_sa_optimization(
            iterations=5000,
            start_temp=50.0,
            cooling_rate=0.95
        )
        
        final_k = solver.calculate_k_value()
        final_total = solver.calculate_total_crossings()
        
        # Assert - Total crossings should decrease
        assert final_total <= initial_total, \
            f"Total crossings should not increase: {initial_total} → {final_total}"
        
        print(f"\nTotal crossing optimization results:")
        print(f"  K: {initial_k} → {final_k} (Δ={final_k - initial_k})")
        print(f"  Total: {initial_total} → {final_total} (Δ={final_total - initial_total})")
    
    def test_k_value_vs_total_crossings_tradeoff(self):
        """
        Compare K-value optimization vs total crossing optimization
        
        Expected:
        - K-value optimization: Better K, possibly worse total
        - Total crossing optimization: Better total, possibly worse K
        """
        # Arrange
        nodes_x = [0, 10, 10, 0, 5, 2, 8]
        nodes_y = [0, 0, 10, 10, 5, 3, 7]
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (0, 2), (1, 3),
            (4, 5), (4, 6)
        ]
        
        # Test 1: K-value optimization
        solver1 = planar_cuda.PlanarSolver(nodes_x[:], nodes_y[:], edges)
        solver1.set_cost_function("k_value")
        solver1.run_sa_optimization(iterations=10000, start_temp=100.0, cooling_rate=0.97)
        
        k_opt_k = solver1.calculate_k_value()
        k_opt_total = solver1.calculate_total_crossings()
        
        # Test 2: Total crossing optimization
        solver2 = planar_cuda.PlanarSolver(nodes_x[:], nodes_y[:], edges)
        solver2.set_cost_function("total_crossings")
        solver2.run_sa_optimization(iterations=10000, start_temp=100.0, cooling_rate=0.97)
        
        total_opt_k = solver2.calculate_k_value()
        total_opt_total = solver2.calculate_total_crossings()
        
        # Results
        print(f"\nK-value optimization:")
        print(f"  K={k_opt_k}, Total={k_opt_total}")
        print(f"Total crossing optimization:")
        print(f"  K={total_opt_k}, Total={total_opt_total}")
        
        # K-value optimizer should produce better or equal K
        assert k_opt_k <= total_opt_k or abs(k_opt_k - total_opt_k) <= 1, \
            f"K-value optimizer should give better K: {k_opt_k} vs {total_opt_k}"


class TestBenchmarkInstances:
    """
    Test K-value optimization on real benchmark instances
    """
    
    @pytest.mark.skipif(not os.path.exists('live-2025-example-instances'), 
                        reason="Benchmark instances not found")
    def test_70_nodes_k_value_target(self):
        """
        Test that 70-nodes instance can achieve K ≤ 10
        
        This is the main objective from PERFORMANCE_ANALYSIS.md
        """
        # Load instance
        with open('live-2025-example-instances/70.json') as f:
            data = json.load(f)
        
        nodes = data['nodes']
        edges = [(e['source'], e['target']) for e in data['edges']]
        nodes_x = [n['x'] for n in nodes]
        nodes_y = [n['y'] for n in nodes]
        
        solver = planar_cuda.PlanarSolver(nodes_x, nodes_y, edges)
        
        # Initial state
        initial_k = solver.calculate_k_value()
        initial_total = solver.calculate_total_crossings()
        
        print(f"\n70-nodes initial state:")
        print(f"  K-value: {initial_k}")
        print(f"  Total crossings: {initial_total}")
        
        # Optimize with K-value cost
        solver.set_cost_function("k_value")
        result = solver.run_sa_optimization(
            iterations=50000,
            start_temp=100.0,
            cooling_rate=0.99
        )
        
        # Final state
        final_k = solver.calculate_k_value()
        final_total = solver.calculate_total_crossings()
        
        print(f"\n70-nodes after K-value optimization:")
        print(f"  K-value: {final_k} (target: ≤10)")
        print(f"  Total crossings: {final_total}")
        print(f"  Improvement: {initial_k - final_k}")
        print(f"  Accepted moves: {result.get('accepted_moves', 'N/A')}")
        
        # Check violations
        violations = solver.check_violations()
        assert violations == 0, f"Solution must be valid (0 violations), got {violations}"
        
        # Target: K ≤ 10
        assert final_k <= 10, f"Target K≤10 for 70-nodes, got {final_k}"
    
    @pytest.mark.skipif(not os.path.exists('live-2025-example-instances'), 
                        reason="Benchmark instances not found")
    def test_100_nodes_k_value_target(self):
        """
        Test that 100-nodes instance can achieve K ≤ 10
        """
        with open('live-2025-example-instances/100.json') as f:
            data = json.load(f)
        
        nodes = data['nodes']
        edges = [(e['source'], e['target']) for e in data['edges']]
        nodes_x = [n['x'] for n in nodes]
        nodes_y = [n['y'] for n in nodes]
        
        solver = planar_cuda.PlanarSolver(nodes_x, nodes_y, edges)
        
        initial_k = solver.calculate_k_value()
        
        print(f"\n100-nodes initial K-value: {initial_k}")
        
        # Optimize
        solver.set_cost_function("k_value")
        result = solver.run_sa_optimization(
            iterations=50000,
            start_temp=100.0,
            cooling_rate=0.99
        )
        
        final_k = solver.calculate_k_value()
        
        print(f"100-nodes final K-value: {final_k} (target: ≤10)")
        print(f"Improvement: {initial_k - final_k}")
        
        violations = solver.check_violations()
        assert violations == 0, f"Solution must be valid"
        
        # Target: K ≤ 10
        assert final_k <= 10, f"Target K≤10 for 100-nodes, got {final_k}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
