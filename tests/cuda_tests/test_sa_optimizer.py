"""
Cycle 4: Simulated Annealing Optimizer Tests (TDD)

Test suite for GPU-accelerated SA optimization.
Validates convergence, solution quality, and performance.
"""

import pytest
import json
import time
import os
import sys

# Add CUDA DLL directory (Windows requirement)
if os.name == 'nt':
    cuda_path = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin'
    if os.path.exists(cuda_path):
        os.add_dll_directory(cuda_path)

# Add build artifacts to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'build_artifacts'))
import planar_cuda


class TestSABasicFunctionality:
    """Test basic SA functionality and API"""
    
    def test_sa_api_exists(self):
        """Test that run_sa_optimization method exists"""
        nodes_x = [0, 10, 5]
        nodes_y = [0, 0, 10]
        edges = [(0, 1), (1, 2), (2, 0)]
        
        solver = planar_cuda.PlanarSolver(nodes_x, nodes_y, edges)
        
        # Check method exists
        assert hasattr(solver, 'run_sa_optimization'), "run_sa_optimization method missing"
    
    def test_sa_returns_stats(self):
        """Test that SA returns optimization statistics"""
        nodes_x = [0, 10, 5]
        nodes_y = [0, 0, 10]
        edges = [(0, 1), (1, 2), (2, 0)]
        
        solver = planar_cuda.PlanarSolver(nodes_x, nodes_y, edges)
        
        # Run minimal optimization
        stats = solver.run_sa_optimization(
            iterations=100,
            start_temp=10.0,
            cooling_rate=0.95
        )
        
        # Check stats structure
        assert isinstance(stats, dict), "Stats should be a dictionary"
        assert 'initial_crossings' in stats
        assert 'final_crossings' in stats
        assert 'iterations' in stats
        assert 'accepted_moves' in stats
    
    def test_sa_preserves_topology(self):
        """Test that SA doesn't modify graph topology"""
        nodes_x = [0, 10, 5]
        nodes_y = [0, 0, 10]
        edges = [(0, 1), (1, 2), (2, 0)]
        
        solver = planar_cuda.PlanarSolver(nodes_x, nodes_y, edges)
        
        initial_crossings = solver.calculate_total_crossings()
        
        # Run SA
        solver.run_sa_optimization(iterations=100, start_temp=10.0, cooling_rate=0.95)
        
        # Verify crossings can still be computed (topology intact)
        final_crossings = solver.calculate_total_crossings()
        assert isinstance(final_crossings, int)


class TestSAConvergence:
    """Test SA convergence behavior"""
    
    def test_sa_reduces_crossings_simple(self):
        """Test that SA reduces crossings on simple graph"""
        # Create X-shape graph (2 crossing edges)
        nodes_x = [0, 10, 0, 10]
        nodes_y = [0, 0, 10, 10]
        edges = [(0, 3), (1, 2)]  # X-shape
        
        solver = planar_cuda.PlanarSolver(nodes_x, nodes_y, edges)
        
        initial_k = solver.calculate_total_crossings()
        assert initial_k == 1, f"Expected 1 crossing, got {initial_k}"
        
        # Run SA to untangle
        stats = solver.run_sa_optimization(
            iterations=1000,
            start_temp=50.0,
            cooling_rate=0.95
        )
        
        final_k = solver.calculate_total_crossings()
        
        print(f"\nSimple X-shape: Initial={initial_k}, Final={final_k}")
        print(f"Stats: {stats}")
        
        # SA should be able to untangle simple X
        assert final_k <= initial_k, "SA should not increase crossings"
        assert final_k == 0, "SA should eliminate all crossings in simple case"
    
    def test_sa_convergence_15_nodes(self):
        """Test convergence on 15-node benchmark"""
        instance_path = os.path.join(
            os.path.dirname(__file__), '..', '..',
            'live-2025-example-instances', '15-nodes.json'
        )
        
        with open(instance_path, 'r') as f:
            data = json.load(f)
        
        nodes_x = [node['x'] for node in data['nodes']]
        nodes_y = [node['y'] for node in data['nodes']]
        edges = [(edge['source'], edge['target']) for edge in data['edges']]
        
        solver = planar_cuda.PlanarSolver(nodes_x, nodes_y, edges)
        
        initial_k = solver.calculate_total_crossings()
        print(f"\n15-node initial crossings: {initial_k}")
        
        # Run SA optimization
        stats = solver.run_sa_optimization(
            iterations=50000,
            start_temp=500.0,
            cooling_rate=0.99
        )
        
        final_k = solver.calculate_total_crossings()
        
        print(f"15-node: Initial={initial_k}, Final={final_k}")
        print(f"Accepted: {stats['accepted_moves']}/{stats['iterations']}")
        
        # Must show improvement
        assert final_k < initial_k, "SA should improve solution"
        
        # Should reach reasonable quality (allow more crossings for now)
        assert final_k <= 50, f"Expected ≤50 crossings, got {final_k}"
    
    def test_sa_convergence_70_nodes(self):
        """Test convergence on 70-node graph"""
        instance_path = os.path.join(
            os.path.dirname(__file__), '..', '..',
            'live-2025-example-instances', '70-nodes.json'
        )
        
        with open(instance_path, 'r') as f:
            data = json.load(f)
        
        nodes_x = [node['x'] for node in data['nodes']]
        nodes_y = [node['y'] for node in data['nodes']]
        edges = [(edge['source'], edge['target']) for edge in data['edges']]
        
        solver = planar_cuda.PlanarSolver(nodes_x, nodes_y, edges)
        
        initial_k = solver.calculate_total_crossings()
        print(f"\n70-node initial crossings: {initial_k}")
        
        # Run SA optimization
        start_time = time.perf_counter()
        stats = solver.run_sa_optimization(
            iterations=100000,
            start_temp=1000.0,
            cooling_rate=0.99
        )
        elapsed = time.perf_counter() - start_time
        
        final_k = solver.calculate_total_crossings()
        
        print(f"70-node: Initial={initial_k}, Final={final_k}")
        print(f"Time: {elapsed:.2f}s, Iterations: {stats['iterations']}")
        print(f"Accepted: {stats['accepted_moves']}/{stats['iterations']} ({100*stats['accepted_moves']/stats['iterations']:.1f}%)")
        
        # Must show improvement
        assert final_k < initial_k, "SA should improve solution"
        
        # Should reach reasonable quality
        improvement_pct = 100 * (initial_k - final_k) / initial_k
        assert improvement_pct >= 10, f"Expected ≥10% improvement, got {improvement_pct:.1f}%"


class TestSAPerformance:
    """Test SA performance vs Python baseline"""
    
    def test_sa_speed_vs_baseline(self):
        """Compare C++/CUDA SA speed to hypothetical Python baseline"""
        instance_path = os.path.join(
            os.path.dirname(__file__), '..', '..',
            'live-2025-example-instances', '15-nodes.json'
        )
        
        with open(instance_path, 'r') as f:
            data = json.load(f)
        
        nodes_x = [node['x'] for node in data['nodes']]
        nodes_y = [node['y'] for node in data['nodes']]
        edges = [(edge['source'], edge['target']) for edge in data['edges']]
        
        solver = planar_cuda.PlanarSolver(nodes_x, nodes_y, edges)
        
        # Measure C++/CUDA SA time
        start = time.perf_counter()
        stats = solver.run_sa_optimization(
            iterations=10000,
            start_temp=100.0,
            cooling_rate=0.95
        )
        cuda_time = time.perf_counter() - start
        
        print(f"\nC++/CUDA SA (10k iterations): {cuda_time:.4f}s")
        print(f"Iterations per second: {stats['iterations']/cuda_time:.0f}")
        
        # Expected: >200 iterations/sec (Python typically ~100)
        iterations_per_sec = stats['iterations'] / cuda_time
        assert iterations_per_sec > 200, f"Expected >200 it/s, got {iterations_per_sec:.0f}"


class TestSAParameters:
    """Test SA parameter sensitivity"""
    
    def test_temperature_effect(self):
        """Test that temperature affects acceptance rate"""
        # Use larger graph to see temperature effects
        instance_path = os.path.join(
            os.path.dirname(__file__), '..', '..',
            'live-2025-example-instances', '15-nodes.json'
        )
        
        with open(instance_path, 'r') as f:
            data = json.load(f)
        
        nodes_x = [node['x'] for node in data['nodes']]
        nodes_y = [node['y'] for node in data['nodes']]
        edges = [(edge['source'], edge['target']) for edge in data['edges']]
        
        # High temperature run
        solver_hot = planar_cuda.PlanarSolver(nodes_x, nodes_y, edges)
        stats_hot = solver_hot.run_sa_optimization(
            iterations=5000, start_temp=5000.0, cooling_rate=0.99
        )
        
        # Low temperature run
        solver_cold = planar_cuda.PlanarSolver(nodes_x, nodes_y, edges)
        stats_cold = solver_cold.run_sa_optimization(
            iterations=5000, start_temp=10.0, cooling_rate=0.99
        )
        
        # High temperature should accept more moves
        hot_accept_rate = stats_hot['accepted_moves'] / stats_hot['iterations']
        cold_accept_rate = stats_cold['accepted_moves'] / stats_cold['iterations']
        
        print(f"\nHot temp accept rate: {100*hot_accept_rate:.1f}%")
        print(f"Cold temp accept rate: {100*cold_accept_rate:.1f}%")
        
        assert hot_accept_rate >= cold_accept_rate, "Higher temp should accept more/equal moves"
    
    def test_cooling_rate_effect(self):
        """Test that cooling rate affects convergence"""
        nodes_x = [0, 10, 5]
        nodes_y = [0, 0, 10]
        edges = [(0, 1), (1, 2), (2, 0)]
        
        # Fast cooling
        solver_fast = planar_cuda.PlanarSolver(nodes_x, nodes_y, edges)
        stats_fast = solver_fast.run_sa_optimization(
            iterations=1000, start_temp=100.0, cooling_rate=0.9
        )
        
        # Slow cooling
        solver_slow = planar_cuda.PlanarSolver(nodes_x, nodes_y, edges)
        stats_slow = solver_slow.run_sa_optimization(
            iterations=1000, start_temp=100.0, cooling_rate=0.99
        )
        
        fast_accept = stats_fast['accepted_moves']
        slow_accept = stats_slow['accepted_moves']
        
        print(f"\nFast cooling accepted: {fast_accept}")
        print(f"Slow cooling accepted: {slow_accept}")
        
        # Slow cooling should explore more
        assert slow_accept >= fast_accept * 0.8, "Slow cooling should explore more"


class TestSAEdgeCases:
    """Test SA edge cases and robustness"""
    
    def test_zero_iterations(self):
        """Test SA with 0 iterations"""
        nodes_x = [0, 10, 5]
        nodes_y = [0, 0, 10]
        edges = [(0, 1), (1, 2), (2, 0)]
        
        solver = planar_cuda.PlanarSolver(nodes_x, nodes_y, edges)
        initial_k = solver.calculate_total_crossings()
        
        stats = solver.run_sa_optimization(iterations=0, start_temp=100.0, cooling_rate=0.95)
        
        final_k = solver.calculate_total_crossings()
        
        # Should not change anything
        assert stats['iterations'] == 0
        assert stats['accepted_moves'] == 0
        assert final_k == initial_k
    
    def test_planar_graph_stays_planar(self):
        """Test that planar graph remains planar"""
        # Triangle (planar, 0 crossings)
        nodes_x = [0, 10, 5]
        nodes_y = [0, 0, 10]
        edges = [(0, 1), (1, 2), (2, 0)]
        
        solver = planar_cuda.PlanarSolver(nodes_x, nodes_y, edges)
        
        initial_k = solver.calculate_total_crossings()
        assert initial_k == 0, "Triangle should be planar"
        
        # Run SA (shouldn't create crossings)
        solver.run_sa_optimization(iterations=1000, start_temp=100.0, cooling_rate=0.95)
        
        final_k = solver.calculate_total_crossings()
        assert final_k == 0, "SA should preserve planarity"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
