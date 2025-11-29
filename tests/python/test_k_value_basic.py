"""
Test K-Value Calculator - Basic Functionality

This test suite verifies that the K-value calculator correctly:
1. Calculates the maximum crossing count among all edges
2. Returns 0 for planar graphs
3. Returns correct values for non-planar graphs

Test-Driven Development (TDD):
- These tests are written BEFORE implementation
- They should FAIL initially (Red phase)
- Implementation will make them PASS (Green phase)
"""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    import planar_cuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    pytest.skip("CUDA module not available", allow_module_level=True)


class TestKValueBasic:
    """Basic K-value calculation tests"""
    
    def test_triangle_graph_k_zero(self):
        """
        Triangle graph should have K=0 (planar, no crossings)
        
        Graph structure:
            0---1
             \ /
              2
        
        Edges: (0,1), (1,2), (0,2)
        Expected K = 0
        """
        # Arrange
        nodes_x = [0, 10, 5]
        nodes_y = [0, 0, 10]
        edges = [(0, 1), (1, 2), (0, 2)]
        
        solver = planar_cuda.PlanarSolver(nodes_x, nodes_y, edges)
        
        # Act
        k_value = solver.calculate_k_value()
        
        # Assert
        assert k_value == 0, f"Triangle should have K=0, got {k_value}"
    
    def test_square_graph_k_zero(self):
        """
        Square graph should have K=0 (planar)
        
        Graph structure:
            0---1
            |   |
            3---2
        
        Edges: (0,1), (1,2), (2,3), (3,0)
        Expected K = 0
        """
        # Arrange
        nodes_x = [0, 10, 10, 0]
        nodes_y = [0, 0, 10, 10]
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        
        solver = planar_cuda.PlanarSolver(nodes_x, nodes_y, edges)
        
        # Act
        k_value = solver.calculate_k_value()
        
        # Assert
        assert k_value == 0, f"Square should have K=0, got {k_value}"
    
    def test_k4_complete_graph(self):
        """
        K4 complete graph (non-planar) should have K >= 1
        
        Graph structure:
            0---1
            |\ /|
            | X |
            |/ \|
            3---2
        
        Edges: Square + 2 diagonals
        The two diagonals must cross each other
        Expected: K >= 1 (each diagonal crosses the other)
        """
        # Arrange
        nodes_x = [0, 10, 10, 0]
        nodes_y = [0, 0, 10, 10]
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Square edges
            (0, 2), (1, 3)                    # Diagonals (must cross!)
        ]
        
        solver = planar_cuda.PlanarSolver(nodes_x, nodes_y, edges)
        
        # Act
        k_value = solver.calculate_k_value()
        
        # Assert
        assert k_value >= 1, f"K4 complete graph must have K>=1, got {k_value}"
        
        # Additional check: get per-edge crossings
        edge_crossings = solver.get_edge_crossings()
        assert len(edge_crossings) == 6, "Should have 6 edges"
        
        # Diagonals should each have at least 1 crossing
        diagonal1_crossings = edge_crossings[4]  # Edge (0,2)
        diagonal2_crossings = edge_crossings[5]  # Edge (1,3)
        
        assert diagonal1_crossings >= 1, f"Diagonal (0,2) should cross diagonal (1,3)"
        assert diagonal2_crossings >= 1, f"Diagonal (1,3) should cross diagonal (0,2)"
    
    def test_k5_complete_graph(self):
        """
        K5 complete graph has more crossings
        
        K5 is non-planar and has many crossings.
        This tests that K-value scales with graph complexity.
        """
        # Arrange - Pentagon with all diagonals
        import math
        nodes_x = [int(50 + 40 * math.cos(2 * math.pi * i / 5)) for i in range(5)]
        nodes_y = [int(50 + 40 * math.sin(2 * math.pi * i / 5)) for i in range(5)]
        
        # Complete graph: all pairs
        edges = [(i, j) for i in range(5) for j in range(i+1, 5)]
        
        solver = planar_cuda.PlanarSolver(nodes_x, nodes_y, edges)
        
        # Act
        k_value = solver.calculate_k_value()
        
        # Assert
        assert k_value >= 2, f"K5 should have K>=2, got {k_value}"
    
    def test_edge_crossings_consistency(self):
        """
        Verify that edge crossings array has correct length
        and values are non-negative.
        """
        # Arrange
        nodes_x = [0, 10, 10, 0]
        nodes_y = [0, 0, 10, 10]
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)]
        
        solver = planar_cuda.PlanarSolver(nodes_x, nodes_y, edges)
        
        # Act
        edge_crossings = solver.get_edge_crossings()
        k_value = solver.calculate_k_value()
        
        # Assert
        assert len(edge_crossings) == len(edges), \
            f"Edge crossings array length mismatch: {len(edge_crossings)} vs {len(edges)}"
        
        assert all(c >= 0 for c in edge_crossings), \
            "All crossing counts must be non-negative"
        
        assert k_value == max(edge_crossings), \
            f"K-value should equal max of edge crossings: {k_value} vs {max(edge_crossings)}"
    
    def test_single_edge_k_zero(self):
        """Single edge graph should have K=0"""
        nodes_x = [0, 10]
        nodes_y = [0, 0]
        edges = [(0, 1)]
        
        solver = planar_cuda.PlanarSolver(nodes_x, nodes_y, edges)
        k_value = solver.calculate_k_value()
        
        assert k_value == 0, f"Single edge should have K=0, got {k_value}"
    
    def test_two_parallel_edges_k_zero(self):
        """Two parallel edges should have K=0"""
        nodes_x = [0, 10, 0, 10]
        nodes_y = [0, 0, 5, 5]
        edges = [(0, 1), (2, 3)]
        
        solver = planar_cuda.PlanarSolver(nodes_x, nodes_y, edges)
        k_value = solver.calculate_k_value()
        
        assert k_value == 0, f"Parallel edges should have K=0, got {k_value}"
    
    def test_two_crossing_edges(self):
        """Two edges forming an X should each have K=1"""
        # Arrange - Create explicit X pattern
        nodes_x = [0, 10, 10, 0]
        nodes_y = [0, 0, 10, 10]
        edges = [(0, 2), (1, 3)]  # Two diagonals that cross
        
        solver = planar_cuda.PlanarSolver(nodes_x, nodes_y, edges)
        
        # Act
        k_value = solver.calculate_k_value()
        edge_crossings = solver.get_edge_crossings()
        
        # Assert
        assert k_value == 1, f"Two crossing edges should have K=1, got {k_value}"
        assert edge_crossings[0] == 1, "First edge should cross once"
        assert edge_crossings[1] == 1, "Second edge should cross once"


class TestKValueAfterMove:
    """Test K-value updates after node movements"""
    
    def test_k_value_changes_after_move(self):
        """
        Test that K-value correctly updates when nodes are moved
        """
        # Arrange - Start with crossing
        nodes_x = [0, 10, 10, 0]
        nodes_y = [0, 0, 10, 10]
        edges = [(0, 2), (1, 3)]  # X pattern
        
        solver = planar_cuda.PlanarSolver(nodes_x, nodes_y, edges)
        initial_k = solver.calculate_k_value()
        
        # Act - Move node to eliminate crossing
        solver.move_node(2, 20, 10)  # Move node 2 far away
        new_k = solver.calculate_k_value()
        
        # Assert
        assert initial_k == 1, "Initial K should be 1 (crossing exists)"
        assert new_k == 0, "After move, K should be 0 (no crossing)"
    
    def test_k_value_increases_with_bad_move(self):
        """Test that a bad move increases K-value"""
        # Arrange - Start planar
        nodes_x = [0, 10, 5]
        nodes_y = [0, 0, 10]
        edges = [(0, 1), (1, 2), (0, 2)]
        
        solver = planar_cuda.PlanarSolver(nodes_x, nodes_y, edges)
        initial_k = solver.calculate_k_value()
        
        # Act - Move node to create crossing
        solver.move_node(2, 5, -1)  # Move node 2 below edge (0,1)
        new_k = solver.calculate_k_value()
        
        # Assert
        assert initial_k == 0, "Initial triangle should be planar"
        assert new_k >= 1, "Bad move should create crossings"


class TestKValueRealWorld:
    """Test K-value on real benchmark instances"""
    
    @pytest.mark.skipif(not os.path.exists('live-2025-example-instances'), 
                        reason="Benchmark instances not found")
    def test_15_nodes_instance(self):
        """Test K-value calculation on 15-nodes instance"""
        import json
        
        with open('live-2025-example-instances/15.json') as f:
            data = json.load(f)
        
        nodes = data['nodes']
        edges = [(e['source'], e['target']) for e in data['edges']]
        nodes_x = [n['x'] for n in nodes]
        nodes_y = [n['y'] for n in nodes]
        
        solver = planar_cuda.PlanarSolver(nodes_x, nodes_y, edges)
        k_value = solver.calculate_k_value()
        
        # Just verify it runs and returns reasonable value
        assert k_value >= 0, "K-value must be non-negative"
        assert k_value < len(edges), "K-value can't exceed number of edges"
        
        print(f"15-nodes instance K-value: {k_value}")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
