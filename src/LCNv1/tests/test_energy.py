"""
Sprint 3: Energy & Delta Update - Test Suite
Critical test: E_total + delta == E'_total (incremental update correctness)

This validates the core optimization loop - incorrect delta calculations
will lead to divergent energy estimates and broken optimization.
"""
import pytest
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.geometry import Point
from core.graph import GraphData, GridState
from core.cost import ICostFunction, SoftMaxCost


class TestGraphDataStructures:
    """Test the graph data structures."""
    
    @pytest.fixture
    def simple_graph_data(self):
        """Create a simple graph for testing."""
        # Triangle: 0-1-2-0
        edges = [
            (0, 1),
            (1, 2),
            (2, 0)
        ]
        return GraphData(num_nodes=3, edges=edges)
    
    def test_graph_creation(self, simple_graph_data):
        """Test basic graph creation."""
        graph = simple_graph_data
        assert graph.num_nodes == 3
        assert len(graph.edges) == 3
    
    def test_adjacency_list(self, simple_graph_data):
        """Test adjacency list construction."""
        graph = simple_graph_data
        
        # Node 0 connects to 1 and 2
        assert len(graph.get_incident_edges(0)) == 2
        # Node 1 connects to 0 and 2
        assert len(graph.get_incident_edges(1)) == 2
        # Node 2 connects to 1 and 0
        assert len(graph.get_incident_edges(2)) == 2
    
    def test_edge_lookup(self, simple_graph_data):
        """Test edge endpoint lookup."""
        graph = simple_graph_data
        
        edge_id = 0  # First edge (0, 1)
        src, tgt = graph.get_edge_endpoints(edge_id)
        assert (src == 0 and tgt == 1) or (src == 1 and tgt == 0)


class TestGridState:
    """Test the grid state management."""
    
    def test_create_grid_state(self):
        """Create grid state with positions."""
        positions = {
            0: Point(0, 0),
            1: Point(10, 10),
            2: Point(20, 0)
        }
        
        state = GridState(positions, width=100, height=100)
        
        assert state.get_position(0) == Point(0, 0)
        assert state.get_position(1) == Point(10, 10)
        assert state.get_position(2) == Point(20, 0)
    
    def test_move_node(self):
        """Test moving a node."""
        positions = {
            0: Point(0, 0),
            1: Point(10, 10)
        }
        
        state = GridState(positions, width=100, height=100)
        
        # Move node 0
        new_pos = Point(5, 5)
        state.move_node(0, new_pos)
        
        assert state.get_position(0) == new_pos
        assert state.get_position(1) == Point(10, 10)  # Unchanged
    
    def test_collision_detection(self):
        """Test that nodes cannot overlap."""
        positions = {
            0: Point(0, 0),
            1: Point(10, 10)
        }
        
        state = GridState(positions, width=100, height=100)
        
        # Try to move node 0 to same position as node 1
        # Should either reject or handle collision
        occupied = state.is_occupied(Point(10, 10))
        assert occupied is True
        
        occupied = state.is_occupied(Point(5, 5))
        assert occupied is False


class TestCostFunction:
    """Test cost function calculations."""
    
    @pytest.fixture
    def load_15_nodes(self):
        """Load 15-nodes.json"""
        json_path = os.path.join(
            os.path.dirname(__file__),
            '..',
            '..',
            '..',
            'live-2025-example-instances',
            '15-nodes.json'
        )
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data
    
    def test_cost_function_interface(self):
        """Test that cost function implements required interface."""
        # Simple graph
        edges = [(0, 1), (1, 2)]
        graph = GraphData(num_nodes=3, edges=edges)
        
        positions = {
            0: Point(0, 0),
            1: Point(10, 10),
            2: Point(20, 0)
        }
        state = GridState(positions, width=100, height=100)
        
        cost_func = SoftMaxCost(w_cross=100.0, w_len=1.0, power=2)
        
        # Must implement calculate
        total_cost = cost_func.calculate(graph, state)
        assert isinstance(total_cost, (int, float))
        assert total_cost >= 0
    
    def test_crossing_energy_basic(self):
        """Test crossing energy calculation."""
        # Two crossing edges: X shape
        # (0,0)-(10,10) and (0,10)-(10,0)
        edges = [(0, 1), (2, 3)]
        graph = GraphData(num_nodes=4, edges=edges)
        
        positions = {
            0: Point(0, 0),
            1: Point(10, 10),
            2: Point(0, 10),
            3: Point(10, 0)
        }
        state = GridState(positions, width=20, height=20)
        
        cost_func = SoftMaxCost(w_cross=100.0, w_len=0.0, power=2)
        cost = cost_func.calculate(graph, state)
        
        # Should have 1 crossing
        # Each edge has k=1 crossing
        # With power=2: cost = w_cross * (1^2 + 1^2) = 100 * 2 = 200
        # Plus the max k term (depending on implementation)
        assert cost > 0
    
    def test_length_energy_basic(self):
        """Test edge length energy calculation."""
        # Single edge
        edges = [(0, 1)]
        graph = GraphData(num_nodes=2, edges=edges)
        
        positions = {
            0: Point(0, 0),
            1: Point(3, 4)  # Distance = 5
        }
        state = GridState(positions, width=10, height=10)
        
        cost_func = SoftMaxCost(w_cross=0.0, w_len=1.0, power=2)
        cost = cost_func.calculate(graph, state)
        
        # Length squared = 3^2 + 4^2 = 25
        # Cost = w_len * 25 = 25
        assert abs(cost - 25.0) < 1e-6


class TestDeltaUpdateCorrectness:
    """
    The most critical test: verify delta update correctness.
    E_total + delta == E'_total must hold EXACTLY.
    """
    
    @pytest.fixture
    def load_15_nodes(self):
        """Load 15-nodes.json dataset."""
        json_path = os.path.join(
            os.path.dirname(__file__), 
            '..', 
            '..',
            '..',
            'live-2025-example-instances', 
            '15-nodes.json'
        )
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data
    
    def test_delta_update_15_nodes(self, load_15_nodes):
        """
        Test incremental delta update on 15-nodes.json.
        
        This is the HEART of the optimization - if this fails,
        the SA algorithm will not converge correctly.
        """
        data = load_15_nodes
        
        # Build graph
        edges = [(e['source'], e['target']) for e in data['edges']]
        graph = GraphData(num_nodes=len(data['nodes']), edges=edges)
        
        # Initial positions
        positions = {}
        for node in data['nodes']:
            nid = node['id']
            positions[nid] = Point(int(node['x']), int(node['y']))
        
        state = GridState(positions, width=data['width'], height=data['height'])
        
        # Cost function
        cost_func = SoftMaxCost(w_cross=100.0, w_len=1.0, power=2)
        
        # Calculate initial energy
        E_initial = cost_func.calculate(graph, state)
        
        print(f"\nDelta Update Test:")
        print(f"  Initial Energy: {E_initial:.2f}")
        
        # Test multiple random moves
        import random
        random.seed(42)
        
        num_tests = 10
        max_error = 0.0
        
        for test_i in range(num_tests):
            # Pick random node to move
            node_id = random.randint(0, graph.num_nodes - 1)
            old_pos = state.get_position(node_id)
            
            # Generate random new position
            new_x = random.randint(0, data['width'])
            new_y = random.randint(0, data['height'])
            new_pos = Point(new_x, new_y)
            
            # Calculate delta (WITHOUT moving)
            delta = cost_func.calculate_delta(graph, state, node_id, new_pos)
            
            # Predicted energy
            E_predicted = E_initial + delta
            
            # Actually move the node
            state.move_node(node_id, new_pos)
            
            # Recalculate full energy
            E_actual = cost_func.calculate(graph, state)
            
            # Check correctness
            error = abs(E_predicted - E_actual)
            max_error = max(max_error, error)
            
            if test_i < 3:  # Print first few
                print(f"  Test {test_i}: delta={delta:.2f}, predicted={E_predicted:.2f}, "
                      f"actual={E_actual:.2f}, error={error:.6f}")
            
            # Tolerance for numerical errors (very small)
            # This should be EXACT for integer coordinates
            tolerance = 1e-6
            assert error < tolerance, \
                f"Delta update error too large: {error:.6f} (should be < {tolerance})"
            
            # Update for next iteration
            E_initial = E_actual
            
            # Move back for next test
            state.move_node(node_id, old_pos)
            E_initial = cost_func.calculate(graph, state)
        
        print(f"  Max Error across {num_tests} tests: {max_error:.10f}")
        print(f"  ✓ Delta update correctness verified!")
    
    def test_delta_with_no_incident_edges(self):
        """Test delta when moving an isolated node."""
        # Node 2 is isolated
        edges = [(0, 1)]
        graph = GraphData(num_nodes=3, edges=edges)
        
        positions = {
            0: Point(0, 0),
            1: Point(10, 0),
            2: Point(5, 5)  # Isolated
        }
        state = GridState(positions, width=20, height=20)
        
        cost_func = SoftMaxCost(w_cross=100.0, w_len=1.0, power=2)
        
        E_initial = cost_func.calculate(graph, state)
        
        # Move isolated node
        delta = cost_func.calculate_delta(graph, state, 2, Point(15, 15))
        
        # Should be zero (no incident edges affected)
        assert abs(delta) < 1e-6
    
    def test_delta_with_single_incident_edge(self):
        """Test delta with one incident edge."""
        edges = [(0, 1)]
        graph = GraphData(num_nodes=2, edges=edges)
        
        positions = {
            0: Point(0, 0),
            1: Point(10, 0)
        }
        state = GridState(positions, width=20, height=20)
        
        # Only length cost (no crossings possible with 1 edge)
        cost_func = SoftMaxCost(w_cross=0.0, w_len=1.0, power=2)
        
        E_initial = cost_func.calculate(graph, state)
        # Initial length = 10, cost = 100
        
        # Move node 1 to (5, 0) - length becomes 5
        delta = cost_func.calculate_delta(graph, state, 1, Point(5, 0))
        
        # Old length^2 = 100, new length^2 = 25
        # Delta = 25 - 100 = -75
        expected_delta = -75.0
        
        assert abs(delta - expected_delta) < 1e-6


class TestCostComponents:
    """Test individual components of the cost function."""
    
    def test_crossing_count_correctness(self):
        """Verify crossing counting matches geometry tests."""
        # X crossing
        edges = [(0, 1), (2, 3)]
        graph = GraphData(num_nodes=4, edges=edges)
        
        positions = {
            0: Point(0, 0),
            1: Point(10, 10),
            2: Point(0, 10),
            3: Point(10, 0)
        }
        state = GridState(positions, width=20, height=20)
        
        cost_func = SoftMaxCost(w_cross=1.0, w_len=0.0, power=1)
        
        # Get crossing details
        k, total = cost_func.get_crossing_stats(graph, state)
        
        assert total == 1  # One crossing
        assert k == 1      # Each edge crosses once
    
    def test_edge_length_calculation(self):
        """Test edge length calculation."""
        edges = [(0, 1)]
        graph = GraphData(num_nodes=2, edges=edges)
        
        positions = {
            0: Point(0, 0),
            1: Point(3, 4)
        }
        state = GridState(positions, width=10, height=10)
        
        cost_func = SoftMaxCost(w_cross=0.0, w_len=1.0, power=2)
        
        # Length = 5, length^2 = 25
        total_length_energy = cost_func.get_length_energy(graph, state)
        
        assert abs(total_length_energy - 25.0) < 1e-6


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
