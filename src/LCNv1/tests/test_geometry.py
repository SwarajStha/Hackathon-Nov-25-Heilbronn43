"""
Sprint 1: Geometry Core - Test Suite
Following TDD Red-Green-Refactor methodology.
All tests use INTEGER arithmetic only (no floating point comparisons).
"""
import pytest
import json
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.geometry import Point, GeometryCore


class TestPoint:
    """Test the immutable Point value object."""
    
    def test_point_creation(self):
        """Points must have integer coordinates."""
        p = Point(10, 20)
        assert p.x == 10
        assert p.y == 20
    
    def test_point_immutability(self):
        """Points should be immutable."""
        p = Point(5, 10)
        with pytest.raises(AttributeError):
            p.x = 15  # Should fail
    
    def test_point_equality(self):
        """Points with same coordinates should be equal."""
        p1 = Point(10, 20)
        p2 = Point(10, 20)
        p3 = Point(10, 21)
        
        assert p1 == p2
        assert p1 != p3
    
    def test_point_hash(self):
        """Points should be hashable for use in sets/dicts."""
        p1 = Point(10, 20)
        p2 = Point(10, 20)
        
        point_set = {p1, p2}
        assert len(point_set) == 1  # Same point


class TestGeometryCore:
    """Test core geometric primitives using integer arithmetic."""
    
    def test_cross_product_positive(self):
        """Test cross product with counter-clockwise turn."""
        # Points: O(0,0), A(4,0), B(2,2)
        # OA x OB should be positive (CCW turn)
        result = GeometryCore.cross_product(
            Point(0, 0), Point(4, 0), Point(2, 2)
        )
        assert result > 0, "CCW turn should give positive cross product"
    
    def test_cross_product_negative(self):
        """Test cross product with clockwise turn."""
        # Points: O(0,0), A(4,0), B(2,-2)
        # OA x OB should be negative (CW turn)
        result = GeometryCore.cross_product(
            Point(0, 0), Point(4, 0), Point(2, -2)
        )
        assert result < 0, "CW turn should give negative cross product"
    
    def test_cross_product_collinear(self):
        """Test cross product with collinear points."""
        # Points: O(0,0), A(4,0), B(8,0)
        # All on same line
        result = GeometryCore.cross_product(
            Point(0, 0), Point(4, 0), Point(8, 0)
        )
        assert result == 0, "Collinear points should give zero cross product"
    
    def test_intersection_basic_cross(self):
        """Test basic intersection: two segments crossing in the middle."""
        # Segment 1: (0,0) -> (4,4)
        # Segment 2: (0,4) -> (4,0)
        # These cross at (2,2)
        p1, p2 = Point(0, 0), Point(4, 4)
        q1, q2 = Point(0, 4), Point(4, 0)
        
        assert GeometryCore.segments_intersect(p1, p2, q1, q2) is True
    
    def test_intersection_no_cross_parallel(self):
        """Test parallel segments that don't intersect."""
        # Segment 1: (0,0) -> (4,0)
        # Segment 2: (0,2) -> (4,2)
        p1, p2 = Point(0, 0), Point(4, 0)
        q1, q2 = Point(0, 2), Point(4, 2)
        
        assert GeometryCore.segments_intersect(p1, p2, q1, q2) is False
    
    def test_intersection_shared_endpoint(self):
        """Test segments sharing an endpoint (should NOT count as intersection)."""
        # Segment 1: (0,0) -> (4,4)
        # Segment 2: (0,0) -> (4,0)
        # Share point (0,0)
        p1, p2 = Point(0, 0), Point(4, 4)
        q1, q2 = Point(0, 0), Point(4, 0)
        
        assert GeometryCore.segments_intersect(p1, p2, q1, q2) is False
    
    def test_intersection_v_shape(self):
        """Test V-shaped segments meeting at endpoint (not an intersection)."""
        # Segment 1: (0,0) -> (2,2)
        # Segment 2: (2,2) -> (4,0)
        # Meet at (2,2)
        p1, p2 = Point(0, 0), Point(2, 2)
        q1, q2 = Point(2, 2), Point(4, 0)
        
        assert GeometryCore.segments_intersect(p1, p2, q1, q2) is False
    
    def test_intersection_t_shape(self):
        """Test T-shaped segments (one endpoint touches other segment)."""
        # Segment 1: (0,2) -> (4,2) (horizontal)
        # Segment 2: (2,0) -> (2,2) (vertical, touches at endpoint)
        p1, p2 = Point(0, 2), Point(4, 2)
        q1, q2 = Point(2, 0), Point(2, 2)
        
        # This is tricky - depends on definition
        # Usually endpoint touching is NOT a crossing
        assert GeometryCore.segments_intersect(p1, p2, q1, q2) is False
    
    def test_intersection_collinear_overlapping(self):
        """Test collinear overlapping segments."""
        # Segment 1: (0,0) -> (4,0)
        # Segment 2: (2,0) -> (6,0)
        p1, p2 = Point(0, 0), Point(4, 0)
        q1, q2 = Point(2, 0), Point(6, 0)
        
        assert GeometryCore.segments_intersect(p1, p2, q1, q2) is False
    
    def test_intersection_collinear_separate(self):
        """Test collinear non-overlapping segments."""
        # Segment 1: (0,0) -> (2,0)
        # Segment 2: (4,0) -> (6,0)
        p1, p2 = Point(0, 0), Point(2, 0)
        q1, q2 = Point(4, 0), Point(6, 0)
        
        assert GeometryCore.segments_intersect(p1, p2, q1, q2) is False


class TestGeometryIntegration:
    """Integration tests using the 15-nodes.json ground truth dataset."""
    
    @pytest.fixture
    def load_15_nodes(self):
        """Load the 15-nodes.json test dataset."""
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
    
    def test_load_15_nodes_structure(self, load_15_nodes):
        """Verify we can correctly parse the JSON structure."""
        data = load_15_nodes
        
        assert 'nodes' in data
        assert 'edges' in data
        assert 'width' in data
        assert 'height' in data
        
        assert len(data['nodes']) == 15
        assert len(data['edges']) > 0
        
        # Check node structure
        for node in data['nodes']:
            assert 'id' in node
            assert 'x' in node
            assert 'y' in node
            # Coordinates should be integers or convertible to integers
            assert isinstance(node['x'], (int, float))
            assert isinstance(node['y'], (int, float))
    
    def test_convert_to_integer_grid(self, load_15_nodes):
        """Verify all coordinates can be converted to integers."""
        data = load_15_nodes
        
        points = {}
        for node in data['nodes']:
            nid = node['id']
            # Convert to integer grid
            x = int(node['x'])
            y = int(node['y'])
            points[nid] = Point(x, y)
        
        assert len(points) == 15
        
        # Verify all points are on integer grid
        for p in points.values():
            assert isinstance(p.x, int)
            assert isinstance(p.y, int)
    
    def test_brute_force_crossing_count(self, load_15_nodes):
        """
        Calculate crossing number using brute force O(E^2) algorithm.
        This serves as ground truth for validating spatial hash later.
        """
        data = load_15_nodes
        
        # Build point map
        points = {}
        for node in data['nodes']:
            nid = node['id']
            points[nid] = Point(int(node['x']), int(node['y']))
        
        # Build edge list
        edges = []
        for edge in data['edges']:
            src = edge['source']
            tgt = edge['target']
            edges.append((points[src], points[tgt]))
        
        # Count crossings using brute force
        crossing_count = 0
        edge_crossing_counts = [0] * len(edges)
        
        for i in range(len(edges)):
            for j in range(i + 1, len(edges)):
                p1, p2 = edges[i]
                q1, q2 = edges[j]
                
                if GeometryCore.segments_intersect(p1, p2, q1, q2):
                    crossing_count += 1
                    edge_crossing_counts[i] += 1
                    edge_crossing_counts[j] += 1
        
        # Calculate k (max crossings on any single edge)
        k = max(edge_crossing_counts) if edge_crossing_counts else 0
        
        # Log results for debugging
        print(f"\n15-nodes.json Analysis:")
        print(f"  Total Edges: {len(edges)}")
        print(f"  Total Crossings: {crossing_count}")
        print(f"  K (max per edge): {k}")
        print(f"  Top 5 edges by crossings: {sorted(edge_crossing_counts, reverse=True)[:5]}")
        
        # Basic sanity checks
        assert crossing_count >= 0, "Crossing count must be non-negative"
        assert k >= 0, "K must be non-negative"
        assert k <= len(edges) - 1, "K cannot exceed number of other edges"
        
        # Store for comparison in later tests
        return {
            'total_crossings': crossing_count,
            'k': k,
            'edge_crossings': edge_crossing_counts
        }
    
    def test_no_self_intersection(self, load_15_nodes):
        """Verify that no edge intersects itself."""
        data = load_15_nodes
        
        points = {}
        for node in data['nodes']:
            nid = node['id']
            points[nid] = Point(int(node['x']), int(node['y']))
        
        for edge in data['edges']:
            p1 = points[edge['source']]
            p2 = points[edge['target']]
            
            # An edge should never intersect itself
            assert GeometryCore.segments_intersect(p1, p2, p1, p2) is False


class TestIntegerArithmeticInvariant:
    """Verify that all geometric operations use integer arithmetic only."""
    
    def test_cross_product_integer_inputs_integer_output(self):
        """Cross product with integer inputs must produce integer output."""
        result = GeometryCore.cross_product(
            Point(1, 2), Point(3, 4), Point(5, 6)
        )
        assert isinstance(result, int), "Cross product must return integer"
    
    def test_large_coordinates(self):
        """Test with large coordinates to verify no overflow issues."""
        # Use coordinates near the bounds mentioned in requirements
        p1 = Point(0, 0)
        p2 = Point(1000000, 1000000)
        q1 = Point(0, 1000000)
        q2 = Point(1000000, 0)
        
        # Should handle large values without overflow
        result = GeometryCore.segments_intersect(p1, p2, q1, q2)
        assert isinstance(result, bool)
        assert result is True  # These segments cross


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
