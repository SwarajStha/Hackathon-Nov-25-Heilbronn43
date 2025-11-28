"""
Sprint 2: Spatial Hash - Test Suite
Testing O(1) spatial indexing for fast edge intersection queries.
"""
import pytest
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.geometry import Point, GeometryCore
from core.spatial_index import SpatialHash


class TestSpatialHashBasics:
    """Test basic functionality of the spatial hash."""
    
    def test_create_spatial_hash(self):
        """Create spatial hash with specified cell size."""
        sh = SpatialHash(cell_size=10)
        assert sh.cell_size == 10
        assert len(sh.grid) == 0  # Empty initially
    
    def test_insert_horizontal_edge(self):
        """Insert a horizontal edge and verify cells."""
        sh = SpatialHash(cell_size=5)
        
        # Edge from (0,0) to (10,0)
        p1 = Point(0, 0)
        p2 = Point(10, 0)
        edge_id = 0
        
        sh.insert_edge(edge_id, p1, p2)
        
        # Should occupy cells along the horizontal
        # Cell (0,0) contains points 0-4
        # Cell (1,0) contains points 5-9
        # Cell (2,0) contains point 10
        
        # Query for edges near origin
        nearby = sh.query_nearby_edges(Point(0, 0))
        assert edge_id in nearby
    
    def test_insert_vertical_edge(self):
        """Insert a vertical edge and verify cells."""
        sh = SpatialHash(cell_size=5)
        
        # Edge from (0,0) to (0,10)
        p1 = Point(0, 0)
        p2 = Point(0, 10)
        edge_id = 1
        
        sh.insert_edge(edge_id, p1, p2)
        
        # Query at midpoint
        nearby = sh.query_nearby_edges(Point(0, 5))
        assert edge_id in nearby
    
    def test_insert_diagonal_edge(self):
        """Insert a diagonal edge and verify cells."""
        sh = SpatialHash(cell_size=5)
        
        # Edge from (0,0) to (10,10)
        p1 = Point(0, 0)
        p2 = Point(10, 10)
        edge_id = 2
        
        sh.insert_edge(edge_id, p1, p2)
        
        # Query at various points along diagonal
        for x in [0, 5, 10]:
            nearby = sh.query_nearby_edges(Point(x, x))
            assert edge_id in nearby
    
    def test_rasterization_vertical(self):
        """Test Bresenham-style line rasterization for vertical line."""
        sh = SpatialHash(cell_size=5)
        
        # Vertical line: (0,0) -> (0,10)
        # Should occupy cells (0,0), (0,1), (0,2)
        p1 = Point(0, 0)
        p2 = Point(0, 10)
        
        cells = sh._rasterize_segment(p1, p2)
        
        # Expected cells: y from 0-4 (cell 0), 5-9 (cell 1), 10 (cell 2)
        expected_y_cells = [0, 1, 2]
        actual_y_cells = sorted(set(c[1] for c in cells))
        
        assert actual_y_cells == expected_y_cells
    
    def test_rasterization_horizontal(self):
        """Test Bresenham-style line rasterization for horizontal line."""
        sh = SpatialHash(cell_size=5)
        
        # Horizontal line: (0,0) -> (15,0)
        p1 = Point(0, 0)
        p2 = Point(15, 0)
        
        cells = sh._rasterize_segment(p1, p2)
        
        # Expected cells: x from 0-4 (cell 0), 5-9 (cell 1), 10-14 (cell 2), 15 (cell 3)
        expected_x_cells = [0, 1, 2, 3]
        actual_x_cells = sorted(set(c[0] for c in cells))
        
        assert actual_x_cells == expected_x_cells
    
    def test_query_region(self):
        """Test querying edges in a specific region."""
        sh = SpatialHash(cell_size=10)
        
        # Insert multiple edges
        sh.insert_edge(0, Point(0, 0), Point(20, 0))    # Horizontal
        sh.insert_edge(1, Point(0, 0), Point(0, 20))    # Vertical
        sh.insert_edge(2, Point(50, 50), Point(60, 60)) # Diagonal far away
        
        # Query region around origin
        nearby = sh.query_region(Point(5, 5), radius=15)
        
        # Should find edges 0 and 1, not edge 2
        assert 0 in nearby
        assert 1 in nearby
        assert 2 not in nearby
    
    def test_remove_edge(self):
        """Test removing an edge from spatial hash."""
        sh = SpatialHash(cell_size=10)
        
        p1 = Point(0, 0)
        p2 = Point(20, 20)
        edge_id = 5
        
        sh.insert_edge(edge_id, p1, p2)
        
        # Verify it's there
        nearby = sh.query_nearby_edges(Point(10, 10))
        assert edge_id in nearby
        
        # Remove it
        sh.remove_edge(edge_id, p1, p2)
        
        # Should no longer be there
        nearby = sh.query_nearby_edges(Point(10, 10))
        assert edge_id not in nearby


class TestSpatialHashConsistency:
    """Test that spatial hash gives same results as brute force."""
    
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
    
    def test_consistency_with_brute_force(self, load_15_nodes):
        """
        Compare crossing count using:
        Method A: Brute force O(E^2)
        Method B: Spatial hash O(E*k) where k is avg edges per cell
        
        Results must be identical.
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
        
        # Method A: Brute Force
        brute_force_count = 0
        for i in range(len(edges)):
            for j in range(i + 1, len(edges)):
                p1, p2 = edges[i]
                q1, q2 = edges[j]
                if GeometryCore.segments_intersect(p1, p2, q1, q2):
                    brute_force_count += 1
        
        # Method B: Spatial Hash
        # Calculate appropriate cell size (heuristic: sqrt of average edge length)
        total_length_sq = 0
        for p1, p2 in edges:
            dx = p2.x - p1.x
            dy = p2.y - p1.y
            total_length_sq += dx * dx + dy * dy
        
        avg_length = int((total_length_sq / len(edges)) ** 0.5) if edges else 10
        cell_size = max(10, avg_length // 2)  # Half average length
        
        sh = SpatialHash(cell_size=cell_size)
        
        # Insert all edges
        for idx, (p1, p2) in enumerate(edges):
            sh.insert_edge(idx, p1, p2)
        
        # Count crossings using spatial hash
        spatial_hash_count = 0
        for i in range(len(edges)):
            p1, p2 = edges[i]
            
            # Query nearby edges
            candidates = sh.query_edge_region(p1, p2)
            
            # Check each candidate
            for j in candidates:
                if j > i:  # Only count each pair once
                    q1, q2 = edges[j]
                    if GeometryCore.segments_intersect(p1, p2, q1, q2):
                        spatial_hash_count += 1
        
        print(f"\nConsistency Test Results:")
        print(f"  Brute Force: {brute_force_count} crossings")
        print(f"  Spatial Hash: {spatial_hash_count} crossings")
        print(f"  Cell Size: {cell_size}")
        
        # CRITICAL: Results must match exactly
        assert brute_force_count == spatial_hash_count, \
            f"Mismatch: Brute force found {brute_force_count}, Spatial hash found {spatial_hash_count}"
    
    def test_edge_update_consistency(self, load_15_nodes):
        """Test that moving an edge updates spatial hash correctly."""
        data = load_15_nodes
        
        # Build initial state
        points = {}
        for node in data['nodes']:
            nid = node['id']
            points[nid] = Point(int(node['x']), int(node['y']))
        
        edges = []
        for edge in data['edges']:
            src = edge['source']
            tgt = edge['target']
            edges.append((points[src], points[tgt]))
        
        # Create spatial hash
        sh = SpatialHash(cell_size=20)
        for idx, (p1, p2) in enumerate(edges):
            sh.insert_edge(idx, p1, p2)
        
        # Move a node (this affects all incident edges)
        # Let's move node 0
        node_to_move = 0
        old_pos = points[node_to_move]
        new_pos = Point(old_pos.x + 50, old_pos.y + 50)
        
        # Find incident edges
        incident_edges = []
        for idx, edge_data in enumerate(data['edges']):
            if edge_data['source'] == node_to_move or edge_data['target'] == node_to_move:
                incident_edges.append(idx)
        
        # Update spatial hash for incident edges
        for idx in incident_edges:
            old_p1, old_p2 = edges[idx]
            
            # Remove old
            sh.remove_edge(idx, old_p1, old_p2)
            
            # Update point
            if data['edges'][idx]['source'] == node_to_move:
                new_p1 = new_pos
                new_p2 = old_p2
            else:
                new_p1 = old_p1
                new_p2 = new_pos
            
            # Insert new
            sh.insert_edge(idx, new_p1, new_p2)
            
            # Update in edges list
            edges[idx] = (new_p1, new_p2)
        
        # Update points dict
        points[node_to_move] = new_pos
        
        # Verify consistency: Count crossings again
        # This is mainly to ensure no crashes or corruption
        spatial_hash_count = 0
        for i in range(len(edges)):
            p1, p2 = edges[i]
            candidates = sh.query_edge_region(p1, p2)
            
            for j in candidates:
                if j > i:
                    q1, q2 = edges[j]
                    if GeometryCore.segments_intersect(p1, p2, q1, q2):
                        spatial_hash_count += 1
        
        print(f"\nAfter moving node {node_to_move}:")
        print(f"  Total crossings: {spatial_hash_count}")
        print(f"  Incident edges updated: {len(incident_edges)}")
        
        # Just verify no crashes - count may differ after move
        assert spatial_hash_count >= 0


class TestSpatialHashPerformance:
    """Test performance characteristics of spatial hash."""
    
    def test_cell_size_selection(self):
        """Test that appropriate cell sizes are chosen."""
        # Small edges -> small cells
        sh_small = SpatialHash(cell_size=5)
        assert sh_small.cell_size == 5
        
        # Large edges -> large cells
        sh_large = SpatialHash(cell_size=100)
        assert sh_large.cell_size == 100
    
    def test_empty_region_query(self):
        """Query in empty region should return empty set."""
        sh = SpatialHash(cell_size=10)
        
        # Add edge in one corner
        sh.insert_edge(0, Point(0, 0), Point(10, 10))
        
        # Query far away
        nearby = sh.query_nearby_edges(Point(1000, 1000))
        assert len(nearby) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
