"""
Sprint 2: Spatial Hash - Implementation
O(1) spatial indexing for fast collision detection and edge intersection queries.

Uses a 2D grid to partition space and Bresenham-style line rasterization
to determine which cells each edge occupies.
"""
from typing import Set, List, Tuple, Dict
from collections import defaultdict
from .geometry import Point, BoundingBox


class SpatialHash:
    """
    2D spatial hash for efficient edge intersection queries.
    
    Maps (cell_x, cell_y) -> Set[edge_ids] that occupy that cell.
    Uses Bresenham-style line rasterization to determine occupancy.
    
    Complexity:
    - Insert: O(L/cell_size) where L is edge length
    - Query: O(k) where k is number of edges in nearby cells
    - Remove: O(L/cell_size)
    """
    
    def __init__(self, cell_size: int = 10):
        """
        Initialize spatial hash with specified cell size.
        
        Args:
            cell_size: Width/height of each grid cell in coordinate units.
                      Smaller = more cells, more granular queries
                      Larger = fewer cells, more edges per cell
        """
        if cell_size <= 0:
            raise ValueError("cell_size must be positive")
        
        self.cell_size = cell_size
        self.grid: Dict[Tuple[int, int], Set[int]] = defaultdict(set)
        # Track which cells each edge occupies for fast removal
        self.edge_cells: Dict[int, Set[Tuple[int, int]]] = defaultdict(set)
    
    def _point_to_cell(self, p: Point) -> Tuple[int, int]:
        """Convert point to grid cell coordinates."""
        return (p.x // self.cell_size, p.y // self.cell_size)
    
    def _rasterize_segment(self, p1: Point, p2: Point) -> Set[Tuple[int, int]]:
        """
        Rasterize line segment to determine which cells it occupies.
        
        Uses Bresenham-style algorithm adapted for grid cells.
        Returns set of (cell_x, cell_y) tuples.
        """
        cells = set()
        
        # Get cell coordinates of endpoints
        cell1 = self._point_to_cell(p1)
        cell2 = self._point_to_cell(p2)
        
        # Add endpoint cells
        cells.add(cell1)
        cells.add(cell2)
        
        # If same cell, we're done
        if cell1 == cell2:
            return cells
        
        # Get all cells intersected by bounding box
        # This is a conservative approximation - may include cells not touched
        min_cx = min(cell1[0], cell2[0])
        max_cx = max(cell1[0], cell2[0])
        min_cy = min(cell1[1], cell2[1])
        max_cy = max(cell1[1], cell2[1])
        
        # For each cell in bounding box, check if segment passes through
        for cx in range(min_cx, max_cx + 1):
            for cy in range(min_cy, max_cy + 1):
                # Check if segment intersects this cell
                if self._segment_intersects_cell(p1, p2, cx, cy):
                    cells.add((cx, cy))
        
        return cells
    
    def _segment_intersects_cell(self, p1: Point, p2: Point, 
                                  cell_x: int, cell_y: int) -> bool:
        """
        Check if line segment intersects given cell.
        
        Uses separating axis theorem: segment and cell don't intersect
        if we can find an axis that separates them.
        """
        # Cell boundaries
        cell_min_x = cell_x * self.cell_size
        cell_min_y = cell_y * self.cell_size
        cell_max_x = (cell_x + 1) * self.cell_size
        cell_max_y = (cell_y + 1) * self.cell_size
        
        # Check if either endpoint is inside cell
        if (cell_min_x <= p1.x <= cell_max_x and cell_min_y <= p1.y <= cell_max_y):
            return True
        if (cell_min_x <= p2.x <= cell_max_x and cell_min_y <= p2.y <= cell_max_y):
            return True
        
        # Check if segment intersects cell boundaries
        # We need to check if segment intersects any of the 4 cell edges
        
        # Cell corners
        corners = [
            Point(cell_min_x, cell_min_y),
            Point(cell_max_x, cell_min_y),
            Point(cell_max_x, cell_max_y),
            Point(cell_min_x, cell_max_y)
        ]
        
        # Check intersection with cell edges
        for i in range(4):
            c1 = corners[i]
            c2 = corners[(i + 1) % 4]
            
            # Use simplified intersection test for axis-aligned cell edges
            if self._segment_intersects_box_edge(p1, p2, c1, c2):
                return True
        
        return False
    
    def _segment_intersects_box_edge(self, p1: Point, p2: Point,
                                     c1: Point, c2: Point) -> bool:
        """
        Check if segment p1-p2 intersects axis-aligned box edge c1-c2.
        Optimized for axis-aligned edges.
        """
        # Check if segment bounding box overlaps edge bounding box
        seg_min_x, seg_max_x = (p1.x, p2.x) if p1.x < p2.x else (p2.x, p1.x)
        seg_min_y, seg_max_y = (p1.y, p2.y) if p1.y < p2.y else (p2.y, p1.y)
        
        edge_min_x, edge_max_x = (c1.x, c2.x) if c1.x < c2.x else (c2.x, c1.x)
        edge_min_y, edge_max_y = (c1.y, c2.y) if c1.y < c2.y else (c2.y, c1.y)
        
        # Check bounding box overlap
        if seg_max_x < edge_min_x or seg_min_x > edge_max_x:
            return False
        if seg_max_y < edge_min_y or seg_min_y > edge_max_y:
            return False
        
        # For axis-aligned edges, use specialized checks
        if c1.x == c2.x:  # Vertical edge
            x = c1.x
            # Check if segment crosses x
            if seg_min_x <= x <= seg_max_x:
                # Calculate y at x
                if p1.x == p2.x:  # Segment is also vertical
                    return x == p1.x and not (seg_max_y < edge_min_y or seg_min_y > edge_max_y)
                else:
                    # Linear interpolation
                    t = (x - p1.x) * 1.0 / (p2.x - p1.x)
                    y = p1.y + t * (p2.y - p1.y)
                    return edge_min_y <= y <= edge_max_y
        
        elif c1.y == c2.y:  # Horizontal edge
            y = c1.y
            # Check if segment crosses y
            if seg_min_y <= y <= seg_max_y:
                # Calculate x at y
                if p1.y == p2.y:  # Segment is also horizontal
                    return y == p1.y and not (seg_max_x < edge_min_x or seg_min_x > edge_max_x)
                else:
                    # Linear interpolation
                    t = (y - p1.y) * 1.0 / (p2.y - p1.y)
                    x = p1.x + t * (p2.x - p1.x)
                    return edge_min_x <= x <= edge_max_x
        
        return False
    
    def insert_edge(self, edge_id: int, p1: Point, p2: Point):
        """
        Insert an edge into the spatial hash.
        
        Args:
            edge_id: Unique identifier for this edge
            p1, p2: Endpoints of the edge
        """
        cells = self._rasterize_segment(p1, p2)
        
        # Add edge to all occupied cells
        for cell in cells:
            self.grid[cell].add(edge_id)
        
        # Track cells for this edge
        self.edge_cells[edge_id] = cells
    
    def remove_edge(self, edge_id: int, p1: Point, p2: Point):
        """
        Remove an edge from the spatial hash.
        
        Args:
            edge_id: Unique identifier for this edge
            p1, p2: Endpoints of the edge (must match insert)
        """
        # Get cells this edge occupied
        cells = self.edge_cells.get(edge_id, set())
        
        # Remove from all cells
        for cell in cells:
            if cell in self.grid:
                self.grid[cell].discard(edge_id)
                # Clean up empty cells
                if not self.grid[cell]:
                    del self.grid[cell]
        
        # Remove from tracking
        if edge_id in self.edge_cells:
            del self.edge_cells[edge_id]
    
    def query_nearby_edges(self, point: Point) -> Set[int]:
        """
        Query all edges in the same cell as the given point.
        
        Args:
            point: Query point
            
        Returns:
            Set of edge IDs in the cell containing this point
        """
        cell = self._point_to_cell(point)
        return self.grid.get(cell, set()).copy()
    
    def query_region(self, center: Point, radius: int) -> Set[int]:
        """
        Query all edges in a circular region around a point.
        
        Args:
            center: Center of query region
            radius: Radius in coordinate units
            
        Returns:
            Set of edge IDs in cells within the region
        """
        edges = set()
        
        # Convert radius to cells (conservative - check extra cells)
        cell_radius = (radius // self.cell_size) + 1
        center_cell = self._point_to_cell(center)
        
        # Check all cells in bounding box
        for dx in range(-cell_radius, cell_radius + 1):
            for dy in range(-cell_radius, cell_radius + 1):
                cell = (center_cell[0] + dx, center_cell[1] + dy)
                if cell in self.grid:
                    edges.update(self.grid[cell])
        
        return edges
    
    def query_edge_region(self, p1: Point, p2: Point) -> Set[int]:
        """
        Query all edges that might intersect with edge p1-p2.
        
        Returns edges in all cells occupied by this edge, plus a buffer.
        
        Args:
            p1, p2: Endpoints of query edge
            
        Returns:
            Set of edge IDs that might intersect
        """
        candidates = set()
        
        # Get cells occupied by this edge
        cells = self._rasterize_segment(p1, p2)
        
        # Also check neighboring cells for safety
        expanded_cells = set(cells)
        for cx, cy in cells:
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    expanded_cells.add((cx + dx, cy + dy))
        
        # Collect all edges in these cells
        for cell in expanded_cells:
            if cell in self.grid:
                candidates.update(self.grid[cell])
        
        return candidates
    
    def clear(self):
        """Remove all edges from the spatial hash."""
        self.grid.clear()
        self.edge_cells.clear()
    
    def __len__(self):
        """Return number of edges in the spatial hash."""
        return len(self.edge_cells)
    
    def __repr__(self):
        """String representation for debugging."""
        return f"SpatialHash(cell_size={self.cell_size}, edges={len(self)}, cells={len(self.grid)})"
