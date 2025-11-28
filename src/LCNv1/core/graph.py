"""
Sprint 3/4: Graph Data Structures
Separates immutable topology (GraphData) from mutable positions (GridState).
"""
from typing import List, Tuple, Set, Dict
from .geometry import Point


class GraphData:
    """
    Immutable graph topology.
    Stores edges and maintains adjacency information.
    """
    
    def __init__(self, num_nodes: int, edges: List[Tuple[int, int]]):
        """
        Initialize graph topology.
        
        Args:
            num_nodes: Number of nodes in the graph
            edges: List of (source, target) tuples
        """
        self.num_nodes = num_nodes
        self.edges = edges
        self.num_edges = len(edges)
        
        # Build adjacency structure: node_id -> list of edge indices
        self._adjacency: Dict[int, List[int]] = {i: [] for i in range(num_nodes)}
        
        for edge_idx, (src, tgt) in enumerate(edges):
            self._adjacency[src].append(edge_idx)
            self._adjacency[tgt].append(edge_idx)
    
    def get_incident_edges(self, node_id: int) -> List[int]:
        """
        Get all edge indices incident to a node.
        
        Args:
            node_id: Node identifier
            
        Returns:
            List of edge indices
        """
        return self._adjacency.get(node_id, [])
    
    def get_edge_endpoints(self, edge_idx: int) -> Tuple[int, int]:
        """
        Get endpoints of an edge.
        
        Args:
            edge_idx: Edge index
            
        Returns:
            (source, target) tuple
        """
        return self.edges[edge_idx]
    
    def get_degree(self, node_id: int) -> int:
        """Get degree (number of incident edges) of a node."""
        return len(self._adjacency.get(node_id, []))
    
    def __repr__(self):
        return f"GraphData(nodes={self.num_nodes}, edges={self.num_edges})"


class GridState:
    """
    Mutable grid state - node positions on integer grid.
    Maintains boundary constraints and collision detection.
    """
    
    def __init__(self, positions: Dict[int, Point], width: int, height: int):
        """
        Initialize grid state.
        
        Args:
            positions: Dictionary mapping node_id -> Point
            width: Grid width
            height: Grid height
        """
        self.width = width
        self.height = height
        self._positions = positions.copy()
        
        # Build reverse map for collision detection
        self._location_to_node: Dict[Point, int] = {}
        for node_id, pos in positions.items():
            self._location_to_node[pos] = node_id
    
    def get_position(self, node_id: int) -> Point:
        """Get current position of a node."""
        return self._positions[node_id]
    
    def get_all_positions(self) -> Dict[int, Point]:
        """Get copy of all positions."""
        return self._positions.copy()
    
    def move_node(self, node_id: int, new_pos: Point):
        """
        Move a node to a new position.
        
        Args:
            node_id: Node to move
            new_pos: New position
        """
        # Validate bounds
        if not (0 <= new_pos.x <= self.width and 0 <= new_pos.y <= self.height):
            raise ValueError(f"Position {new_pos} out of bounds ({self.width}x{self.height})")
        
        # Remove from old location
        old_pos = self._positions.get(node_id)
        if old_pos and old_pos in self._location_to_node:
            if self._location_to_node[old_pos] == node_id:
                del self._location_to_node[old_pos]
        
        # Update position
        self._positions[node_id] = new_pos
        self._location_to_node[new_pos] = node_id
    
    def is_occupied(self, pos: Point) -> bool:
        """Check if a position is occupied by a node."""
        return pos in self._location_to_node
    
    def get_bounding_box(self) -> Tuple[Point, Point]:
        """
        Get bounding box of all nodes.
        
        Returns:
            (min_point, max_point) tuple
        """
        if not self._positions:
            return (Point(0, 0), Point(0, 0))
        
        min_x = min(p.x for p in self._positions.values())
        min_y = min(p.y for p in self._positions.values())
        max_x = max(p.x for p in self._positions.values())
        max_y = max(p.y for p in self._positions.values())
        
        return (Point(min_x, min_y), Point(max_x, max_y))
    
    def clone(self) -> 'GridState':
        """Create a deep copy of this state."""
        return GridState(self._positions.copy(), self.width, self.height)
    
    def copy(self) -> 'GridState':
        """Alias for clone() for backward compatibility."""
        return self.clone()
    
    def __repr__(self):
        bbox_min, bbox_max = self.get_bounding_box()
        return (f"GridState(nodes={len(self._positions)}, "
                f"bounds={self.width}x{self.height}, "
                f"bbox=({bbox_min.x},{bbox_min.y})-({bbox_max.x},{bbox_max.y}))")
