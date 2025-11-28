"""
Sprint 3: Cost Function with Delta Updates
Implements energy function: W_cross * Σk^p + W_len * Σlen^2

The delta calculation is THE KEY to fast simulated annealing.
"""
from abc import ABC, abstractmethod
from typing import Tuple
import math

from .geometry import Point, GeometryCore
from .graph import GraphData, GridState
from .spatial_index import SpatialHash


class ICostFunction(ABC):
    """
    Interface for cost functions.
    All cost functions must implement full calculation and delta calculation.
    """
    
    @abstractmethod
    def calculate(self, graph: GraphData, state: GridState) -> float:
        """
        Calculate total cost from scratch.
        
        Args:
            graph: Graph topology
            state: Current node positions
            
        Returns:
            Total energy/cost
        """
        pass
    
    @abstractmethod
    def calculate_delta(self, graph: GraphData, state: GridState, 
                       node_id: int, new_pos: Point) -> float:
        """
        Calculate change in cost if node_id moves to new_pos.
        
        CRITICAL: Must be mathematically exact.
        cost(after_move) - cost(before_move) == calculate_delta
        
        Args:
            graph: Graph topology
            state: Current node positions (not modified)
            node_id: Node to move
            new_pos: Proposed new position
            
        Returns:
            Delta in cost (can be negative)
        """
        pass


class SoftMaxCost(ICostFunction):
    """
    Cost function: W_cross * Σ(k^p) + W_len * Σ(len^2)
    
    Where:
    - k = number of crossings for each edge
    - p = power (typically 2)
    - len = edge length
    
    Uses spatial hash for O(d*k) delta calculation instead of O(E).
    """
    
    def __init__(self, w_cross: float = 100.0, w_len: float = 1.0, 
                 power: int = 2, cell_size: int = 50):
        """
        Initialize cost function.
        
        Args:
            w_cross: Weight for crossing penalty
            w_len: Weight for edge length penalty
            power: Exponent for crossing penalty (k^power)
            cell_size: Cell size for spatial hash
        """
        self.w_cross = w_cross
        self.w_len = w_len
        self.power = power
        self.cell_size = cell_size
        
        # Spatial hash for fast intersection queries
        self._spatial_hash = None
        self._graph = None
        self._state = None
    
    def _ensure_spatial_hash(self, graph: GraphData, state: GridState):
        """Build or rebuild spatial hash if needed."""
        # Always rebuild for now to ensure correctness
        # TODO: Implement incremental updates for better performance
        self._spatial_hash = SpatialHash(cell_size=self.cell_size)
        self._graph = graph
        self._state = state
        
        # Insert all edges
        for edge_idx in range(graph.num_edges):
            src, tgt = graph.get_edge_endpoints(edge_idx)
            p1 = state.get_position(src)
            p2 = state.get_position(tgt)
            self._spatial_hash.insert_edge(edge_idx, p1, p2)
    
    def calculate(self, graph: GraphData, state: GridState) -> float:
        """
        Calculate total cost from scratch.
        Time: O(E^2) worst case, O(E*k) average with spatial hash
        """
        # Ensure spatial hash is up to date
        self._ensure_spatial_hash(graph, state)
        
        # Calculate crossing energy
        crossing_energy = self._calculate_crossing_energy(graph, state)
        
        # Calculate length energy
        length_energy = self._calculate_length_energy(graph, state)
        
        return crossing_energy + length_energy
    
    def _calculate_crossing_energy(self, graph: GraphData, state: GridState) -> float:
        """Calculate W_cross * Σ(k^p) where k is crossings per edge."""
        edge_crossings = [0] * graph.num_edges
        
        # Count crossings for each edge
        for i in range(graph.num_edges):
            src_i, tgt_i = graph.get_edge_endpoints(i)
            p1 = state.get_position(src_i)
            p2 = state.get_position(tgt_i)
            
            # Query nearby edges using spatial hash
            candidates = self._spatial_hash.query_edge_region(p1, p2)
            
            for j in candidates:
                if j > i:  # Only count each pair once
                    src_j, tgt_j = graph.get_edge_endpoints(j)
                    q1 = state.get_position(src_j)
                    q2 = state.get_position(tgt_j)
                    
                    if GeometryCore.segments_intersect(p1, p2, q1, q2):
                        edge_crossings[i] += 1
                        edge_crossings[j] += 1
        
        # Calculate energy: Σ(k^p)
        energy = sum(k ** self.power for k in edge_crossings)
        
        return self.w_cross * energy
    
    def _calculate_length_energy(self, graph: GraphData, state: GridState) -> float:
        """Calculate W_len * Σ(len^2)."""
        total = 0.0
        
        for edge_idx in range(graph.num_edges):
            src, tgt = graph.get_edge_endpoints(edge_idx)
            p1 = state.get_position(src)
            p2 = state.get_position(tgt)
            
            # Length squared (avoiding sqrt)
            dx = p2.x - p1.x
            dy = p2.y - p1.y
            len_sq = dx * dx + dy * dy
            
            total += len_sq
        
        return self.w_len * total
    
    def calculate_delta(self, graph: GraphData, state: GridState, 
                       node_id: int, new_pos: Point) -> float:
        """
        Calculate delta in cost for moving node_id to new_pos.
        
        Time: O(d * k) where d = degree of node, k = avg edges per cell
        
        Only affected edges are those incident to node_id.
        """
        # Ensure spatial hash is up to date
        self._ensure_spatial_hash(graph, state)
        
        old_pos = state.get_position(node_id)
        
        # Get incident edges
        incident_edges = graph.get_incident_edges(node_id)
        
        if not incident_edges:
            return 0.0  # No incident edges, no change
        
        # Calculate delta in crossing energy
        delta_crossing = self._calculate_delta_crossing(
            graph, state, node_id, old_pos, new_pos, incident_edges
        )
        
        # Calculate delta in length energy
        delta_length = self._calculate_delta_length(
            graph, state, node_id, old_pos, new_pos, incident_edges
        )
        
        return delta_crossing + delta_length
    
    def _calculate_delta_crossing(self, graph: GraphData, state: GridState,
                                   node_id: int, old_pos: Point, new_pos: Point,
                                   incident_edges: list) -> float:
        """
        Calculate delta in crossing energy.
        
        Strategy:
        1. For each incident edge, find all edges it crosses in OLD config
        2. For each incident edge, find all edges it crosses in NEW config  
        3. Affected edges = union of all edges that cross incident edges
        4. For each affected edge, calculate old k and new k
        5. Delta = sum((new_k^p - old_k^p)) for all affected edges
        """
        # Track all edges affected by this move
        affected_edges = set()
        
        # Maps: edge_id -> (old_crossings_with_incident, new_crossings_with_incident)
        incident_edge_data = {}
        
        # Step 1: For each incident edge, count crossings in both configs
        for edge_idx in incident_edges:
            src, tgt = graph.get_edge_endpoints(edge_idx)
            
            # Old configuration
            p1_old = old_pos if src == node_id else state.get_position(src)
            p2_old = old_pos if tgt == node_id else state.get_position(tgt)
            
            # New configuration
            p1_new = new_pos if src == node_id else state.get_position(src)
            p2_new = new_pos if tgt == node_id else state.get_position(tgt)
            
            # Find edges crossed in OLD config
            old_crossed = set()
            candidates_old = self._spatial_hash.query_edge_region(p1_old, p2_old)
            for other_idx in candidates_old:
                if other_idx == edge_idx:
                    continue
                if other_idx in incident_edges:
                    continue  # Skip other incident edges (handled separately)
                    
                other_src, other_tgt = graph.get_edge_endpoints(other_idx)
                q1 = state.get_position(other_src)
                q2 = state.get_position(other_tgt)
                
                if GeometryCore.segments_intersect(p1_old, p2_old, q1, q2):
                    old_crossed.add(other_idx)
                    affected_edges.add(other_idx)
            
            # Find edges crossed in NEW config
            new_crossed = set()
            candidates_new = self._get_candidates_for_moved_edge(p1_new, p2_new, p1_old, p2_old)
            for other_idx in candidates_new:
                if other_idx == edge_idx:
                    continue
                if other_idx in incident_edges:
                    continue  # Skip other incident edges
                    
                other_src, other_tgt = graph.get_edge_endpoints(other_idx)
                q1 = state.get_position(other_src)
                q2 = state.get_position(other_tgt)
                
                if GeometryCore.segments_intersect(p1_new, p2_new, q1, q2):
                    new_crossed.add(other_idx)
                    affected_edges.add(other_idx)
            
            incident_edge_data[edge_idx] = (old_crossed, new_crossed)
        
        # Step 2: Calculate old and new crossing counts for all affected edges
        delta = 0.0
        
        # For incident edges
        for edge_idx in incident_edges:
            old_crossed, new_crossed = incident_edge_data[edge_idx]
            old_k = len(old_crossed)
            new_k = len(new_crossed)
            
            delta += (new_k ** self.power) - (old_k ** self.power)
        
        # For non-incident affected edges
        for edge_idx in affected_edges:
            # Count how many incident edges this edge crossed before and after
            old_count = 0
            new_count = 0
            
            for inc_edge_idx in incident_edges:
                old_crossed, new_crossed = incident_edge_data[inc_edge_idx]
                if edge_idx in old_crossed:
                    old_count += 1
                if edge_idx in new_crossed:
                    new_count += 1
            
            # This edge's k changes by the difference
            # But we need its TOTAL k, not just crossings with incident edges
            # So we need to count its OTHER crossings too
            
            # Get full crossing count for this edge
            base_crossings = self._count_edge_crossings_excluding(
                graph, state, edge_idx, set(incident_edges)
            )
            
            old_k_total = base_crossings + old_count
            new_k_total = base_crossings + new_count
            
            delta += (new_k_total ** self.power) - (old_k_total ** self.power)
        
        return self.w_cross * delta
    
    def _calculate_delta_length(self, graph: GraphData, state: GridState,
                                node_id: int, old_pos: Point, new_pos: Point,
                                incident_edges: list) -> float:
        """
        Calculate delta in length energy.
        
        Only incident edges change length.
        """
        delta = 0.0
        
        for edge_idx in incident_edges:
            src, tgt = graph.get_edge_endpoints(edge_idx)
            
            # Get other endpoint
            other_id = tgt if src == node_id else src
            other_pos = state.get_position(other_id)
            
            # Old length squared
            dx_old = old_pos.x - other_pos.x
            dy_old = old_pos.y - other_pos.y
            len_sq_old = dx_old * dx_old + dy_old * dy_old
            
            # New length squared
            dx_new = new_pos.x - other_pos.x
            dy_new = new_pos.y - other_pos.y
            len_sq_new = dx_new * dx_new + dy_new * dy_new
            
            delta += len_sq_new - len_sq_old
        
        return self.w_len * delta
    
    def _count_edge_crossings(self, graph: GraphData, state: GridState, 
                             edge_idx: int) -> int:
        """Count crossings for a single edge."""
        src, tgt = graph.get_edge_endpoints(edge_idx)
        p1 = state.get_position(src)
        p2 = state.get_position(tgt)
        
        candidates = self._spatial_hash.query_edge_region(p1, p2)
        
        count = 0
        for other_idx in candidates:
            if other_idx != edge_idx:
                other_src, other_tgt = graph.get_edge_endpoints(other_idx)
                q1 = state.get_position(other_src)
                q2 = state.get_position(other_tgt)
                
                if GeometryCore.segments_intersect(p1, p2, q1, q2):
                    count += 1
        
        return count
    
    def _count_edge_crossings_excluding(self, graph: GraphData, state: GridState, 
                                       edge_idx: int, exclude_edges: set) -> int:
        """Count crossings for an edge, excluding specified edges."""
        src, tgt = graph.get_edge_endpoints(edge_idx)
        p1 = state.get_position(src)
        p2 = state.get_position(tgt)
        
        candidates = self._spatial_hash.query_edge_region(p1, p2)
        
        count = 0
        for other_idx in candidates:
            if other_idx != edge_idx and other_idx not in exclude_edges:
                other_src, other_tgt = graph.get_edge_endpoints(other_idx)
                q1 = state.get_position(other_src)
                q2 = state.get_position(other_tgt)
                
                if GeometryCore.segments_intersect(p1, p2, q1, q2):
                    count += 1
        
        return count
    
    def _get_candidates_for_moved_edge(self, p1_new: Point, p2_new: Point,
                                       p1_old: Point, p2_old: Point) -> set:
        """
        Get candidate edges for intersection with moved edge.
        Queries both old and new positions to be safe.
        """
        candidates = set()
        
        # Query new position
        candidates.update(self._spatial_hash.query_edge_region(p1_new, p2_new))
        
        # Also query old position in case we miss edges in transition
        candidates.update(self._spatial_hash.query_edge_region(p1_old, p2_old))
        
        return candidates
    
    def get_crossing_stats(self, graph: GraphData, state: GridState) -> Tuple[int, int]:
        """
        Get crossing statistics.
        
        Returns:
            (k, total) where k is max crossings on any edge, 
            total is total number of crossings
        """
        self._ensure_spatial_hash(graph, state)
        
        edge_crossings = [0] * graph.num_edges
        total_crossings = 0
        
        for i in range(graph.num_edges):
            src_i, tgt_i = graph.get_edge_endpoints(i)
            p1 = state.get_position(src_i)
            p2 = state.get_position(tgt_i)
            
            candidates = self._spatial_hash.query_edge_region(p1, p2)
            
            for j in candidates:
                if j > i:
                    src_j, tgt_j = graph.get_edge_endpoints(j)
                    q1 = state.get_position(src_j)
                    q2 = state.get_position(tgt_j)
                    
                    if GeometryCore.segments_intersect(p1, p2, q1, q2):
                        edge_crossings[i] += 1
                        edge_crossings[j] += 1
                        total_crossings += 1
        
        k = max(edge_crossings) if edge_crossings else 0
        
        return (k, total_crossings)
    
    def get_length_energy(self, graph: GraphData, state: GridState) -> float:
        """Get total length energy."""
        return self._calculate_length_energy(graph, state)
