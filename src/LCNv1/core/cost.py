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
    
    def _cross_product(self, ox: int, oy: int, ax: int, ay: int, bx: int, by: int) -> int:
        """Cross product (A-O) × (B-O) - 匹配 CUDA"""
        dx1, dy1 = ax - ox, ay - oy
        dx2, dy2 = bx - ox, by - oy
        return dx1 * dy2 - dy1 * dx2
    
    def _segments_properly_intersect_cuda(self, p1: Point, p2: Point, q1: Point, q2: Point) -> bool:
        """真交叉檢測（匹配 CUDA segments_intersect）"""
        # 共享端點 → False
        if ((p1.x == q1.x and p1.y == q1.y) or (p1.x == q2.x and p1.y == q2.y) or
            (p2.x == q1.x and p2.y == q1.y) or (p2.x == q2.x and p2.y == q2.y)):
            return False
        
        d1 = self._cross_product(p1.x, p1.y, p2.x, p2.y, q1.x, q1.y)
        d2 = self._cross_product(p1.x, p1.y, p2.x, p2.y, q2.x, q2.y)
        d3 = self._cross_product(q1.x, q1.y, q2.x, q2.y, p1.x, p1.y)
        d4 = self._cross_product(q1.x, q1.y, q2.x, q2.y, p2.x, p2.y)
        
        opposite_12 = (d1 < 0 and d2 > 0) or (d1 > 0 and d2 < 0)
        opposite_34 = (d3 < 0 and d4 > 0) or (d3 > 0 and d4 < 0)
        return opposite_12 and opposite_34
    
    def _point_on_segment_strict(self, a: Point, b: Point, c: Point) -> bool:
        """點在線段內部檢測（匹配 CUDA point_on_segment_interior）"""
        if self._cross_product(a.x, a.y, b.x, b.y, c.x, c.y) != 0:
            return False
        if (c.x == a.x and c.y == a.y) or (c.x == b.x and c.y == b.y):
            return False
        if c.x < min(a.x, b.x) or c.x > max(a.x, b.x):
            return False
        if c.y < min(a.y, b.y) or c.y > max(a.y, b.y):
            return False
        return True
    
    def _count_edge_through_node_violations(self, graph: GraphData, state: GridState,
                                           node_id: int, new_pos: Point) -> int:
        """
        計算邊穿過節點的違規（優化版本 - 只檢查與 node_id 相關的部分）
        
        優化策略：
        1. 只檢查與 node_id 相連的邊（Scenario 1）
        2. 只檢查 node_id 新位置附近的邊（使用 spatial hash）（Scenario 2）
        
        時間複雜度：O(d * n) → O(d + k) 其中 d = degree, k = nearby edges
        """
        violations = 0
        incident_edges = graph.get_incident_edges(node_id)
        
        # Scenario 1: 與 node_id 相連的邊是否穿過其他節點
        # 只需檢查 degree(node_id) 條邊
        for edge_idx in incident_edges:
            src, tgt = graph.get_edge_endpoints(edge_idx)
            
            # 獲取移動後的邊端點
            p1 = new_pos if src == node_id else state.get_position(src)
            p2 = new_pos if tgt == node_id else state.get_position(tgt)
            
            # 檢查這條邊是否穿過其他節點
            # 優化：使用 spatial hash 只檢查附近的節點（如果有實現）
            # 目前先全檢查（但只針對 incident edges，已經大幅減少）
            for nid in range(graph.num_nodes):
                if nid == src or nid == tgt:
                    continue
                
                node_pos = state.get_position(nid) if nid != node_id else new_pos
                if self._point_on_segment_strict(p1, p2, node_pos):
                    violations += 1
        
        # Scenario 2: 其他邊是否穿過 node_id 的新位置
        # 優化：使用 spatial hash 只檢查新位置附近的邊
        # 創建一個極小的查詢區域（new_pos 的單點）
        if hasattr(self, '_spatial_hash') and self._spatial_hash is not None:
            # 使用 spatial hash 查詢 new_pos 附近的邊
            # 查詢一個微小的區域（±1 像素）
            nearby_edges = self._spatial_hash.query_edge_region(
                Point(new_pos.x - 1, new_pos.y - 1),
                Point(new_pos.x + 1, new_pos.y + 1)
            )
            
            # 只檢查附近的邊
            for edge_idx in nearby_edges:
                if edge_idx in incident_edges:
                    continue  # 已在 Scenario 1 處理
                
                src, tgt = graph.get_edge_endpoints(edge_idx)
                p1 = state.get_position(src)
                p2 = state.get_position(tgt)
                
                if self._point_on_segment_strict(p1, p2, new_pos):
                    violations += 1
        else:
            # 沒有 spatial hash，退回全掃描（但跳過 incident edges）
            for edge_idx in range(graph.num_edges):
                if edge_idx in incident_edges:
                    continue
                
                src, tgt = graph.get_edge_endpoints(edge_idx)
                p1 = state.get_position(src)
                p2 = state.get_position(tgt)
                
                if self._point_on_segment_strict(p1, p2, new_pos):
                    violations += 1
        
        return violations
    
    def _count_duplicate_coords(self, graph: GraphData, state: GridState,
                               node_id: int, new_pos: Point) -> int:
        """
        檢測移動後是否有重複坐標（優化版本）
        
        優化：只檢查 new_pos 是否與其他節點重複
        時間複雜度：O(n) → O(1) with spatial hash
        """
        # 優化：使用 spatial hash 只檢查 new_pos 位置的節點
        # 這裡簡化版本：直接檢查所有節點（因為 O(n) 相對於交叉檢測已經很快）
        for nid in range(graph.num_nodes):
            if nid != node_id:
                other_pos = state.get_position(nid)
                if other_pos.x == new_pos.x and other_pos.y == new_pos.y:
                    return 1  # 一旦發現就返回
        return 0
    
    def calculate(self, graph: GraphData, state: GridState) -> float:
        """
        Cross product of vectors OA and OB: (A-O) × (B-O)
        匹配 CUDA 實現
        """
        dx1 = ax - ox
        dy1 = ay - oy
        dx2 = bx - ox
        dy2 = by - oy
        return dx1 * dy2 - dy1 * dx2
    
    def _segments_properly_intersect_cuda(self, p1: Point, p2: Point,
                                         q1: Point, q2: Point) -> bool:
        """
        真交叉檢測：兩線段在內部相交（嚴格匹配 CUDA segments_intersect）
        """
        # 快速拒絕：共享端點
        if ((p1.x == q1.x and p1.y == q1.y) or (p1.x == q2.x and p1.y == q2.y) or
            (p2.x == q1.x and p2.y == q1.y) or (p2.x == q2.x and p2.y == q2.y)):
            return False
        
        # 計算 cross products
        d1 = self._cross_product(p1.x, p1.y, p2.x, p2.y, q1.x, q1.y)
        d2 = self._cross_product(p1.x, p1.y, p2.x, p2.y, q2.x, q2.y)
        d3 = self._cross_product(q1.x, q1.y, q2.x, q2.y, p1.x, p1.y)
        d4 = self._cross_product(q1.x, q1.y, q2.x, q2.y, p2.x, p2.y)
        
        # 對立邊測試（嚴格不等，排除 0）
        opposite_12 = (d1 < 0 and d2 > 0) or (d1 > 0 and d2 < 0)
        opposite_34 = (d3 < 0 and d4 > 0) or (d3 > 0 and d4 < 0)
        
        return opposite_12 and opposite_34
    
    def _point_on_segment_strict(self, a: Point, b: Point, c: Point) -> bool:
        """
        判斷點 c 是否在線段 ab 的「內部」（匹配 CUDA point_on_segment_interior）
        """
        # 共線性檢查
        if self._cross_product(a.x, a.y, b.x, b.y, c.x, c.y) != 0:
            return False
        
        # 排除端點
        if (c.x == a.x and c.y == a.y) or (c.x == b.x and c.y == b.y):
            return False
        
        # 包含於邊的包圍盒
        if c.x < min(a.x, b.x) or c.x > max(a.x, b.x):
            return False
        if c.y < min(a.y, b.y) or c.y > max(a.y, b.y):
            return False
        
        return True
    
    def _count_edge_through_node_violations(self, graph: GraphData, state: GridState,
                                           node_id: int, new_pos: Point) -> int:
        """
        計算「邊穿過非端點節點」的違規數量（匹配 CUDA count_edge_through_node_violations）
        """
        def get_pos(v: int) -> Point:
            return new_pos if v == node_id else state.get_position(v)
        
        violations = 0
        
        # Scenario 1: 與 node_id 相連的邊穿過其他節點
        incident_edges = graph.get_incident_edges(node_id)
        for edge_idx in incident_edges:
            src, tgt = graph.get_edge_endpoints(edge_idx)
            p1 = get_pos(src)
            p2 = get_pos(tgt)
            
            # 檢查所有其他節點
            for nid in range(graph.num_nodes):
                if nid == src or nid == tgt:
                    continue
                
                c = get_pos(nid)
                if self._point_on_segment_strict(p1, p2, c):
                    violations += 1
        
        # Scenario 2: 其他邊穿過 node_id 的新位置
        for edge_idx in range(graph.num_edges):
            src, tgt = graph.get_edge_endpoints(edge_idx)
            
            # 跳過與 node_id 相連的邊
            if src == node_id or tgt == node_id:
                continue
            
            p1 = state.get_position(src)
            p2 = state.get_position(tgt)
            
            if self._point_on_segment_strict(p1, p2, new_pos):
                violations += 1
        
        return violations
    
    def _count_duplicate_coords(self, graph: GraphData, state: GridState,
                               node_id: int, new_pos: Point) -> int:
        """
        檢測移動後是否有重複坐標
        """
        violations = 0
        
        for nid in range(graph.num_nodes):
            if nid == node_id:
                continue
            
            other_pos = state.get_position(nid)
            if other_pos.x == new_pos.x and other_pos.y == new_pos.y:
                violations += 1
        
        return violations
    
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
                    
                    # Skip edges that share endpoints (proper intersections only)
                    if src_i in (src_j, tgt_j) or tgt_i in (src_j, tgt_j):
                        continue
                    
                    q1 = state.get_position(src_j)
                    q2 = state.get_position(tgt_j)
                    
                    if self._segments_properly_intersect_cuda(p1, p2, q1, q2):
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
        
        匹配 CUDA compute_delta_e：先檢查違規，有違規則返回 +inf
        
        優化：
        1. 先檢查重複坐標（O(n)，快速失敗）
        2. 再檢查邊穿過節點（O(d*n + k)，較慢）
        3. 兩者都通過才計算真實 delta
        
        Time: O(d * k) where d = degree of node, k = avg edges per cell
        
        Only affected edges are those incident to node_id.
        """
        # CRITICAL: 先檢查違規（匹配 CUDA）
        # 優化順序：先檢查快的（重複坐標），再檢查慢的（邊穿過節點）
        
        # 1. 檢查「節點重合」違規（快速失敗，O(n)）
        dup_violations = self._count_duplicate_coords(
            graph, state, node_id, new_pos
        )
        
        if dup_violations > 0:
            # 有重複坐標 → 立即拒絕
            return math.inf
        
        # 2. 檢查「邊穿過節點」違規（較慢，O(d*n + k)）
        edge_violations = self._count_edge_through_node_violations(
            graph, state, node_id, new_pos
        )
        
        if edge_violations > 0:
            # 有違規 → 立即拒絕
            return math.inf
        
        # 3. 無違規 → 計算真實的 cost delta
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
                
                # 使用 CUDA 風格的相交檢測
                if self._segments_properly_intersect_cuda(p1_old, p2_old, q1, q2):
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
                
                # 使用 CUDA 風格的相交檢測
                if self._segments_properly_intersect_cuda(p1_new, p2_new, q1, q2):
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
                
                if self._segments_properly_intersect_cuda(p1, p2, q1, q2):
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
                
                if self._segments_properly_intersect_cuda(p1, p2, q1, q2):
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
                    
                    # Skip edges that share endpoints
                    if src_i in (src_j, tgt_j) or tgt_i in (src_j, tgt_j):
                        continue
                    
                    q1 = state.get_position(src_j)
                    q2 = state.get_position(tgt_j)
                    
                    # 使用 CUDA 風格的相交檢測
                    if self._segments_properly_intersect_cuda(p1, p2, q1, q2):
                        edge_crossings[i] += 1
                        edge_crossings[j] += 1
                        total_crossings += 1
        
        k = max(edge_crossings) if edge_crossings else 0
        
        return (k, total_crossings)
    
    def get_length_energy(self, graph: GraphData, state: GridState) -> float:
        """Get total length energy."""
        return self._calculate_length_energy(graph, state)
