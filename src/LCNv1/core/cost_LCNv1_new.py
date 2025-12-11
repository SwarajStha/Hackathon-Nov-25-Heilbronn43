"""
Sprint 3: Cost Function with Delta Updates
Implements energy function: W_cross * Σk^p + W_len * Σlen^2

The delta calculation is THE KEY to fast simulated annealing.
"""
from abc import ABC, abstractmethod
from typing import Tuple
import math

from geometry import Point, GeometryCore
from graph import GraphData, GridState
from spatial_index import SpatialHash


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
    Cost function for GD contest objective:

    Minimize K where
        K = max_e crossings(e)

    Cost:
        cost(G, pos) = w_cross * K

    Rules:
    - Crossings are counted pairwise between edges.
    - Edges that only meet at a common endpoint are NOT crossings.
    - If two distinct vertices have identical coordinates -> invalid, cost = +inf.
    - If a vertex that is not an endpoint of an edge lies in the interior of that
      edge -> invalid, cost = +inf.

    Note:
    - w_len and power are kept for API compatibility but not used in the cost.
    """

    def __init__(self, w_cross: float = 100.0, w_len: float = 1.0,
                 power: int = 2, cell_size: int = 50):
        self.w_cross = w_cross
        self.w_len = w_len    # kept for compatibility, not used
        self.power = power    # kept for compatibility, not used
        self.cell_size = cell_size

        # Spatial hash for fast intersection queries
        self._spatial_hash = None
        self._graph = None
        self._state = None

    # ----------------------------------------------------------------------
    # 基础工具函数
    # ----------------------------------------------------------------------
    def _ensure_spatial_hash(self, graph: GraphData, state: GridState):
        """Build or rebuild spatial hash for the current state."""
        self._spatial_hash = SpatialHash(cell_size=self.cell_size)
        self._graph = graph
        self._state = state

        for edge_idx in range(graph.num_edges):
            src, tgt = graph.get_edge_endpoints(edge_idx)
            p1 = state.get_position(src)
            p2 = state.get_position(tgt)
            self._spatial_hash.insert_edge(edge_idx, p1, p2)

    def _get_num_nodes(self, graph: GraphData, state: GridState) -> int:
        """Try to get number of nodes from graph/state."""
        if hasattr(graph, "num_nodes"):
            return graph.num_nodes
        if hasattr(state, "num_nodes"):
            return state.num_nodes
        if hasattr(state, "num_vertices"):
            return state.num_vertices
        raise AttributeError("Cannot determine number of nodes from graph/state.")

    def _cross_product(self, ox: int, oy: int, ax: int, ay: int, bx: int, by: int) -> int:
        """
        Cross product of vectors OA and OB: (A-O) × (B-O)
        
        匹配 CUDA 實現：
        cross = (ax - ox) * (by - oy) - (ay - oy) * (bx - ox)
        
        Returns:
            > 0: Counter-clockwise turn (B is left of OA)
            < 0: Clockwise turn (B is right of OA)
            = 0: Collinear (O, A, B on same line)
        """
        dx1 = ax - ox
        dy1 = ay - oy
        dx2 = bx - ox
        dy2 = by - oy
        return dx1 * dy2 - dy1 * dx2

    def _segments_properly_intersect(self, p1: Point, p2: Point,
                                     q1: Point, q2: Point) -> bool:
        """
        真交叉檢測：兩線段在內部相交（嚴格匹配 CUDA 實現）
        
        Algorithm (from planar_cuda.cu):
        - 檢查共享端點 → 返回 False
        - 計算 cross products: d1, d2, d3, d4
        - 檢查對立邊條件：
          * q1 和 q2 在 p1-p2 兩側：(d1 < 0 && d2 > 0) || (d1 > 0 && d2 < 0)
          * p1 和 p2 在 q1-q2 兩側：(d3 < 0 && d4 > 0) || (d3 > 0 && d4 < 0)
        - 必須嚴格不等（排除共線/接觸）
        """
        # 快速拒絕：共享端點
        if ((p1.x == q1.x and p1.y == q1.y) or (p1.x == q2.x and p1.y == q2.y) or
            (p2.x == q1.x and p2.y == q1.y) or (p2.x == q2.x and p2.y == q2.y)):
            return False
        
        # 計算 cross products（匹配 CUDA）
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
        判斷點 c 是否在線段 ab 的「內部」（嚴格意義上的內部，不含端點）
        用於「頂點落在邊上 → 非法」的檢測
        
        匹配 CUDA point_on_segment_interior 邏輯
        """
        # 共線性檢查（使用 cross_product）
        if self._cross_product(a.x, a.y, b.x, b.y, c.x, c.y) != 0:
            return False

        # 排除端點
        if (c.x == a.x and c.y == a.y) or (c.x == b.x and c.y == b.y):
            return False

        # 包含於邊的包圍盒（內部）
        if c.x < min(a.x, b.x) or c.x > max(a.x, b.x):
            return False
        if c.y < min(a.y, b.y) or c.y > max(a.y, b.y):
            return False

        return True

    def _count_duplicate_coords(self, graph: GraphData, state: GridState, get_pos) -> int:
        """
        檢測重複坐標數量（匹配 CUDA）
        返回：重複坐標的節點對數量
        """
        n = self._get_num_nodes(graph, state)
        seen = {}
        violations = 0
        
        for v in range(n):
            p = get_pos(v)
            key = (p.x, p.y)
            if key in seen:
                violations += 1  # 每個重複計 1
            seen[key] = v
        
        return violations
    
    def _count_edge_through_node_violations(self, graph: GraphData, state: GridState,
                                           node_id: int, new_pos: Point) -> int:
        """
        計算「邊穿過非端點節點」的違規數量（優化版本）
        
        優化策略：
        1. Scenario 1: 只檢查與 node_id 相連的邊（degree(node_id) 條）
        2. Scenario 2: 使用 spatial hash 只檢查 new_pos 附近的邊
        
        時間複雜度：O(d*n + E) → O(d*n + k) 其中 k = nearby edges << E
        """
        n = self._get_num_nodes(graph, state)
        violations = 0
        
        # 獲取與 node_id 相連的邊（只有這些邊會受影響）
        incident_edge_indices = []
        for edge_idx in range(graph.num_edges):
            src, tgt = graph.get_edge_endpoints(edge_idx)
            if src == node_id or tgt == node_id:
                incident_edge_indices.append(edge_idx)
        
        # Scenario 1: 與 node_id 相連的邊穿過其他節點
        for edge_idx in incident_edge_indices:
            src, tgt = graph.get_edge_endpoints(edge_idx)
            
            # 獲取移動後的邊端點
            p1 = new_pos if src == node_id else state.get_position(src)
            p2 = new_pos if tgt == node_id else state.get_position(tgt)
            
            # 檢查這條邊是否穿過其他節點
            for nid in range(n):
                if nid == src or nid == tgt:
                    continue
                
                node_pos = state.get_position(nid) if nid != node_id else new_pos
                if self._point_on_segment_strict(p1, p2, node_pos):
                    violations += 1
        
        # Scenario 2: 其他邊穿過 node_id 的新位置
        # 優化：使用 spatial hash（如果可用）
        if hasattr(self, '_spatial_hash') and self._spatial_hash is not None:
            # 查詢 new_pos 附近的邊（±1 像素的微小區域）
            nearby_edges = self._spatial_hash.query_edge_region(
                Point(new_pos.x - 1, new_pos.y - 1),
                Point(new_pos.x + 1, new_pos.y + 1)
            )
            
            for edge_idx in nearby_edges:
                # 跳過與 node_id 相連的邊
                if edge_idx in incident_edge_indices:
                    continue
                
                src, tgt = graph.get_edge_endpoints(edge_idx)
                p1 = state.get_position(src)
                p2 = state.get_position(tgt)
                
                if self._point_on_segment_strict(p1, p2, new_pos):
                    violations += 1
        else:
            # 沒有 spatial hash，全掃描（但跳過 incident edges）
            for edge_idx in range(graph.num_edges):
                if edge_idx in incident_edge_indices:
                    continue
                
                src, tgt = graph.get_edge_endpoints(edge_idx)
                p1 = state.get_position(src)
                p2 = state.get_position(tgt)
                
                if self._point_on_segment_strict(p1, p2, new_pos):
                    violations += 1
        
        return violations
    
    def _has_illegal_configuration(self, graph: GraphData,
                                   state: GridState,
                                   get_pos) -> bool:
        """
        檢測兩類非法情況：
        1) 兩個不同頂點坐標相同；
        2) 非端點頂點落在線段內部。
        """
        n = self._get_num_nodes(graph, state)

        # 1) 頂點重合
        seen = {}
        for v in range(n):
            p = get_pos(v)
            key = (p.x, p.y)
            if key in seen and seen[key] != v:
                return True
            seen[key] = v

        # 2) 非端點頂點落在某條邊內部
        for edge_idx in range(graph.num_edges):
            u, v = graph.get_edge_endpoints(edge_idx)
            a = get_pos(u)
            b = get_pos(v)

            for w in range(n):
                if w == u or w == v:
                    continue
                c = get_pos(w)
                if self._point_on_segment_strict(a, b, c):
                    return True

        return False

    def _compute_edge_crossings(self, graph: GraphData, get_pos,
                                spatial_hash: SpatialHash):
        """
        计算每条边的交叉次数（按 contest 规则），返回：
            edge_crossings: list[int], 长度 = #edges
            total: 总交叉数（每对边交叉计 1）
        """
        edge_crossings = [0] * graph.num_edges
        total = 0

        for i in range(graph.num_edges):
            src_i, tgt_i = graph.get_edge_endpoints(i)
            p1 = get_pos(src_i)
            p2 = get_pos(tgt_i)

            candidates = spatial_hash.query_edge_region(p1, p2)

            for j in candidates:
                if j <= i:
                    continue

                src_j, tgt_j = graph.get_edge_endpoints(j)

                # 端点相同不算交叉
                if (src_i == src_j or src_i == tgt_j or
                    tgt_i == src_j or tgt_i == tgt_j):
                    continue

                q1 = get_pos(src_j)
                q2 = get_pos(tgt_j)

                if self._segments_properly_intersect(p1, p2, q1, q2):
                    edge_crossings[i] += 1
                    edge_crossings[j] += 1
                    total += 1

        return edge_crossings, total

    # ----------------------------------------------------------------------
    # 完整 cost 计算：cost = w_cross * max_e crossings(e)
    # 非法时返回 +inf
    # ----------------------------------------------------------------------
    def calculate(self, graph: GraphData, state: GridState) -> float:
        """
        Calculate total cost from scratch.

        cost = w_cross * K,
        K = max_e (#crossings of edge e)
        """
        self._ensure_spatial_hash(graph, state)

        def get_pos(v: int) -> Point:
            return state.get_position(v)

        # 非法布局直接返回 +inf
        if self._has_illegal_configuration(graph, state, get_pos):
            return math.inf

        edge_crossings, _ = self._compute_edge_crossings(
            graph, get_pos, self._spatial_hash
        )
        k_max = max(edge_crossings) if edge_crossings else 0

        return float(k_max) * self.w_cross

    # ----------------------------------------------------------------------
    # 带单点移动的 cost 计算（用于 delta）
    # 这里不修改 state，只通过 get_pos 覆盖该点的位置。
    # ----------------------------------------------------------------------
    def _calculate_with_move(self, graph: GraphData, state: GridState,
                             node_id: int, new_pos: Point) -> float:
        """计算将 node_id 移动到 new_pos 后的完整 cost。"""

        def get_pos(v: int) -> Point:
            return new_pos if v == node_id else state.get_position(v)

        # 为“移动后”的布局新建局部 spatial hash
        spatial_hash = SpatialHash(cell_size=self.cell_size)
        for edge_idx in range(graph.num_edges):
            src, tgt = graph.get_edge_endpoints(edge_idx)
            p1 = get_pos(src)
            p2 = get_pos(tgt)
            spatial_hash.insert_edge(edge_idx, p1, p2)

        # 非法布局 -> +inf
        if self._has_illegal_configuration(graph, state, get_pos):
            return math.inf

        edge_crossings, _ = self._compute_edge_crossings(
            graph, get_pos, spatial_hash
        )
        k_max = max(edge_crossings) if edge_crossings else 0

        return float(k_max) * self.w_cross

    # ----------------------------------------------------------------------
    # 精確 delta：cost(after_move) - cost(before_move)
    # 非法 new_pos -> 返回 +inf（匹配 CUDA compute_delta_e 邏輯）
    # ----------------------------------------------------------------------
    def calculate_delta(self, graph: GraphData, state: GridState,
                        node_id: int, new_pos: Point) -> float:
        """
        Calculate exact delta in cost for moving node_id to new_pos.
        
        匹配 CUDA compute_delta_e 的完整邏輯（優化版本）：
        1. 快速檢查：節點重合（O(n)，提前失敗）
        2. 慢速檢查：邊穿過節點（O(d*n + k)，使用 spatial hash 優化）
        3. 無違規：計算真實的 K-value delta

        優化亮點：
        - 檢查順序優化（先快後慢）
        - 使用 spatial hash 減少 Scenario 2 的檢查範圍
        - 一旦發現違規立即返回（提前終止）

        返回:
            cost(after_move) - cost(before_move)
            若移動後布局非法，則返回 +inf
        """
        # CRITICAL: 優化的違規檢查順序
        
        # 1. 快速檢查：節點重合（O(n)，最快失敗）
        def get_pos_after_move(v: int) -> Point:
            return new_pos if v == node_id else state.get_position(v)
        
        dup_violations = self._count_duplicate_coords(graph, state, get_pos_after_move)
        
        if dup_violations > 0:
            # 有重複坐標 → 立即拒絕
            return math.inf
        
        # 2. 較慢檢查：邊穿過節點（O(d*n + k)，使用 spatial hash 優化）
        edge_violations = self._count_edge_through_node_violations(
            graph, state, node_id, new_pos
        )
        
        if edge_violations > 0:
            # 有違規 → 立即拒絕
            return math.inf
        
        # 3. 無違規 → 計算真實的 cost delta
        old_cost = self.calculate(graph, state)
        new_cost = self._calculate_with_move(graph, state, node_id, new_pos)
        
        # 雙重檢查：如果 cost 計算返回 inf（理論上不應發生）
        if not math.isfinite(new_cost):
            return math.inf
        
        if not math.isfinite(old_cost):
            return -math.inf
        
        return new_cost - old_cost

    # ----------------------------------------------------------------------
    # 统计函数：返回 (K, total_crossings)
    # ----------------------------------------------------------------------
    def get_crossing_stats(self, graph: GraphData, state: GridState) -> Tuple[int, int]:
        """
        Get crossing statistics under contest rules.

        Returns:
            (K, total) where
                K     = max crossings on any edge,
                total = total number of crossings (pairwise).
        """
        self._ensure_spatial_hash(graph, state)

        def get_pos(v: int) -> Point:
            return state.get_position(v)

        edge_crossings, total_crossings = self._compute_edge_crossings(
            graph, get_pos, self._spatial_hash
        )
        k = max(edge_crossings) if edge_crossings else 0

        return (k, total_crossings)

    def get_length_energy(self, graph: GraphData, state: GridState) -> float:
        """
        For compatibility with old API. Length is no longer part of the objective,
        so this always returns 0.
        """
        return 0.0
