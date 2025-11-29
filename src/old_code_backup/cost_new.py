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
    GD Contest 目标：

        cost(G, pos) = w_cross * K
        K = max_e (#crossings of edge e)

    规则：
    - 只考虑直线绘制。
    - 边的交叉按 pairwise 计算。
    - 有公共端点的两条边相交时不计为交叉（端点相同不算交叉）。
    - 若存在以下任一情况，视为非法布局，cost = +inf：
        1) 两个不同顶点坐标完全相同；
        2) 某顶点（非该边端点）严格落在该边上。

    说明：
    - w_len, power 参数保留以兼容旧接口，但不再参与 cost。
    """

    def __init__(self, w_cross: float = 100.0, w_len: float = 1.0,
                 power: int = 2, cell_size: int = 50):
        self.w_cross = w_cross
        self.w_len = w_len    # 保留但不用
        self.power = power    # 保留但不用
        self.cell_size = cell_size

        self._spatial_hash = None
        self._graph = None
        self._state = None

    # ------------------------------------------------------------------
    # 基础工具
    # ------------------------------------------------------------------
    def _ensure_spatial_hash(self, graph: GraphData, state: GridState):
        """为当前 state 重建 spatial hash。"""
        self._spatial_hash = SpatialHash(cell_size=self.cell_size)
        self._graph = graph
        self._state = state

        for edge_idx in range(graph.num_edges):
            src, tgt = graph.get_edge_endpoints(edge_idx)
            p1 = state.get_position(src)
            p2 = state.get_position(tgt)
            self._spatial_hash.insert_edge(edge_idx, p1, p2)

    def _get_num_nodes(self, graph: GraphData, state: GridState) -> int:
        """尽量从 graph/state 里拿到点数。"""
        if hasattr(graph, "num_nodes"):
            return graph.num_nodes
        if hasattr(state, "num_nodes"):
            return state.num_nodes
        if hasattr(state, "num_vertices"):
            return state.num_vertices
        raise AttributeError("Cannot determine number of nodes from graph/state.")

    def _orientation(self, a: Point, b: Point, c: Point) -> int:
        """(b - a) × (c - a) 的符号，用于方向判断。"""
        return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)

    def _segments_properly_intersect(self, p1: Point, p2: Point,
                                     q1: Point, q2: Point) -> bool:
        """
        真交叉：两线段在内部相交（不含端点接触、重合等退化情况）。
        """
        o1 = self._orientation(p1, p2, q1)
        o2 = self._orientation(p1, p2, q2)
        o3 = self._orientation(q1, q2, p1)
        o4 = self._orientation(q1, q2, p2)

        return (o1 * o2 < 0) and (o3 * o4 < 0)

    def _point_on_segment_strict(self, a: Point, b: Point, c: Point) -> bool:
        """
        判断点 c 是否在线段 ab 的“内部”（严格意义上的内部，不含端点）。
        用于顶点落在边上的非法性检查。
        """
        # 共线性
        if self._orientation(a, b, c) != 0:
            return False

        # 包围盒内
        if c.x < min(a.x, b.x) or c.x > max(a.x, b.x):
            return False
        if c.y < min(a.y, b.y) or c.y > max(a.y, b.y):
            return False

        # 排除端点
        if (c.x == a.x and c.y == a.y) or (c.x == b.x and c.y == b.y):
            return False

        return True

    def _has_illegal_configuration(self, graph: GraphData,
                                   state: GridState,
                                   get_pos) -> bool:
        """
        检查两类非法情况：
        1) 两个不同顶点坐标相同；
        2) 某顶点（非该边端点）严格落在某条边内部。
        """
        n = self._get_num_nodes(graph, state)

        # 1) 顶点重合
        seen = {}
        for v in range(n):
            p = get_pos(v)
            key = (p.x, p.y)
            if key in seen and seen[key] != v:
                return True
            seen[key] = v

        # 2) 顶点落在边内部
        for edge_idx in range(graph.num_edges):
            u, w = graph.get_edge_endpoints(edge_idx)
            a = get_pos(u)
            b = get_pos(w)

            for v in range(n):
                if v == u or v == w:
                    continue
                c = get_pos(v)
                if self._point_on_segment_strict(a, b, c):
                    return True

        return False

    def _compute_edge_crossings(self, graph: GraphData, get_pos,
                                spatial_hash: SpatialHash):
        """
        计算每条边的交叉次数（contest 规则）：
            - 共端点边对不计交叉
            - 只计真交叉（内部相交）
        返回：
            edge_crossings: list[int], len = num_edges
            total: 总 pairwise 交叉数
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

    # ------------------------------------------------------------------
    # 完整 cost 计算：cost = w_cross * max_e crossings(e)
    # 非法时返回 +inf
    # ------------------------------------------------------------------
    def calculate(self, graph: GraphData, state: GridState) -> float:
        """
        Calculate total cost from scratch.

        cost = w_cross * K,
        K = max_e (#crossings of edge e)
        """
        self._ensure_spatial_hash(graph, state)

        def get_pos(v: int) -> Point:
            return state.get_position(v)

        # 非法布局 -> +inf
        if self._has_illegal_configuration(graph, state, get_pos):
            return math.inf

        edge_crossings, _ = self._compute_edge_crossings(
            graph, get_pos, self._spatial_hash
        )
        k_max = max(edge_crossings) if edge_crossings else 0

        return float(k_max) * self.w_cross

    # ------------------------------------------------------------------
    # “移动后”完整 cost：不修改 state，只通过 get_pos 覆盖该点位置；
    # 使用新的 spatial hash（对应新位置）。
    # ------------------------------------------------------------------
    def _calculate_with_move(self, graph: GraphData, state: GridState,
                             node_id: int, new_pos: Point) -> float:
        """计算将 node_id 移动到 new_pos 后的完整 cost。"""

        def get_pos(v: int) -> Point:
            return new_pos if v == node_id else state.get_position(v)

        spatial_hash = SpatialHash(cell_size=self.cell_size)
        for edge_idx in range(graph.num_edges):
            src, tgt = graph.get_edge_endpoints(edge_idx)
            p1 = get_pos(src)
            p2 = get_pos(tgt)
            spatial_hash.insert_edge(edge_idx, p1, p2)

        # 非法 -> +inf
        if self._has_illegal_configuration(graph, state, get_pos):
            return math.inf

        edge_crossings, _ = self._compute_edge_crossings(
            graph, get_pos, spatial_hash
        )
        k_max = max(edge_crossings) if edge_crossings else 0

        return float(k_max) * self.w_cross

    # ------------------------------------------------------------------
    # 精确 delta：cost(after_move) - cost(before_move)
    # 移动后非法 -> +inf
    # ------------------------------------------------------------------
    def calculate_delta(self, graph: GraphData, state: GridState,
                        node_id: int, new_pos: Point) -> float:
        """
        Calculate exact delta in cost for moving node_id to new_pos.

        返回：
            cost(after_move) - cost(before_move)

        若移动后布局非法，则返回 +inf。
        """
        old_cost = self.calculate(graph, state)
        new_cost = self._calculate_with_move(graph, state, node_id, new_pos)

        if not math.isfinite(new_cost):
            return math.inf
        if not math.isfinite(old_cost):
            # 理论上不会出现：若当前布局已非法，可视为巨大改进
            return -math.inf

        return new_cost - old_cost

    # ------------------------------------------------------------------
    # 统计函数
    # ------------------------------------------------------------------
    def get_crossing_stats(self, graph: GraphData, state: GridState) -> Tuple[int, int]:
        """
        返回 (K, total)：
            K     = 单条边上的最大交叉数（max_e crossings(e)）
            total = 总交叉数（按边对计数）
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
        兼容旧接口：长度不再参与目标函数，这里恒为 0。
        """
        return 0.0