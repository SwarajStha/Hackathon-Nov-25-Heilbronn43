"""
K-Plane优化成本函数

专门针对K-Plane Drawing问题的成本函数。
目标：最小化K（任一边的最大交叉数）
"""
from typing import Tuple
import math

from .geometry import Point, GeometryCore
from .graph import GraphData, GridState
from .spatial_index import SpatialHash
from .cost import ICostFunction


class KPlaneCost(ICostFunction):
    """
    K-Plane优化成本函数
    
    目标优先级：
    1. 最小化 K（最大交叉数） - 主要目标
    2. 最小化总交叉数 - 次要目标
    3. 最小化边长度 - 辅助目标
    
    能量函数：
    E = W_k * K^3 + W_cross * Σ(k^2) + W_len * Σ(len^2)
    
    其中：
    - K = max(k_i) 所有边中最大的交叉数
    - k_i = 边i的交叉数
    - W_k >> W_cross >> W_len （通过权重确保优先级）
    """
    
    def __init__(self, w_k: float = 10000.0, w_cross: float = 100.0, 
                 w_len: float = 1.0, cell_size: int = 50):
        """
        初始化K-Plane成本函数
        
        Args:
            w_k: K值权重（非常大，确保优先优化K）
            w_cross: 总交叉数权重
            w_len: 边长度权重
            cell_size: 空间哈希单元格大小
        """
        self.w_k = w_k
        self.w_cross = w_cross
        self.w_len = w_len
        self.cell_size = cell_size
        
        # 空间哈希用于快速相交查询
        self._spatial_hash = None
        self._graph = None
        self._state = None
    
    def _ensure_spatial_hash(self, graph: GraphData, state: GridState):
        """构建或重建空间哈希"""
        self._spatial_hash = SpatialHash(cell_size=self.cell_size)
        self._graph = graph
        self._state = state
        
        # 插入所有边
        for edge_idx in range(graph.num_edges):
            src, tgt = graph.get_edge_endpoints(edge_idx)
            p1 = state.get_position(src)
            p2 = state.get_position(tgt)
            self._spatial_hash.insert_edge(edge_idx, p1, p2)
    
    def calculate(self, graph: GraphData, state: GridState) -> float:
        """
        完整计算能量
        
        🆕 添加违规检测：如果状态违反几何约束，返回极大惩罚
        """
        # 🆕 首先检测违规
        from .violation_repair import ViolationRepair
        checker = ViolationRepair(graph, state, max_attempts=0)
        violations = checker.detect_violations()
        total_violations = (len(violations['duplicate_coords']) +
                          len(violations['nodes_on_edges']) +
                          len(violations['overlapping_edges']))
        
        if total_violations > 0:
            # 违规状态：返回极大惩罚（每个违规1亿）
            return 1e8 * total_violations
        
        # 确保空间哈希是最新的
        self._ensure_spatial_hash(graph, state)
        
        # 计算每条边的交叉数
        edge_crossings = self._calculate_edge_crossings(graph, state)
        
        # 计算K值（最大交叉数）
        k_value = max(edge_crossings) if edge_crossings else 0
        
        # K值能量（立方惩罚，强烈优先减小K）
        k_energy = self.w_k * (k_value ** 3)
        
        # 总交叉数能量（平方惩罚）
        crossing_energy = self.w_cross * sum(k ** 2 for k in edge_crossings)
        
        # 边长度能量
        length_energy = self._calculate_length_energy(graph, state)
        
        return k_energy + crossing_energy + length_energy
    
    def _calculate_edge_crossings(self, graph: GraphData, state: GridState) -> list:
        """计算每条边的交叉数"""
        edge_crossings = [0] * graph.num_edges
        
        # 计数每条边的交叉
        for i in range(graph.num_edges):
            src_i, tgt_i = graph.get_edge_endpoints(i)
            p1 = state.get_position(src_i)
            p2 = state.get_position(tgt_i)
            
            # 使用空间哈希查询附近的边
            candidates = self._spatial_hash.query_edge_region(p1, p2)
            
            for j in candidates:
                if j > i:  # 只计数每对一次
                    src_j, tgt_j = graph.get_edge_endpoints(j)
                    q1 = state.get_position(src_j)
                    q2 = state.get_position(tgt_j)
                    
                    if GeometryCore.segments_intersect(p1, p2, q1, q2):
                        edge_crossings[i] += 1
                        edge_crossings[j] += 1
        
        return edge_crossings
    
    def _calculate_length_energy(self, graph: GraphData, state: GridState) -> float:
        """计算边长度能量"""
        total = 0.0
        
        for edge_idx in range(graph.num_edges):
            src, tgt = graph.get_edge_endpoints(edge_idx)
            p1 = state.get_position(src)
            p2 = state.get_position(tgt)
            
            # 长度平方（避免sqrt）
            dx = p2.x - p1.x
            dy = p2.y - p1.y
            len_sq = dx * dx + dy * dy
            
            total += len_sq
        
        return self.w_len * total
    
    def calculate_delta(self, graph: GraphData, state: GridState, node_id: int, new_pos: Point) -> float:
        """
        计算移动节点后的能量变化
        
        简化版本：直接计算变化前后的能量差
        虽然不是最优化的，但保证正确性
        
        🆕 使用完整的calculate()来确保违规检测
        """
        # 计算移动前的能量（包括违规检测）
        old_energy = self.calculate(graph, state)
        
        # 创建临时状态并移动
        temp_state = state.clone()
        temp_state.move_node(node_id, new_pos)
        
        # 计算移动后的能量（包括违规检测）
        new_energy = self.calculate(graph, temp_state)
        
        # 返回能量差
        return new_energy - old_energy
    
    def get_crossing_stats(self, graph: GraphData, state: GridState) -> tuple:
        """
        获取交叉统计信息
        
        Returns:
            (k, total_crossings) - K值和总交叉数
        """
        # 确保空间哈希是最新的
        self._ensure_spatial_hash(graph, state)
        
        # 计算每条边的交叉数
        edge_crossings = self._calculate_edge_crossings(graph, state)
        
        # K值（最大交叉数）
        k = max(edge_crossings) if edge_crossings else 0
        
        # 总交叉数（每个交叉被计数两次，所以除以2）
        total_crossings = sum(edge_crossings) // 2
        
        return (k, total_crossings)
