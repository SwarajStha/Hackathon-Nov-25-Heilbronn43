"""
K-Plane Solver - Refactored with Strategy Pattern

重构亮点：
1. ✅ 依赖注入 (Dependency Injection) - 策略通过构造函数传入
2. ✅ 开闭原则 (OCP) - 新增策略无需修改此文件
3. ✅ 单一职责 (SRP) - Solver 仅负责优化循环，初始化委托给策略
4. ✅ 策略模式 (Strategy Pattern) - 可热插拔的初始化算法

使用示例：
>>> from strategies import FMMEStrategy, PlanarizationStrategy
>>> 
>>> # 使用 FMME 策略
>>> solver = KPlaneSolver(nodes, edges, 1000, 1000, strategy=FMMEStrategy())
>>> best_x, best_y, energy = solver.optimize()
>>> 
>>> # 切换到 Planarization 策略
>>> solver.strategy = PlanarizationStrategy()
>>> solver._reinitialize_with_strategy()
>>> best_x, best_y, energy = solver.optimize()
"""

import numpy as np
import math
import random
from typing import List, Dict, Tuple, Optional

# 导入策略模式接口
try:
    from strategies import InitializationStrategy, OptimizerConfig, FMMEStrategy
except ImportError:
    from src.strategies import InitializationStrategy, OptimizerConfig, FMMEStrategy

# 导入评分工具
try:
    from scorer import count_crossings, count_intersections_for_edges
except ImportError:
    from src.scorer import count_crossings, count_intersections_for_edges


class KPlaneSolver:
    """
    K-Planar Graph Solver (重构版)
    
    使用模拟退火 (Simulated Annealing) 最小化图的交叉数。
    
    架构改进：
    - 初始化逻辑分离到 InitializationStrategy
    - SA 参数由策略提供（高温/低温）
    - 支持策略动态切换
    
    原始实现：SimulatedAnnealingSolver (硬编码 NetworkX)
    重构后：KPlaneSolver (策略模式)
    """
    
    def __init__(
        self,
        nodes: List[Dict],
        edges: List[Dict],
        width: float,
        height: float,
        strategy: Optional[InitializationStrategy] = None
    ):
        """
        初始化 K-Plane Solver
        
        Args:
            nodes: 节点列表 [{'id': 0, 'x': ..., 'y': ...}, ...]
            edges: 边列表 [{'source': 0, 'target': 1}, ...]
            width: 画布宽度
            height: 画布高度
            strategy: 初始化策略（默认 FMMEStrategy）
        
        设计要点：
        - strategy 是可选的，默认使用 FMME（保持向后兼容）
        - 策略在构造函数中调用，生成初始布局
        - 策略推荐的优化器配置被存储为默认值
        """
        self.width = width
        self.height = height
        self.edges_data = edges
        
        # 构建边的数组表示
        self.edges_source = np.array([e['source'] for e in edges], dtype=np.int32)
        self.edges_target = np.array([e['target'] for e in edges], dtype=np.int32)
        self.num_nodes = len(nodes)
        self.num_edges = len(edges)
        
        # 构建邻接表：node_id -> [edge_indices]
        self.node_to_edges = [[] for _ in range(self.num_nodes)]
        for i, e in enumerate(edges):
            u, v = e['source'], e['target']
            if u < self.num_nodes:
                self.node_to_edges[u].append(i)
            if v < self.num_nodes:
                self.node_to_edges[v].append(i)
        
        # 策略模式：依赖注入
        if strategy is None:
            # 默认策略：FMME (保持向后兼容)
            strategy = FMMEStrategy()
        
        self.strategy = strategy
        
        # 调用策略生成初始布局
        layout = self.strategy.generate_layout(nodes, edges, width, height)
        self.nodes_x = layout['nodes_x'].copy()
        self.nodes_y = layout['nodes_y'].copy()
        
        # 获取策略推荐的优化器配置
        self.optimizer_config = self.strategy.get_optimizer_config()
        
        # 预计算初始交叉数
        self.edge_crossings, _, _ = count_crossings(
            self.nodes_x,
            self.nodes_y,
            self.edges_source,
            self.edges_target
        )
        
        # 存储原始节点（用于策略切换时重新初始化）
        self._original_nodes = nodes.copy()
    
    def _reinitialize_with_strategy(self):
        """
        使用当前策略重新初始化布局
        
        用于策略动态切换：
        >>> solver.strategy = PlanarizationStrategy()
        >>> solver._reinitialize_with_strategy()
        """
        # 重新生成布局
        layout = self.strategy.generate_layout(
            self._original_nodes,
            self.edges_data,
            self.width,
            self.height
        )
        self.nodes_x = layout['nodes_x'].copy()
        self.nodes_y = layout['nodes_y'].copy()
        
        # 更新优化器配置
        self.optimizer_config = self.strategy.get_optimizer_config()
        
        # 重新计算交叉数
        self.edge_crossings, _, _ = count_crossings(
            self.nodes_x,
            self.nodes_y,
            self.edges_source,
            self.edges_target
        )
    
    def current_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取当前节点坐标"""
        return self.nodes_x.copy(), self.nodes_y.copy()
    
    def energy(
        self,
        nodes_x: Optional[np.ndarray] = None,
        nodes_y: Optional[np.ndarray] = None
    ) -> float:
        """
        计算能量函数
        
        能量组成：
        1. 交叉能量：K * k_weight + total_crossings
        2. 斥力能量：节点间距离小于 radius 的惩罚
        3. 弹力能量：边长度的惩罚
        
        Args:
            nodes_x: 节点 x 坐标（None = 使用当前状态）
            nodes_y: 节点 y 坐标（None = 使用当前状态）
        
        Returns:
            总能量值
        """
        if nodes_x is None:
            nodes_x = self.nodes_x
        if nodes_y is None:
            nodes_y = self.nodes_y
        
        # 1. 交叉能量
        if nodes_x is not self.nodes_x or nodes_y is not self.nodes_y:
            _, k, total = count_crossings(
                nodes_x, nodes_y,
                self.edges_source,
                self.edges_target
            )
        else:
            k = np.max(self.edge_crossings) if self.num_edges > 0 else 0
            total = np.sum(self.edge_crossings) // 2
        
        # 动态 K 权重：确保交叉优先
        k_weight = max(10000, self.width * 0.1)
        crossing_energy = k * k_weight + total
        
        # 2. 斥力能量（半径 = 15% 画布宽度）
        radius = self.width * 0.15
        coords = np.stack([nodes_x, nodes_y], axis=1)
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        
        repulsion_mask = dists < radius
        repulsion_vals = radius - dists
        repulsion_energy = np.sum(repulsion_vals[repulsion_mask]) / 2
        
        # 3. 弹力能量（边长度）
        u_x = nodes_x[self.edges_source]
        u_y = nodes_y[self.edges_source]
        v_x = nodes_x[self.edges_target]
        v_y = nodes_y[self.edges_target]
        
        edge_lengths = np.sqrt((u_x - v_x)**2 + (u_y - v_y)**2)
        spring_energy = np.sum(edge_lengths) * 0.05
        
        return crossing_energy + repulsion_energy + spring_energy
    
    def optimize(
        self,
        iterations: Optional[int] = None,
        temp: Optional[float] = None,
        cooling_rate: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        运行模拟退火优化
        
        Args:
            iterations: 迭代次数（None = 使用策略推荐值）
            temp: 初始温度（None = 使用策略推荐值）
            cooling_rate: 冷却率（None = 使用策略推荐值）
        
        Returns:
            (best_x, best_y, best_energy)
        
        设计要点：
        - 参数优先级：显式参数 > 策略配置 > 硬编码默认值
        - 策略推荐的参数作为默认值
        """
        # 使用策略推荐的配置作为默认值
        if iterations is None:
            iterations = self.optimizer_config.iterations
        if temp is None:
            temp = self.optimizer_config.initial_temp
        if cooling_rate is None:
            cooling_rate = self.optimizer_config.cooling_rate
        
        return self.solve(iterations, temp, cooling_rate)
    
    def solve(
        self,
        iterations: int = 1000,
        temp: float = 10.0,
        cooling_rate: float = 0.995
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        模拟退火核心算法（保持与原实现兼容）
        
        这是原始 SimulatedAnnealingSolver.solve() 的重构版本。
        主要改进：
        - 初始化逻辑已移除（委托给策略）
        - SA 参数来自策略配置
        """
        # 常量
        radius = self.width * 0.15
        k_weight = max(10000, self.width * 0.1)
        
        # 初始状态
        current_x = self.nodes_x.copy()
        current_y = self.nodes_y.copy()
        
        # 初始能量组件
        current_k = np.max(self.edge_crossings) if self.num_edges > 0 else 0
        current_total_crossings = np.sum(self.edge_crossings) // 2
        current_crossing_energy = current_k * k_weight + current_total_crossings
        
        # 斥力
        coords = np.stack([current_x, current_y], axis=1)
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        rep_mask = dists < radius
        current_repulsion_energy = np.sum((radius - dists)[rep_mask]) / 2
        
        # 弹力
        u_x = current_x[self.edges_source]
        u_y = current_y[self.edges_source]
        v_x = current_x[self.edges_target]
        v_y = current_y[self.edges_target]
        edge_lengths = np.sqrt((u_x - v_x)**2 + (u_y - v_y)**2)
        current_spring_energy = np.sum(edge_lengths) * 0.05
        
        current_total_energy = (
            current_crossing_energy +
            current_repulsion_energy +
            current_spring_energy
        )
        
        best_x = current_x.copy()
        best_y = current_y.copy()
        best_energy = current_total_energy
        
        steps_since_improvement = 0
        
        # 主循环
        for i in range(iterations):
            # 随机选择节点
            node_idx = np.random.randint(0, self.num_nodes)
            
            old_x = self.nodes_x[node_idx]
            old_y = self.nodes_y[node_idx]
            
            # 生成新位置
            delta = temp
            new_x = old_x + np.random.uniform(-delta, delta)
            new_y = old_y + np.random.uniform(-delta, delta)
            
            # 边界约束
            new_x = max(0, min(self.width, new_x))
            new_y = max(0, min(self.height, new_y))
            
            # --- 增量更新 ---
            
            # 1. 斥力 Delta (O(N))
            d_old = np.sqrt((self.nodes_x - old_x)**2 + (self.nodes_y - old_y)**2)
            d_old[node_idx] = np.inf
            rep_old = np.sum(radius - d_old[d_old < radius])
            
            d_new = np.sqrt((self.nodes_x - new_x)**2 + (self.nodes_y - new_y)**2)
            d_new[node_idx] = np.inf
            rep_new = np.sum(radius - d_new[d_new < radius])
            
            delta_repulsion = rep_new - rep_old
            
            # 2. 弹力 Delta (O(d))
            incident_edge_indices = self.node_to_edges[node_idx]
            delta_spring = 0
            
            if incident_edge_indices:
                incident_indices = np.array(incident_edge_indices, dtype=np.int32)
                e_src = self.edges_source[incident_indices]
                e_tgt = self.edges_target[incident_indices]
                
                is_source = (e_src == node_idx)
                neighbor_indices = np.where(is_source, e_tgt, e_src)
                
                nbr_x = self.nodes_x[neighbor_indices]
                nbr_y = self.nodes_y[neighbor_indices]
                
                l_old = np.sqrt((old_x - nbr_x)**2 + (old_y - nbr_y)**2)
                l_new = np.sqrt((new_x - nbr_x)**2 + (new_y - nbr_y)**2)
                
                delta_spring = np.sum(l_new - l_old) * 0.05
            
            # 3. 交叉 Delta (O(d*E))
            if not incident_edge_indices:
                delta_crossing_energy = 0
                temp_crossings = self.edge_crossings
            else:
                # 计算交叉变化（简化版）
                # 临时移动节点
                self.nodes_x[node_idx] = new_x
                self.nodes_y[node_idx] = new_y
                
                # 重新计算受影响边的交叉
                new_crossings, new_k, new_total = count_crossings(
                    self.nodes_x,
                    self.nodes_y,
                    self.edges_source,
                    self.edges_target
                )
                
                # 恢复
                self.nodes_x[node_idx] = old_x
                self.nodes_y[node_idx] = old_y
                
                temp_crossings = new_crossings
                new_crossing_energy = new_k * k_weight + new_total
                delta_crossing_energy = new_crossing_energy - current_crossing_energy
            
            # 总 Delta
            total_delta = delta_crossing_energy + delta_repulsion + delta_spring
            new_energy_val = current_total_energy + total_delta
            
            # Metropolis 准则
            diff = current_total_energy - new_energy_val  # 正数 = 改进
            prob = 1.0
            if diff < 0:
                try:
                    prob = math.exp(diff / temp)
                except OverflowError:
                    prob = 0.0
            
            # 接受/拒绝
            if diff > 0 or np.random.random() < prob:
                # 接受
                current_total_energy = new_energy_val
                current_crossing_energy += delta_crossing_energy
                current_repulsion_energy += delta_repulsion
                current_spring_energy += delta_spring
                
                self.nodes_x[node_idx] = new_x
                self.nodes_y[node_idx] = new_y
                self.edge_crossings = temp_crossings
                
                steps_since_improvement = 0
                
                if new_energy_val < best_energy:
                    best_energy = new_energy_val
                    best_x = self.nodes_x.copy()
                    best_y = self.nodes_y.copy()
            else:
                # 拒绝
                steps_since_improvement += 1
            
            # 重新加热
            reheat_threshold = self.optimizer_config.reheat_threshold
            if steps_since_improvement > reheat_threshold:
                temp = max(temp * 2.0, self.width * 0.1)
                steps_since_improvement = 0
            
            # 降温
            temp *= cooling_rate
        
        # 恢复最佳状态
        self.nodes_x = best_x
        self.nodes_y = best_y
        
        return best_x, best_y, best_energy


# ============================================================
# Backward Compatibility Alias
# ============================================================

# 向后兼容：保持原类名
SimulatedAnnealingSolver = KPlaneSolver


# ============================================================
# Module Exports
# ============================================================

__all__ = ['KPlaneSolver', 'SimulatedAnnealingSolver']
