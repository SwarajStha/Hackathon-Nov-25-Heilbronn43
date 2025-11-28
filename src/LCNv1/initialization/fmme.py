"""
FMME 初始化策略

使用 NetworkX Spring Layout 作为力导向布局。
适合完全随机或聚集的初始布局。
"""

import networkx as nx
import numpy as np
from typing import Dict

from .base import InitializationStrategy, OptimizerConfig


class FMMEInitializer(InitializationStrategy):
    """
    Force-Directed Layout Initializer
    
    特点：
    - 高温策略 (T=100)
    - 适用于随机初始布局
    """
    
    def __init__(self, spring_iterations: int = 50):
        self.spring_iterations = spring_iterations
    
    def generate_positions(
        self,
        num_nodes: int,
        edges: list,
        width: int,
        height: int,
        existing_positions: Dict = None
    ) -> Dict[int, tuple]:
        """使用 Spring Layout 生成布局"""
        if num_nodes == 0:
            return {}
        
        if num_nodes == 1:
            return {0: (width // 2, height // 2)}
        
        # 构建 NetworkX 图
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        G.add_edges_from(edges)
        
        # Spring Layout
        pos = nx.spring_layout(G, iterations=self.spring_iterations)
        
        # 归一化并缩放
        px = np.array([pos[i][0] for i in range(num_nodes)])
        py = np.array([pos[i][1] for i in range(num_nodes)])
        
        px_min, px_max = np.min(px), np.max(px)
        py_min, py_max = np.min(py), np.max(py)
        
        if px_max > px_min:
            px = (px - px_min) / (px_max - px_min)
        else:
            px = np.full(num_nodes, 0.5)
        
        if py_max > py_min:
            py = (py - py_min) / (py_max - py_min)
        else:
            py = np.full(num_nodes, 0.5)
        
        # 缩放到画布（5% padding）
        padding = 0.05
        px = px * width * (1 - 2 * padding) + width * padding
        py = py * height * (1 - 2 * padding) + height * padding
        
        return {i: (int(px[i]), int(py[i])) for i in range(num_nodes)}
    
    def get_optimizer_config(self) -> OptimizerConfig:
        """FMME 推荐高温配置"""
        return OptimizerConfig(
            initial_temp=100.0,
            cooling_rate=0.995,
            reheat_threshold=500,
            iterations=1000
        )
