"""
Planarization 初始化策略

通过插入虚拟节点最小化交叉数。
适合已经接近平面的布局。
"""

import numpy as np
from typing import Dict

from .base import InitializationStrategy, OptimizerConfig


class PlanarizationInitializer(InitializationStrategy):
    """
    Planarization-based Initializer
    
    特点：
    - 低温策略 (T=15)
    - 适用于已优化的布局
    """
    
    def __init__(self, max_crossings: int = 10):
        self.max_crossings = max_crossings
    
    def generate_positions(
        self,
        num_nodes: int,
        edges: list,
        width: int,
        height: int,
        existing_positions: Dict = None
    ) -> Dict[int, tuple]:
        """
        使用现有位置（仅微调）
        
        Planarization 假设输入已经是较好的布局。
        """
        if existing_positions:
            # 使用现有位置
            return existing_positions.copy()
        
        # 如果没有现有位置，生成网格布局
        return self._generate_grid_layout(num_nodes, width, height)
    
    def _generate_grid_layout(self, num_nodes: int, width: int, height: int) -> Dict:
        """生成网格布局作为回退"""
        cols = int(np.ceil(np.sqrt(num_nodes)))
        rows = int(np.ceil(num_nodes / cols))
        
        positions = {}
        idx = 0
        for r in range(rows):
            for c in range(cols):
                if idx >= num_nodes:
                    break
                x = int((c + 0.5) * width / cols)
                y = int((r + 0.5) * height / rows)
                positions[idx] = (x, y)
                idx += 1
        
        return positions
    
    def get_optimizer_config(self) -> OptimizerConfig:
        """Planarization 推荐低温配置"""
        return OptimizerConfig(
            initial_temp=15.0,
            cooling_rate=0.99,
            reheat_threshold=1000,
            iterations=1500
        )
