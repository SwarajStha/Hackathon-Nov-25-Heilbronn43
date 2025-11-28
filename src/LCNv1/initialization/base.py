"""
初始化策略抽象接口

与 ISolverStrategy 配合使用：
- InitializationStrategy: 生成初始布局
- ISolverStrategy: 运行优化算法
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict
import numpy as np


@dataclass
class OptimizerConfig:
    """
    优化器推荐配置
    
    不同的初始化策略需要不同的 SA 参数：
    - FMME (随机布局) → 高温，大步长
    - Planarization (已优化) → 低温，微调
    """
    initial_temp: float
    cooling_rate: float
    reheat_threshold: int
    iterations: int = 1000


class InitializationStrategy(ABC):
    """初始化策略抽象接口"""
    
    @abstractmethod
    def generate_positions(
        self,
        num_nodes: int,
        edges: list,
        width: int,
        height: int,
        existing_positions: Dict = None
    ) -> Dict[int, tuple]:
        """
        生成初始节点位置
        
        Args:
            num_nodes: 节点数量
            edges: 边列表 [(src, tgt), ...]
            width: 画布宽度
            height: 画布高度
            existing_positions: 可选的现有位置 {node_id: (x, y)}
        
        Returns:
            {node_id: (x, y)}
        """
        pass
    
    @abstractmethod
    def get_optimizer_config(self) -> OptimizerConfig:
        """返回推荐的优化器配置"""
        pass
