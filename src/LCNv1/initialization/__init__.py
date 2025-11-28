"""
LCNv1 图初始化策略模块

提供可插拔的初始布局生成算法。

设计目标：
- 分离"初始化"与"优化"职责
- 支持不同的布局算法（FMME, Planarization, Random）
- 每个策略推荐最佳的 SA 参数

整合到现有架构：
- ISolverStrategy 继续负责优化循环
- InitializationStrategy 负责生成初始位置
"""

from .base import InitializationStrategy, OptimizerConfig
from .fmme import FMMEInitializer
from .planarization import PlanarizationInitializer
from .factory import InitializationFactory

__all__ = [
    'InitializationStrategy',
    'OptimizerConfig',
    'FMMEInitializer',
    'PlanarizationInitializer',
    'InitializationFactory'
]
