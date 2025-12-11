"""
LCNv1 求解策略模塊
包含所有優化策略的實現
"""

from enum import Enum
from .base import ISolverStrategy, SolverFactory


class StrategyType(str, Enum):
    """求解策略類型枚舉"""
    LEGACY = 'legacy'
    NEW = 'new'
    NUMBA = 'numba'
    CUDA = 'cuda'
    ENHANCED = 'enhanced'  # 增強版（支持初始化策略）


# 自動註冊策略
from . import register

# 導出增強策略
from .enhanced import EnhancedSolverStrategy

__all__ = [
    'ISolverStrategy',
    'SolverFactory',
    'StrategyType',
    'EnhancedSolverStrategy',
]
