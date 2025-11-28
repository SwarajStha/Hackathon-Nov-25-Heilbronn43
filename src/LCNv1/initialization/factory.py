"""
初始化策略工厂
"""

from .base import InitializationStrategy
from .fmme import FMMEInitializer
from .planarization import PlanarizationInitializer


class InitializationFactory:
    """初始化策略工厂"""
    
    _strategies = {
        'fmme': FMMEInitializer,
        'spring': FMMEInitializer,  # 别名
        'planarization': PlanarizationInitializer,
        'planar': PlanarizationInitializer  # 别名
    }
    
    @classmethod
    def create(cls, name: str, **kwargs) -> InitializationStrategy:
        """创建初始化策略"""
        name = name.lower()
        if name not in cls._strategies:
            raise ValueError(f"Unknown initialization strategy: {name}")
        
        return cls._strategies[name](**kwargs)
    
    @classmethod
    def register(cls, name: str, strategy_class: type):
        """注册新策略"""
        cls._strategies[name.lower()] = strategy_class
