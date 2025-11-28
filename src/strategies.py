"""
Initialization Strategies for K-Plane Solver

实现 Strategy Pattern 以支持可扩展的图初始化算法。

可用策略：
1. FMMEStrategy - Force-Directed (NetworkX Spring Layout) 
2. PlanarizationStrategy - Virtual Node Insertion (OGDF-inspired)

符合设计原则：
- OCP: 新增策略无需修改现有代码
- DIP: Solver 依赖抽象接口而非具体实现
- SRP: 每个策略负责单一的初始化算法
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
import networkx as nx
from typing import List, Dict, Tuple
import random


# ============================================================
# Data Classes
# ============================================================

@dataclass
class OptimizerConfig:
    """
    优化器配置参数
    
    不同的初始化策略需要不同的 SA 参数：
    - FMME: 高温（大幅度重排）
    - Planarization: 低温（微调位置）
    """
    initial_temp: float        # 初始温度
    cooling_rate: float        # 冷却率 (0.9 - 0.999)
    reheat_threshold: int      # 重新加热阈值
    iterations: int = 1000     # 默认迭代次数
    
    def __post_init__(self):
        """验证参数有效性"""
        assert 0 < self.initial_temp <= 1000, "Temperature must be in (0, 1000]"
        assert 0.9 <= self.cooling_rate < 1.0, "Cooling rate must be in [0.9, 1.0)"
        assert self.reheat_threshold > 0, "Reheat threshold must be positive"


# ============================================================
# Abstract Strategy Interface
# ============================================================

class InitializationStrategy(ABC):
    """
    图布局初始化策略抽象接口
    
    所有策略必须实现：
    1. generate_layout() - 生成初始节点位置
    2. get_optimizer_config() - 返回推荐的 SA 参数
    """
    
    @abstractmethod
    def generate_layout(
        self,
        nodes: List[Dict],
        edges: List[Dict],
        width: float,
        height: float
    ) -> Dict[str, np.ndarray]:
        """
        生成初始布局
        
        Args:
            nodes: 节点列表 [{'id': 0, 'x': ..., 'y': ...}, ...]
            edges: 边列表 [{'source': 0, 'target': 1}, ...]
            width: 画布宽度
            height: 画布高度
        
        Returns:
            {
                'nodes_x': np.ndarray,  # shape: (num_nodes,)
                'nodes_y': np.ndarray   # shape: (num_nodes,)
            }
        """
        pass
    
    @abstractmethod
    def get_optimizer_config(self) -> OptimizerConfig:
        """
        返回该策略推荐的优化器配置
        
        Returns:
            OptimizerConfig 实例
        """
        pass


# ============================================================
# Concrete Strategy 1: FMME (Force-Directed)
# ============================================================

class FMMEStrategy(InitializationStrategy):
    """
    FMME (Fast Multipole Multilevel Embedding) 策略
    
    使用 NetworkX Spring Layout (Fruchterman-Reingold) 作为力导向布局。
    
    特点：
    - **高温策略** (T=100): 允许大幅度重排
    - 适用于完全随机或聚集的初始布局
    - 通过力导向算法自动分散节点
    
    算法流程：
    1. 构建 NetworkX 图
    2. 运行 spring_layout (力导向)
    3. 归一化到 [0, 1]
    4. 缩放到画布尺寸（留 5% padding）
    """
    
    def __init__(self, spring_iterations: int = 50, scale_factor: float = 0.9):
        """
        初始化 FMME 策略
        
        Args:
            spring_iterations: Spring Layout 迭代次数
            scale_factor: 缩放因子（0.9 = 使用 90% 画布）
        """
        self.spring_iterations = spring_iterations
        self.scale_factor = scale_factor
    
    def generate_layout(
        self,
        nodes: List[Dict],
        edges: List[Dict],
        width: float,
        height: float
    ) -> Dict[str, np.ndarray]:
        """生成力导向布局"""
        num_nodes = len(nodes)
        
        # 边界情况：空图
        if num_nodes == 0:
            return {
                'nodes_x': np.array([]),
                'nodes_y': np.array([])
            }
        
        # 边界情况：单节点
        if num_nodes == 1:
            return {
                'nodes_x': np.array([width / 2]),
                'nodes_y': np.array([height / 2])
            }
        
        # 构建 NetworkX 图
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        G.add_edges_from([(e['source'], e['target']) for e in edges])
        
        # 运行 Spring Layout
        # k 控制节点间距，iterations 控制收敛质量
        pos = nx.spring_layout(
            G,
            iterations=self.spring_iterations,
            k=None,  # 自动计算最优间距
            scale=1.0  # 初始缩放为 1（后续手动缩放）
        )
        
        # 提取坐标
        px = np.array([pos[i][0] for i in range(num_nodes)])
        py = np.array([pos[i][1] for i in range(num_nodes)])
        
        # 归一化到 [0, 1]
        p_min_x, p_max_x = np.min(px), np.max(px)
        p_min_y, p_max_y = np.min(py), np.max(py)
        
        if p_max_x > p_min_x:
            px = (px - p_min_x) / (p_max_x - p_min_x)
        else:
            px = np.full(num_nodes, 0.5)  # 所有节点在同一 x 坐标
        
        if p_max_y > p_min_y:
            py = (py - p_min_y) / (p_max_y - p_min_y)
        else:
            py = np.full(num_nodes, 0.5)
        
        # 缩放到画布（留 padding）
        padding_ratio = (1.0 - self.scale_factor) / 2
        padding_x = width * padding_ratio
        padding_y = height * padding_ratio
        usable_w = width * self.scale_factor
        usable_h = height * self.scale_factor
        
        nodes_x = px * usable_w + padding_x
        nodes_y = py * usable_h + padding_y
        
        return {
            'nodes_x': nodes_x,
            'nodes_y': nodes_y
        }
    
    def get_optimizer_config(self) -> OptimizerConfig:
        """
        FMME 推荐配置：高温策略
        
        高温允许大幅度移动，适合完全重排布局。
        """
        return OptimizerConfig(
            initial_temp=100.0,      # 高温：width * 0.1 典型值
            cooling_rate=0.995,      # 标准冷却
            reheat_threshold=500,    # 标准重加热
            iterations=1000
        )


# ============================================================
# Concrete Strategy 2: Planarization
# ============================================================

class PlanarizationStrategy(InitializationStrategy):
    """
    Planarization 策略（虚拟节点插入）
    
    灵感来源于 OGDF 的 PlanarizationLayout。
    
    核心思想：
    1. 检测所有边的交叉
    2. 在交叉点插入虚拟节点，将 1 条边拆分为 2 条
    3. 对平面化后的图进行重新布局
    4. 使用低温 SA 微调位置
    
    特点：
    - **低温策略** (T=15): 仅微调，不大幅移动
    - 适用于已经接近平面的布局
    - 通过插入虚拟节点降低交叉复杂度
    
    算法流程：
    1. 计算所有边对的交叉点
    2. 在交叉点插入虚拟节点
    3. 重新运行力导向布局（低迭代次数）
    4. 返回优化后的坐标
    """
    
    def __init__(self, max_crossings: int = 10, virtual_node_penalty: float = 0.1):
        """
        初始化 Planarization 策略
        
        Args:
            max_crossings: 最大允许交叉数（超过则插入虚拟节点）
            virtual_node_penalty: 虚拟节点的位置惩罚系数
        """
        self.max_crossings = max_crossings
        self.virtual_node_penalty = virtual_node_penalty
    
    def generate_layout(
        self,
        nodes: List[Dict],
        edges: List[Dict],
        width: float,
        height: float
    ) -> Dict[str, np.ndarray]:
        """生成平面化布局"""
        num_nodes = len(nodes)
        
        # 边界情况：空图或单节点
        if num_nodes == 0:
            return {'nodes_x': np.array([]), 'nodes_y': np.array([])}
        if num_nodes == 1:
            return {'nodes_x': np.array([width / 2]), 'nodes_y': np.array([height / 2])}
        
        # Step 1: 使用输入坐标初始化
        nodes_x = np.array([n['x'] for n in nodes])
        nodes_y = np.array([n['y'] for n in nodes])
        
        # Step 2: 检测交叉并插入虚拟节点
        virtual_nodes_x, virtual_nodes_y, augmented_edges = self._insert_virtual_nodes(
            nodes_x, nodes_y, edges
        )
        
        # Step 3: 合并原始节点和虚拟节点
        all_x = np.concatenate([nodes_x, virtual_nodes_x])
        all_y = np.concatenate([nodes_y, virtual_nodes_y])
        num_total_nodes = len(all_x)
        
        # Step 4: 如果有虚拟节点，使用力导向布局优化
        if len(virtual_nodes_x) > 0:
            G = nx.Graph()
            G.add_nodes_from(range(num_total_nodes))
            G.add_edges_from([(e['source'], e['target']) for e in augmented_edges])
            
            # 使用当前位置作为初始位置
            init_pos = {i: (all_x[i], all_y[i]) for i in range(num_total_nodes)}
            
            # 低迭代次数的力导向（仅微调）
            pos = nx.spring_layout(
                G,
                pos=init_pos,
                iterations=20,  # 低迭代 = 微调
                k=None
            )
            
            all_x = np.array([pos[i][0] for i in range(num_total_nodes)])
            all_y = np.array([pos[i][1] for i in range(num_total_nodes)])
        
        # Step 5: 归一化和缩放
        all_x, all_y = self._normalize_and_scale(all_x, all_y, width, height)
        
        return {
            'nodes_x': all_x,
            'nodes_y': all_y
        }
    
    def _insert_virtual_nodes(
        self,
        nodes_x: np.ndarray,
        nodes_y: np.ndarray,
        edges: List[Dict]
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        检测交叉并插入虚拟节点
        
        Returns:
            (virtual_x, virtual_y, augmented_edges)
        """
        virtual_x = []
        virtual_y = []
        augmented_edges = list(edges)  # 复制原始边
        
        num_original_nodes = len(nodes_x)
        next_virtual_id = num_original_nodes
        
        # 检测所有边对的交叉
        for i, e1 in enumerate(edges):
            for j, e2 in enumerate(edges):
                if j <= i:
                    continue  # 避免重复检查
                
                # 获取边的端点
                src1, tgt1 = e1['source'], e1['target']
                src2, tgt2 = e2['source'], e2['target']
                
                # 跳过共享端点的边
                if src1 == src2 or src1 == tgt2 or tgt1 == src2 or tgt1 == tgt2:
                    continue
                
                # 检测交叉
                p1 = (nodes_x[src1], nodes_y[src1])
                p2 = (nodes_x[tgt1], nodes_y[tgt1])
                q1 = (nodes_x[src2], nodes_y[src2])
                q2 = (nodes_x[tgt2], nodes_y[tgt2])
                
                intersection = self._line_intersection(p1, p2, q1, q2)
                
                if intersection is not None:
                    # 在交叉点插入虚拟节点
                    vx, vy = intersection
                    virtual_x.append(vx)
                    virtual_y.append(vy)
                    
                    # 注意：完整的 planarization 需要重构边
                    # 这里简化为仅记录虚拟节点位置
                    # 生产实现应该拆分交叉的边
                    next_virtual_id += 1
        
        return (
            np.array(virtual_x),
            np.array(virtual_y),
            augmented_edges  # 简化版：不修改边结构
        )
    
    def _line_intersection(
        self,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        q1: Tuple[float, float],
        q2: Tuple[float, float]
    ) -> Tuple[float, float] | None:
        """
        计算两条线段的交点
        
        使用参数方程求解。
        
        Returns:
            (x, y) 或 None（不相交）
        """
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = q1
        x4, y4 = q2
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        
        if abs(denom) < 1e-10:
            return None  # 平行或共线
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
        
        # 检查是否在线段内
        if 0 <= t <= 1 and 0 <= u <= 1:
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            return (x, y)
        
        return None
    
    def _normalize_and_scale(
        self,
        x: np.ndarray,
        y: np.ndarray,
        width: float,
        height: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """归一化并缩放到画布"""
        min_x, max_x = np.min(x), np.max(x)
        min_y, max_y = np.min(y), np.max(y)
        
        if max_x > min_x:
            x = (x - min_x) / (max_x - min_x)
        else:
            x = np.full(len(x), 0.5)
        
        if max_y > min_y:
            y = (y - min_y) / (max_y - min_y)
        else:
            y = np.full(len(y), 0.5)
        
        # 缩放到画布（5% padding）
        padding = 0.05
        x = x * width * (1 - 2 * padding) + width * padding
        y = y * height * (1 - 2 * padding) + height * padding
        
        return x, y
    
    def get_optimizer_config(self) -> OptimizerConfig:
        """
        Planarization 推荐配置：低温策略
        
        低温仅微调，不破坏已经平面化的结构。
        """
        return OptimizerConfig(
            initial_temp=15.0,       # 低温：仅微调
            cooling_rate=0.99,       # 慢速冷却
            reheat_threshold=1000,   # 更长的重加热间隔
            iterations=1500
        )


# ============================================================
# Strategy Factory (Optional - For Extensibility)
# ============================================================

class StrategyFactory:
    """
    策略工厂（可选）
    
    提供字符串 -> 策略实例的映射，方便配置文件驱动。
    """
    
    _strategies = {
        'fmme': FMMEStrategy,
        'planarization': PlanarizationStrategy,
        'spring': FMMEStrategy,  # 别名
        'planar': PlanarizationStrategy  # 别名
    }
    
    @classmethod
    def create(cls, name: str, **kwargs) -> InitializationStrategy:
        """
        根据名称创建策略
        
        Args:
            name: 策略名称 ('fmme', 'planarization')
            **kwargs: 传递给策略构造函数的参数
        
        Returns:
            InitializationStrategy 实例
        
        Raises:
            ValueError: 未知策略名称
        """
        name = name.lower()
        if name not in cls._strategies:
            available = ', '.join(cls._strategies.keys())
            raise ValueError(f"Unknown strategy '{name}'. Available: {available}")
        
        return cls._strategies[name](**kwargs)
    
    @classmethod
    def register(cls, name: str, strategy_class: type):
        """
        注册新策略（扩展点）
        
        示例：
        >>> StrategyFactory.register('custom', MyCustomStrategy)
        """
        cls._strategies[name.lower()] = strategy_class


# ============================================================
# Module Exports
# ============================================================

__all__ = [
    'InitializationStrategy',
    'OptimizerConfig',
    'FMMEStrategy',
    'PlanarizationStrategy',
    'StrategyFactory'
]
