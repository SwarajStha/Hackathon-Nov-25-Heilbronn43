"""
LCNv1 增强版求解策略（整合初始化策略）

展示如何将 InitializationStrategy 整合到现有的 ISolverStrategy 中。
"""

import json
import random
import math
from typing import Dict, Any, Optional

from .base import ISolverStrategy, SolverFactory
from ..core.geometry import Point
from ..core.graph import GraphData, GridState
from ..core.cost import SoftMaxCost
from ..initialization import InitializationStrategy, FMMEInitializer


class EnhancedSolverStrategy(ISolverStrategy):
    """
    增强版 Solver（整合初始化策略）
    
    改进点：
    1. ✅ 依赖注入：接受 InitializationStrategy
    2. ✅ 动态配置：SA 参数由初始化策略推荐
    3. ✅ 策略切换：可以运行时更换初始化策略
    
    使用示例：
    >>> from LCNv1.initialization import FMMEInitializer, PlanarizationInitializer
    >>> 
    >>> # 使用 FMME 初始化
    >>> solver = EnhancedSolverStrategy(init_strategy=FMMEInitializer())
    >>> solver.load_from_json('input.json')
    >>> result = solver.solve(iterations=1000)  # 自动使用高温配置
    >>> 
    >>> # 切换到 Planarization
    >>> solver.set_initialization_strategy(PlanarizationInitializer())
    >>> result = solver.solve()  # 自动使用低温配置
    """
    
    def __init__(
        self,
        w_cross: float = 100.0,
        w_len: float = 1.0,
        power: int = 2,
        cell_size: int = 50,
        init_strategy: Optional[InitializationStrategy] = None
    ):
        """
        初始化增强版 Solver
        
        Args:
            w_cross: 交叉惩罚权重
            w_len: 长度惩罚权重
            power: 交叉惩罚指数
            cell_size: 空间哈希单元大小
            init_strategy: 初始化策略（默认 FMME）
        """
        self.graph = None
        self.state = None
        self.cost_func = SoftMaxCost(
            w_cross=w_cross,
            w_len=w_len,
            power=power,
            cell_size=cell_size
        )
        self.data = None
        
        # 策略注入
        self.init_strategy = init_strategy or FMMEInitializer()
        
        # 优化状态
        self.current_energy = None
        self.best_state = None
        self.best_energy = float('inf')
        self.iteration = 0
    
    def set_initialization_strategy(self, strategy: InitializationStrategy):
        """
        动态切换初始化策略
        
        切换后需要重新调用 load_from_json() 以应用新策略。
        """
        self.init_strategy = strategy
    
    def load_from_json(self, json_path: str):
        """
        从 JSON 加载图（使用初始化策略）
        
        关键改进：
        - 不再硬编码初始位置
        - 委托给 InitializationStrategy.generate_positions()
        """
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        # 构建图拓扑
        edges = [(e['source'], e['target']) for e in self.data['edges']]
        num_nodes = len(self.data['nodes'])
        
        self.graph = GraphData(num_nodes=num_nodes, edges=edges)
        
        # 提取现有位置（可能作为 hint）
        existing_positions = {}
        for node in self.data['nodes']:
            nid = node['id']
            existing_positions[nid] = (int(node['x']), int(node['y']))
        
        # 🆕 调用初始化策略生成布局
        positions_dict = self.init_strategy.generate_positions(
            num_nodes=num_nodes,
            edges=edges,
            width=self.data.get('width', 1000),
            height=self.data.get('height', 1000),
            existing_positions=existing_positions
        )
        
        # 转换为 GridState
        positions = {}
        for node_id, (x, y) in positions_dict.items():
            positions[node_id] = Point(int(x), int(y))
        
        self.state = GridState(
            positions,
            width=self.data.get('width', 1000),
            height=self.data.get('height', 1000)
        )
        
        # 计算初始能量
        self.current_energy = self.cost_func.calculate(self.graph, self.state)
        self.best_state = self.state.clone()
        self.best_energy = self.current_energy
    
    def solve(self, iterations: int = None, **kwargs) -> Dict[str, Any]:
        """
        运行模拟退火（使用策略推荐的配置）
        
        关键改进：
        - 参数优先级：显式参数 > 策略推荐 > 默认值
        - 策略推荐的配置自动应用
        """
        if self.graph is None or self.state is None:
            raise RuntimeError("Must load data first using load_from_json()")
        
        # 🆕 获取策略推荐的配置
        config = self.init_strategy.get_optimizer_config()
        
        # 参数优先级：显式 > 策略 > 默认
        iterations = iterations or kwargs.get('iterations', config.iterations)
        initial_temp = kwargs.get('initial_temp', config.initial_temp)
        cooling_rate = kwargs.get('cooling_rate', config.cooling_rate)
        reheat_threshold = kwargs.get('reheat_threshold', config.reheat_threshold)
        
        # 初始化
        current_temp = initial_temp
        accepted_count = 0
        rejected_count = 0
        steps_since_improvement = 0
        energy_history = []
        
        print(f"\n[Enhanced Solver] Starting optimization...")
        print(f"  Strategy: {self.init_strategy.__class__.__name__}")
        print(f"  Initial Temp: {initial_temp:.1f}")
        print(f"  Cooling Rate: {cooling_rate:.4f}")
        print(f"  Iterations: {iterations}")
        
        for i in range(iterations):
            self.iteration = i
            
            # 生成移动
            node_id = random.randint(0, self.graph.num_nodes - 1)
            old_pos = self.state.get_position(node_id)
            
            # 温度依赖的步长
            step_size = max(1, int(current_temp))
            new_x = old_pos.x + random.randint(-step_size, step_size)
            new_y = old_pos.y + random.randint(-step_size, step_size)
            
            # 边界约束
            new_x = max(0, min(self.state.width, new_x))
            new_y = max(0, min(self.state.height, new_y))
            new_pos = Point(new_x, new_y)
            
            # 计算 Delta
            delta = self.cost_func.calculate_delta(
                self.graph, self.state, node_id, new_pos
            )
            
            # Metropolis 准则
            if delta < 0 or random.random() < math.exp(-delta / max(current_temp, 0.1)):
                # 接受
                self.state.move_node(node_id, new_pos)
                self.current_energy += delta
                accepted_count += 1
                
                # 更新最佳
                if self.current_energy < self.best_energy:
                    self.best_energy = self.current_energy
                    self.best_state = self.state.clone()
                    steps_since_improvement = 0
                else:
                    steps_since_improvement += 1
            else:
                # 拒绝
                rejected_count += 1
                steps_since_improvement += 1
            
            # 降温
            current_temp *= cooling_rate
            
            # 重新加热
            if steps_since_improvement > reheat_threshold:
                current_temp = initial_temp * 0.5
                steps_since_improvement = 0
            
            # 记录能量
            if i % 10 == 0:
                energy_history.append(self.current_energy)
            
            # 进度报告
            if (i + 1) % 100 == 0:
                k, total = self.cost_func.get_crossing_stats(self.graph, self.state)
                print(f"  [{i+1:>4}/{iterations}] E={self.current_energy:>8.0f}, "
                      f"K={k:>2}, X={total:>4}, T={current_temp:>5.2f}, "
                      f"Accept={accepted_count/(i+1)*100:>4.1f}%")
        
        # 使用最佳状态
        self.state = self.best_state.clone()
        
        # 计算最终统计
        k, total = self.cost_func.get_crossing_stats(self.graph, self.state)
        
        print(f"\n[Done] Best Energy: {self.best_energy:.0f}, K={k}, Crossings={total}")
        
        # 转换为 JSON 格式
        nodes = []
        for node_id in range(self.graph.num_nodes):
            pos = self.state.get_position(node_id)
            nodes.append({
                'id': node_id,
                'x': pos.x,
                'y': pos.y
            })
        
        return {
            'nodes': nodes,
            'stats': {
                'iterations': iterations,
                'accepted': accepted_count,
                'rejected': rejected_count,
                'acceptance_rate': accepted_count / iterations,
                'final_temp': current_temp,
                'energy_history': energy_history,
                'method': 'enhanced_with_init_strategy',
                'init_strategy': self.init_strategy.__class__.__name__
            },
            'energy': self.best_energy,
            'k': k,
            'total_crossings': total
        }
    
    def get_current_stats(self) -> Dict[str, Any]:
        """获取当前统计"""
        if self.graph is None or self.state is None:
            return {'error': 'No graph loaded'}
        
        k, total = self.cost_func.get_crossing_stats(self.graph, self.state)
        
        return {
            'energy': self.current_energy,
            'best_energy': self.best_energy,
            'k': k,
            'total_crossings': total,
            'num_nodes': self.graph.num_nodes,
            'num_edges': self.graph.num_edges,
            'iteration': self.iteration,
            'init_strategy': self.init_strategy.__class__.__name__
        }
    
    def export_to_json(self, output_path: str):
        """导出当前解到 JSON"""
        nodes = []
        for node_id in range(self.graph.num_nodes):
            pos = self.state.get_position(node_id)
            nodes.append({
                'id': node_id,
                'x': pos.x,
                'y': pos.y
            })
        
        output_data = {
            'nodes': nodes,
            'edges': self.data['edges'],
            'width': self.data.get('width', 1000),
            'height': self.data.get('height', 1000)
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)


# 注册到工厂
SolverFactory.register_strategy('enhanced', EnhancedSolverStrategy)

__all__ = ['EnhancedSolverStrategy']
