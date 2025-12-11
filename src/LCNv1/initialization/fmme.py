"""
FMME 初始化策略

使用 NetworkX Spring Layout 作为力导向布局。
适合完全随机或聚集的初始布局。

支持动态优化：根据收敛情况自动调整迭代次数。
"""

import networkx as nx
import numpy as np
from typing import Dict, Optional

from .base import InitializationStrategy, OptimizerConfig


class FMMEInitializer(InitializationStrategy):
    """
    Force-Directed Layout Initializer (带动态优化)
    
    特点：
    - 高温策略 (T=100)
    - 适用于随机初始布局
    - 支持自适应迭代：根据收敛情况自动调整
    
    推荐迭代次数：
    - 小图（<30节点）：100-200 次
    - 中图（30-100节点）：50-100 次
    - 大图（>100节点）：20-50 次
    
    动态优化策略：
    - 至少运行 min_iterations 次（默认50）
    - 如果连续 patience 次（默认5）都有改善 → 继续
    - 如果连续 patience 次没有改善 → 终止
    - 最多运行 max_iterations 次（默认200）
    """
    
    def __init__(
        self, 
        spring_iterations: Optional[int] = None,
        enable_adaptive: bool = True,
        min_iterations: int = 50,
        max_iterations: int = 200,
        patience: int = 5,
        improvement_threshold: float = 0.001
    ):
        """
        初始化 FMME 策略
        
        Args:
            spring_iterations: 固定迭代次数（如果指定，则禁用自适应）
            enable_adaptive: 是否启用自适应迭代（默认True）
            min_iterations: 最小迭代次数（默认50）
            max_iterations: 最大迭代次数（默认200）
            patience: 容忍无改善的连续次数（默认5）
            improvement_threshold: 判定为"改善"的最小变化率（默认0.001=0.1%）
        
        示例：
            # 固定迭代
            FMMEInitializer(spring_iterations=100)
            
            # 自适应迭代（推荐）
            FMMEInitializer(enable_adaptive=True, min_iterations=50, patience=5)
            
            # 自定义自适应参数
            FMMEInitializer(min_iterations=30, max_iterations=150, patience=10)
        """
        self.enable_adaptive = enable_adaptive and (spring_iterations is None)
        self.spring_iterations = spring_iterations
        self.min_iterations = min_iterations
        self.max_iterations = max_iterations
        self.patience = patience
        self.improvement_threshold = improvement_threshold
        
        # 统计信息（用于调试和分析）
        self.last_actual_iterations = 0
        self.last_convergence_reason = ""
    
    def _calculate_spring_energy(self, G: nx.Graph, pos: dict) -> float:
        """
        计算 Spring Layout 的能量（用于判断收敛）
        
        能量 = 引力能 + 斥力能
        - 引力能：连接的节点希望靠近
        - 斥力能：所有节点希望分散
        
        能量越低 = 布局越优
        """
        energy = 0.0
        k = 1.0 / np.sqrt(len(G.nodes()))  # 理想距离
        
        nodes = list(G.nodes())
        
        # 引力能（只计算有边的节点对）
        for u, v in G.edges():
            dx = pos[u][0] - pos[v][0]
            dy = pos[u][1] - pos[v][1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist > 0:
                energy += dist * dist / k  # 引力 ∝ d²
        
        # 斥力能（所有节点对）
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                u, v = nodes[i], nodes[j]
                dx = pos[u][0] - pos[v][0]
                dy = pos[u][1] - pos[v][1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist > 0:
                    energy += k * k / dist  # 斥力 ∝ k²/d
        
        return energy
    
    def _adaptive_spring_layout(self, G: nx.Graph) -> dict:
        """
        自适应 Spring Layout（根据收敛情况动态调整迭代次数）
        
        策略：
        1. 至少运行 min_iterations 次
        2. 每次迭代后计算能量
        3. 如果连续 patience 次都有改善（能量下降）→ 继续
        4. 如果连续 patience 次没有明显改善 → 终止
        5. 最多运行 max_iterations 次
        """
        # 初始化随机位置
        pos = nx.random_layout(G)
        
        # 能量历史（用于判断收敛）
        energy_history = []
        no_improvement_count = 0
        
        print(f"[FMME Adaptive] Starting with min={self.min_iterations}, max={self.max_iterations}, patience={self.patience}")
        
        for iteration in range(self.max_iterations):
            # 执行一次 Spring Layout 迭代
            pos = nx.spring_layout(G, pos=pos, iterations=1, seed=42)
            
            # 计算当前能量
            current_energy = self._calculate_spring_energy(G, pos)
            energy_history.append(current_energy)
            
            # 检查是否达到最小迭代次数
            if iteration < self.min_iterations - 1:
                # 还没到最小次数，继续
                if (iteration + 1) % 10 == 0:
                    print(f"  [{iteration + 1:>3}/{self.min_iterations}] Energy: {current_energy:.6f} (warming up)")
                continue
            
            # 已达到最小迭代次数，开始检查收敛
            if len(energy_history) >= 2:
                prev_energy = energy_history[-2]
                improvement = (prev_energy - current_energy) / prev_energy if prev_energy > 0 else 0
                
                if improvement > self.improvement_threshold:
                    # 有改善
                    no_improvement_count = 0
                    if (iteration + 1) % 10 == 0:
                        print(f"  [{iteration + 1:>3}] Energy: {current_energy:.6f} (improved {improvement*100:.3f}%)")
                else:
                    # 无改善
                    no_improvement_count += 1
                    if (iteration + 1) % 10 == 0:
                        print(f"  [{iteration + 1:>3}] Energy: {current_energy:.6f} (plateau: {no_improvement_count}/{self.patience})")
                
                # 检查是否应该终止
                if no_improvement_count >= self.patience:
                    self.last_actual_iterations = iteration + 1
                    self.last_convergence_reason = f"converged (no improvement for {self.patience} iterations)"
                    print(f"  [CONVERGED] Stopped at iteration {iteration + 1} (energy plateaued)")
                    break
        else:
            # 达到最大迭代次数
            self.last_actual_iterations = self.max_iterations
            self.last_convergence_reason = "max_iterations reached"
            print(f"  [MAX ITER] Stopped at {self.max_iterations} iterations")
        
        return pos
    
    def generate_positions(
        self,
        num_nodes: int,
        edges: list,
        width: int,
        height: int,
        existing_positions: Dict = None
    ) -> Dict[int, tuple]:
        """使用 Spring Layout 生成布局（支持自适应优化）"""
        if num_nodes == 0:
            return {}
        
        if num_nodes == 1:
            return {0: (width // 2, height // 2)}
        
        # 构建 NetworkX 图
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        G.add_edges_from(edges)
        
        # 选择策略：固定迭代 vs 自适应迭代
        if self.enable_adaptive:
            print(f"[FMME] Using adaptive Spring Layout")
            pos = self._adaptive_spring_layout(G)
            print(f"[FMME] Completed in {self.last_actual_iterations} iterations ({self.last_convergence_reason})")
        else:
            # 固定迭代次数
            iterations = self.spring_iterations or 50
            print(f"[FMME] Using fixed Spring Layout ({iterations} iterations)")
            pos = nx.spring_layout(G, iterations=iterations, seed=42)
            self.last_actual_iterations = iterations
            self.last_convergence_reason = "fixed iterations"
        
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
