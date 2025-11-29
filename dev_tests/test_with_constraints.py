#!/usr/bin/env python3
"""
使用新的几何约束重新运行 15-nodes 优化
"""

import sys
sys.path.insert(0, 'src')

from LCNv1.strategies.enhanced import EnhancedSolverStrategy
from LCNv1.initialization.fmme import FMMEInitializer
from LCNv1.core.k_plane_cost import KPlaneCost
import json
from datetime import datetime
import os

def main():
    # 配置
    input_file = "live-2025-example-instances/15-nodes.json"
    timestamp = datetime.now().strftime("%m-%d-%H")
    output_dir = f"results/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/15-nodes-ours-v2.json"
    
    print("="*70)
    print("使用新的几何约束优化 15-nodes")
    print("="*70)
    print(f"输入: {input_file}")
    print(f"输出: {output_file}")
    print()
    print("约束:")
    print("  ✓ 无重复坐标")
    print("  ✓ 无节点在边内部")
    print("  ✓ 无边重叠")
    print("="*70)
    
    # 创建 solver
    cost_func = KPlaneCost(
        w_k=10000.0,      # 极高的 K 惩罚
        w_cross=100.0,    # 交叉数惩罚
        w_len=1.0         # 边长惩罚
    )
    
    init_strategy = FMMEInitializer(spring_iterations=50)
    solver = EnhancedSolverStrategy(
        init_strategy=init_strategy,
        cost_function=cost_func
    )
    
    # 加载数据
    solver.load_from_json(input_file)
    
    # 运行优化
    result = solver.solve(
        iterations=5000,
        initial_temp=100.0,
        cooling_rate=0.9950
    )
    
    # 构建输出格式
    with open(input_file, 'r') as f:
        original_data = json.load(f)
    
    output_data = {
        'nodes': result['nodes'],
        'edges': original_data['edges'],
        'width': original_data.get('width', 1000),
        'height': original_data.get('height', 1000)
    }
    
    # 保存结果
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n结果已保存: {output_file}")
    print(f"Best K-value: {result['k']}")
    print(f"Total crossings: {result['total_crossings']}")
    print(f"Energy: {result['energy']:.0f}")
    print(f"Acceptance rate: {result['stats']['acceptance_rate']*100:.1f}%")

if __name__ == "__main__":
    main()
