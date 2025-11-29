"""
测试单个实例 - 验证增强策略
"""
import json
import time
from pathlib import Path
import sys
from datetime import datetime

# 修正路徑
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from LCNv1.core.geometry import Point, GeometryCore
from LCNv1.core.graph import GraphData
from LCNv1.core.cost import SoftMaxCost
from LCNv1.initialization import FMMEInitializer
from LCNv1.strategies.enhanced import EnhancedSolverStrategy


def main():
    # 测试文件
    instance_file = Path(__file__).parent / 'live-2025-example-instances' / '15-nodes.json'
    
    # 加载数据
    with open(instance_file, 'r') as f:
        instance = json.load(f)
    
    print(f"加载实例: 15-nodes.json")
    print(f"  节点数: {len(instance['nodes'])}")
    print(f"  边数: {len(instance['edges'])}")
    
    # 创建初始化策略
    print(f"\n创建 FMME 初始化策略...")
    init_strategy = FMMEInitializer(
        spring_iterations=100  # NetworkX spring layout 迭代次数
    )
    
    # 创建增强求解器
    print(f"创建增强求解器...")
    strategy = EnhancedSolverStrategy(
        w_cross=100.0,
        w_len=1.0,
        power=2,
        init_strategy=init_strategy
    )
    
    # 加载JSON（自动应用初始化策略）
    print(f"加载数据并初始化...")
    strategy.load_from_json(str(instance_file))
    
    # 运行优化
    print(f"\n开始优化 (5000次迭代)...")
    start = time.time()
    result = strategy.solve(iterations=5000)
    elapsed = time.time() - start
    
    print(f"\n✅ 结果:")
    print(f"  最终交叉数: {result['total_crossings']}")
    print(f"  K值: {result['k']}")
    print(f"  能量: {result['energy']:.0f}")
    print(f"  用时: {elapsed:.2f}s")
    
    # 保存结果
    now = datetime.now()
    timestamp = now.strftime("%d-%H-%M")
    output_dir = Path(__file__).parent / 'results' / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / '15-nodes-ours.json'
    strategy.export_to_json(str(output_file))
    
    print(f"  结果已保存: {output_file}")


if __name__ == '__main__':
    main()
