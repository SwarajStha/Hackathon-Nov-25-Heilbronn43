#!/usr/bin/env python3
"""
测试 FMME 初始化多次，看是否产生违规
"""

import sys
sys.path.insert(0, 'src')

from LCNv1.strategies.enhanced import EnhancedSolverStrategy
from LCNv1.initialization.fmme import FMMEInitializer
from LCNv1.core.k_plane_cost import KPlaneCost
from LCNv1.core.violation_repair import ViolationRepair

# 创建 solver
cost_func = KPlaneCost(w_k=10000.0, w_cross=100.0, w_len=1.0)
init_strategy = FMMEInitializer(spring_iterations=50)
solver = EnhancedSolverStrategy(
    init_strategy=init_strategy,
    cost_function=cost_func
)

print("测试 FMME 初始化 10 次...")
print("="*70)

for trial in range(10):
    # 重新加载数据（重新初始化）
    solver.load_from_json("live-2025-example-instances/15-nodes.json")
    
    # 检测违规
    repairer = ViolationRepair(solver.graph, solver.state)
    violations = repairer.detect_violations()
    
    total = (len(violations['duplicate_coords']) +
            len(violations['nodes_on_edges']) +
            len(violations['overlapping_edges']))
    
    status = "✅" if total == 0 else "❌"
    print(f"Trial {trial+1}: {status} Total violations: {total}")
    
    if total > 0:
        print(f"  - Duplicate: {len(violations['duplicate_coords'])}")
        print(f"  - Nodes on edges: {len(violations['nodes_on_edges'])}")
        print(f"  - Overlapping edges: {len(violations['overlapping_edges'])}")

print("\n结论:")
print("如果FMME经常产生违规，说明需要在初始化后总是运行修复算法")
