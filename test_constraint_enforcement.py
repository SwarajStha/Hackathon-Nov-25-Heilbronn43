#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试约束检查是否在瓶颈惩罚模式下工作
"""
import json
import sys
import os

sys.stdout.reconfigure(encoding='utf-8')
os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin')
sys.path.insert(0, 'build_artifacts')

import planar_cuda


def test_constraint_enforcement():
    """测试约束在优化过程中是否被强制执行"""
    print("="*80)
    print("测试: 约束强制执行（瓶颈惩罚模式）")
    print("="*80)
    
    # 简单的4个节点
    # 初始状态: 节点不共线
    nodes_x = [0, 10, 5, 15]
    nodes_y = [0, 0,  5, 5]
    edges = [(0, 2), (1, 3)]  # 两条不相交的边
    
    solver = planar_cuda.PlanarSolver(nodes_x, nodes_y, edges)
    
    # 初始状态
    initial_k = solver.calculate_k_value()
    print(f"\n初始状态:")
    print(f"  节点: {list(zip(nodes_x, nodes_y))}")
    print(f"  边: {edges}")
    print(f"  K值: {initial_k}")
    
    # 运行优化 (瓶颈惩罚p=3)
    print(f"\n运行 SA 优化 (瓶颈惩罚 p=3, 5000 迭代)...")
    stats = solver.run_sa_optimization(5000, 100.0, 0.95, "bottleneck_p3")
    
    # 获取最终坐标
    final_x, final_y = solver.get_coordinates()
    final_k = solver.calculate_k_value()
    
    print(f"\n最终状态:")
    print(f"  节点: {list(zip(final_x, final_y))}")
    print(f"  K值: {final_k}")
    
    # 检查是否有共线节点（三个或更多在同一条线）
    from collections import Counter
    x_counter = Counter(final_x)
    y_counter = Counter(final_y)
    
    x_violations = [x for x, count in x_counter.items() if count >= 3]
    y_violations = [y for y, count in y_counter.items() if count >= 3]
    
    print(f"\n约束检查:")
    if x_violations:
        print(f"  ❌ 发现垂直共线: x={x_violations}")
    else:
        print(f"  ✅ 无垂直共线")
    
    if y_violations:
        print(f"  ❌ 发现水平共线: y={y_violations}")
    else:
        print(f"  ✅ 无水平共线")
    
    # 检查边是否共线
    has_collinear_edges = False
    for i in range(len(edges)):
        for j in range(i+1, len(edges)):
            s1, t1 = edges[i]
            s2, t2 = edges[j]
            
            # 跳过共享端点
            if s1 in (s2, t2) or t1 in (s2, t2):
                continue
            
            p1x, p1y = final_x[s1], final_y[s1]
            p2x, p2y = final_x[t1], final_y[t1]
            p3x, p3y = final_x[s2], final_y[s2]
            p4x, p4y = final_x[t2], final_y[t2]
            
            # 检查共线
            cross1 = (p2x - p1x) * (p3y - p1y) - (p2y - p1y) * (p3x - p1x)
            cross2 = (p2x - p1x) * (p4y - p1y) - (p2y - p1y) * (p4x - p1x)
            
            if cross1 == 0 and cross2 == 0:
                print(f"  ❌ 边 {i} 和边 {j} 共线!")
                has_collinear_edges = True
    
    if not has_collinear_edges:
        print(f"  ✅ 无共线边")
    
    if not x_violations and not y_violations and not has_collinear_edges:
        print(f"\n✅ 所有约束都被满足!")
        return True
    else:
        print(f"\n❌ 存在约束违反!")
        return False


def main():
    success = test_constraint_enforcement()
    
    if success:
        print("\n" + "="*80)
        print("测试通过：约束在瓶颈惩罚模式下被正确强制执行")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("测试失败：约束检查可能仍有问题")
        print("="*80)


if __name__ == "__main__":
    main()
