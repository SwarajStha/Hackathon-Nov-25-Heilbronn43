#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试违规文件是否能被修正
"""
import json
import sys
import os
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding='utf-8')
os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin')
sys.path.insert(0, 'build_artifacts')

import planar_cuda


def check_collinear_violations(nodes_x, nodes_y):
    """检查共线违规"""
    violations = []
    
    # 检查垂直共线 (相同x坐标)
    x_groups = defaultdict(list)
    for i, x in enumerate(nodes_x):
        x_groups[x].append((i, nodes_y[i]))
    
    for x, nodes in x_groups.items():
        if len(nodes) >= 3:
            violations.append(f"垂直共线 x={x}: {len(nodes)}个节点")
    
    # 检查水平共线 (相同y坐标)
    y_groups = defaultdict(list)
    for i, y in enumerate(nodes_y):
        y_groups[y].append((i, nodes_x[i]))
    
    for y, nodes in y_groups.items():
        if len(nodes) >= 3:
            violations.append(f"水平共线 y={y}: {len(nodes)}个节点")
    
    return violations


def test_file(file_path, num_runs=5):
    """测试单个文件"""
    print("="*80)
    print(f"测试文件: {file_path}")
    print("="*80)
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    nodes_x = [n['x'] for n in data['nodes']]
    nodes_y = [n['y'] for n in data['nodes']]
    edges = [(e['source'], e['target']) for e in data['edges']]
    width = data.get('width', 1000000)
    height = data.get('height', 1000000)
    
    print(f"节点数: {len(nodes_x)}, 边数: {len(edges)}")
    
    # 检查初始违规
    print(f"\n初始状态检查:")
    initial_violations = check_collinear_violations(nodes_x, nodes_y)
    if initial_violations:
        print(f"  ❌ 发现 {len(initial_violations)} 个违规:")
        for v in initial_violations[:5]:
            print(f"     {v}")
        if len(initial_violations) > 5:
            print(f"     ... 还有 {len(initial_violations)-5} 个")
    else:
        print(f"  ✅ 初始文件无违规")
    
    # 运行优化
    print(f"\n运行 {num_runs} 次优化...")
    best_k = float('inf')
    best_result = None
    
    for run in range(num_runs):
        print(f"  [Run {run+1}/{num_runs}] ", end="", flush=True)
        
        solver = planar_cuda.PlanarSolver(nodes_x, nodes_y, edges, 
                                          cell_size=100, width=width, height=height)
        
        # 使用较短的迭代次数快速测试
        iterations = 10000 if len(nodes_x) <= 100 else 5000
        stats = solver.run_sa_optimization(iterations, 100.0, 0.95, "bottleneck_p3")
        
        final_x, final_y = solver.get_coordinates()
        k_final = solver.calculate_k_value()
        total_final = solver.calculate_total_crossings()
        
        # 检查违规
        violations = check_collinear_violations(final_x, final_y)
        
        print(f"K={k_final}, 总交叉={total_final}", end="")
        
        if violations:
            print(f" ❌ {len(violations)}个违规")
        else:
            print(f" ✅ 无违规", end="")
            
            if k_final < best_k:
                best_k = k_final
                best_result = {
                    'k': k_final,
                    'total': total_final,
                    'nodes_x': final_x,
                    'nodes_y': final_y,
                    'run': run + 1
                }
                print(" ⭐ 新最佳!")
            else:
                print()
    
    # 总结
    print(f"\n{'='*80}")
    if best_result:
        print(f"✅ 成功修正! 最佳结果 (Run {best_result['run']}):")
        print(f"  K值: {best_result['k']}")
        print(f"  总交叉数: {best_result['total']}")
        print(f"  无共线违规")
        
        # 验证
        final_violations = check_collinear_violations(best_result['nodes_x'], best_result['nodes_y'])
        if final_violations:
            print(f"  ⚠️  警告: 仍有 {len(final_violations)} 个违规!")
            for v in final_violations[:3]:
                print(f"     {v}")
        
        return True
    else:
        print(f"❌ 所有运行都有违规，无法修正")
        return False


def main():
    print("\n测试违规文件修正\n")
    
    # 测试两个违规文件
    files = [
        r"results\06-27-38\15-nodes-cu-k4-fixed.json",  # 之前显示为正确的
        r"results\06-36-59\225-nodes-cu-k477.json",     # 有大量违规
    ]
    
    results = []
    for file_path in files:
        if os.path.exists(file_path):
            success = test_file(file_path, num_runs=5)
            results.append((file_path, success))
        else:
            print(f"文件不存在: {file_path}")
            results.append((file_path, False))
        print()
    
    # 最终总结
    print("="*80)
    print("总结")
    print("="*80)
    for file_path, success in results:
        filename = os.path.basename(file_path)
        status = "✅ 成功修正" if success else "❌ 修正失败"
        print(f"{filename:<30} {status}")
    
    all_success = all(r[1] for r in results)
    if all_success:
        print("\n✅ 所有违规文件都已成功修正!")
    else:
        print("\n⚠️  部分文件仍有问题")


if __name__ == "__main__":
    main()
