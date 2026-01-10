#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重新优化 15-nodes 实例（带约束检查）
"""
import json
import sys
import os
import time
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')
os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin')
sys.path.insert(0, 'build_artifacts')

import planar_cuda


def check_constraints(nodes_x, nodes_y, edges):
    """检查约束违反"""
    from collections import Counter
    
    violations = []
    
    # 检查共线节点
    x_counter = Counter(nodes_x)
    y_counter = Counter(nodes_y)
    
    for x, count in x_counter.items():
        if count >= 3:
            violations.append(f"垂直共线 x={x}: {count}个节点")
    
    for y, count in y_counter.items():
        if count >= 3:
            violations.append(f"水平共线 y={y}: {count}个节点")
    
    # 检查共线边
    collinear_edges = 0
    for i in range(len(edges)):
        for j in range(i+1, len(edges)):
            s1, t1 = edges[i]
            s2, t2 = edges[j]
            
            if s1 in (s2, t2) or t1 in (s2, t2):
                continue
            
            p1x, p1y = nodes_x[s1], nodes_y[s1]
            p2x, p2y = nodes_x[t1], nodes_y[t1]
            p3x, p3y = nodes_x[s2], nodes_y[s2]
            p4x, p4y = nodes_x[t2], nodes_y[t2]
            
            cross1 = (p2x - p1x) * (p3y - p1y) - (p2y - p1y) * (p3x - p1x)
            cross2 = (p2x - p1x) * (p4y - p1y) - (p2y - p1y) * (p4x - p1x)
            
            if cross1 == 0 and cross2 == 0:
                collinear_edges += 1
    
    if collinear_edges > 0:
        violations.append(f"{collinear_edges}对共线边")
    
    return violations


def main():
    instance_path = r'live-2025-example-instances\15-nodes.json'
    
    with open(instance_path, 'r') as f:
        data = json.load(f)
    
    nodes_x = [n['x'] for n in data['nodes']]
    nodes_y = [n['y'] for n in data['nodes']]
    edges = [(e['source'], e['target']) for e in data['edges']]
    width = data.get('width', 1000000)
    height = data.get('height', 1000000)
    
    print("="*80)
    print("重新优化 15-nodes 实例（带约束检查）")
    print("="*80)
    print(f"节点数: {len(nodes_x)}, 边数: {len(edges)}")
    
    # 检查初始约束
    print(f"\n初始约束检查...")
    initial_violations = check_constraints(nodes_x, nodes_y, edges)
    if initial_violations:
        print(f"  ⚠️  初始文件有约束违反: {', '.join(initial_violations)}")
    else:
        print(f"  ✅ 初始文件满足所有约束")
    
    best_k = float('inf')
    best_result = None
    num_runs = 15
    
    for run in range(num_runs):
        print(f"\n[Run {run+1}/{num_runs}] ", end="", flush=True)
        
        solver = planar_cuda.PlanarSolver(nodes_x, nodes_y, edges, 
                                          cell_size=100, width=width, height=height)
        
        start = time.time()
        stats = solver.run_sa_optimization(30000, 100.0, 0.95, "bottleneck_p3")
        elapsed = time.time() - start
        
        final_x, final_y = solver.get_coordinates()
        k_final = solver.calculate_k_value()
        total_final = solver.calculate_total_crossings()
        
        # 检查约束
        violations = check_constraints(final_x, final_y, edges)
        
        print(f"K={k_final}, 总交叉={total_final}, 时间={elapsed:.2f}s", end="")
        
        if violations:
            print(f" ❌ 违反约束: {', '.join(violations)}")
            continue
        else:
            print(f" ✅ 满足约束", end="")
        
        if k_final < best_k or (k_final == best_k and total_final < best_result['total']):
            best_k = k_final
            best_result = {
                'k': k_final,
                'total': total_final,
                'nodes_x': final_x,
                'nodes_y': final_y,
                'time': elapsed,
                'run': run + 1
            }
            print(" ⭐ 新最佳!")
        else:
            print()
    
    if best_result:
        print(f"\n{'='*80}")
        print(f"最佳结果 (Run {best_result['run']}):")
        print(f"  K值: {best_result['k']}")
        print(f"  总交叉数: {best_result['total']}")
        print(f"  时间: {best_result['time']:.2f}s")
        
        # 最终约束检查
        final_violations = check_constraints(best_result['nodes_x'], best_result['nodes_y'], edges)
        if final_violations:
            print(f"  ❌ 约束违反: {', '.join(final_violations)}")
        else:
            print(f"  ✅ 满足所有约束")
        
        # 保存结果
        timestamp = datetime.now().strftime("%H-%M-%S")
        result_dir = Path(f"results/{timestamp}")
        result_dir.mkdir(parents=True, exist_ok=True)
        
        output_data = {
            'nodes': [
                {'id': i, 'x': int(best_result['nodes_x'][i]), 'y': int(best_result['nodes_y'][i])}
                for i in range(len(best_result['nodes_x']))
            ],
            'edges': data['edges'],
            'width': width,
            'height': height
        }
        
        output_file = result_dir / f"15-nodes-cu-k{best_result['k']}-fixed.json"
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"  保存到: {output_file}")
    else:
        print(f"\n❌ 所有运行都违反了约束!")


if __name__ == "__main__":
    main()
