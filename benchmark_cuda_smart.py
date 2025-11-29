#!/usr/bin/env python3
"""
智能CUDA Benchmark - 優化K值
策略: 多次運行SA,選擇K值最小的結果
"""
import json
import sys
import os
import time
from pathlib import Path
from datetime import datetime

# 添加CUDA DLL路徑
os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'build_artifacts'))
sys.path.insert(0, 'src')

import planar_cuda
from LCNv1.core.geometry import Point, GeometryCore


def calculate_k_value(nodes_x, nodes_y, edges):
    """
    計算K值 (每條邊的最大交叉數)
    
    Args:
        nodes_x: x坐標列表
        nodes_y: y坐標列表
        edges: 邊列表 [(src, tgt), ...]
    
    Returns:
        k值, 總交叉數, 每條邊的交叉數字典
    """
    # 創建節點字典
    nodes = {i: {'x': nodes_x[i], 'y': nodes_y[i]} for i in range(len(nodes_x))}
    
    edge_crossings = {}
    
    for i in range(len(edges)):
        src1, tgt1 = edges[i]
        p1 = Point(nodes[src1]['x'], nodes[src1]['y'])
        p2 = Point(nodes[tgt1]['x'], nodes[tgt1]['y'])
        
        crossings = 0
        for j in range(len(edges)):
            if i == j:
                continue
            
            src2, tgt2 = edges[j]
            
            # 跳過共享端點
            if src1 in (src2, tgt2) or tgt1 in (src2, tgt2):
                continue
            
            q1 = Point(nodes[src2]['x'], nodes[src2]['y'])
            q2 = Point(nodes[tgt2]['x'], nodes[tgt2]['y'])
            
            if GeometryCore.segments_intersect(p1, p2, q1, q2):
                crossings += 1
        
        edge_crossings[i] = crossings
    
    k = max(edge_crossings.values()) if edge_crossings else 0
    total = sum(edge_crossings.values()) // 2
    
    return k, total, edge_crossings


def run_cuda_optimization(instance_path, iterations=5000, cell_size=100, seed=None):
    """
    運行一次CUDA優化
    
    Returns:
        (k值, 總交叉數, nodes_x, nodes_y, 運行時間)
    """
    # 加載實例
    with open(instance_path, 'r') as f:
        data = json.load(f)
    
    nodes_x = [n['x'] for n in data['nodes']]
    nodes_y = [n['y'] for n in data['nodes']]
    edges = [(e['source'], e['target']) for e in data['edges']]
    
    # 獲取width/height約束
    width = data.get('width', 1000000)
    height = data.get('height', 1000000)
    
    # 創建求解器 (使用width/height約束)
    solver = planar_cuda.PlanarSolver(nodes_x, nodes_y, edges, cell_size=cell_size,
                                      width=width, height=height)
    
    # 運行SA
    start_time = time.time()
    stats = solver.run_sa_optimization(iterations, 100.0, 0.95)
    elapsed = time.time() - start_time
    
    # 獲取結果
    final_nodes_x, final_nodes_y = solver.get_coordinates()
    
    # 計算K值
    k, total, edge_crossings = calculate_k_value(final_nodes_x, final_nodes_y, edges)
    
    return k, total, final_nodes_x, final_nodes_y, elapsed, edge_crossings


def optimize_for_k(instance_path, num_runs=5, iterations=5000, cell_size=100):
    """
    多次運行優化,選擇K值最小的結果
    
    Args:
        instance_path: 實例文件路徑
        num_runs: 運行次數
        iterations: 每次SA迭代數
        cell_size: 空間哈希單元大小
    
    Returns:
        最佳結果字典
    """
    print(f"\n{'='*80}")
    print(f"智能優化: {Path(instance_path).name}")
    print(f"{'='*80}")
    print(f"策略: 運行{num_runs}次,選擇K值最小的結果")
    print(f"每次迭代: {iterations}")
    
    best_result = None
    best_k = float('inf')
    
    results = []
    
    for run in range(num_runs):
        print(f"\n[Run {run+1}/{num_runs}]", end=" ")
        
        k, total, nodes_x, nodes_y, elapsed, edge_crossings = run_cuda_optimization(
            instance_path, iterations, cell_size
        )
        
        results.append({
            'run': run + 1,
            'k': k,
            'total': total,
            'time': elapsed
        })
        
        print(f"K={k}, 總交叉={total}, 時間={elapsed:.2f}s", end="")
        
        if k < best_k or (k == best_k and total < best_result['total']):
            best_k = k
            best_result = {
                'k': k,
                'total': total,
                'nodes_x': nodes_x,
                'nodes_y': nodes_y,
                'time': elapsed,
                'edge_crossings': edge_crossings,
                'run': run + 1
            }
            print(" ⭐ 新最佳!")
        else:
            print()
    
    # 顯示統計
    print(f"\n{'='*80}")
    print("運行統計")
    print(f"{'='*80}")
    print(f"\n{'Run':<8} {'K值':<8} {'總交叉數':<12} {'時間(s)':<10}")
    print("-"*40)
    for r in results:
        marker = " ⭐" if r['run'] == best_result['run'] else ""
        print(f"{r['run']:<8} {r['k']:<8} {r['total']:<12} {r['time']:<10.2f}{marker}")
    
    print(f"\n{'='*80}")
    print(f"✅ 最佳結果: Run {best_result['run']}")
    print(f"   K值: {best_result['k']}")
    print(f"   總交叉數: {best_result['total']}")
    print(f"   時間: {best_result['time']:.2f}s")
    
    # 顯示交叉最多的邊
    sorted_edges = sorted(best_result['edge_crossings'].items(), 
                         key=lambda x: x[1], reverse=True)[:5]
    print(f"\n   交叉數最多的邊:")
    for idx, count in sorted_edges:
        print(f"     邊{idx}: {count} 次交叉")
    
    return best_result


def save_result(instance_path, result, output_dir='results'):
    """保存結果到JSON文件"""
    # 加載原始實例
    with open(instance_path, 'r') as f:
        data = json.load(f)
    
    # 更新節點坐標
    for i, node in enumerate(data['nodes']):
        node['x'] = int(result['nodes_x'][i])
        node['y'] = int(result['nodes_y'][i])
    
    # 創建輸出目錄
    now = datetime.now()
    output_path = Path(output_dir) / f"{now.month:02d}-{now.day:02d}-{now.hour:02d}"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 構建文件名
    num_nodes = len(data['nodes'])
    filename = f"{num_nodes}-nodes-cu-k{result['k']}.json"
    output_file = output_path / filename
    
    # 保存
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n[SAVED] Result saved to: {output_file}")
    return output_file


def main():
    print("="*80)
    print("智能CUDA Benchmark - K值優化")
    print("="*80)
    
    # 測試實例
    instances = [
        'live-2025-example-instances/15-nodes.json',
        'live-2025-example-instances/70-nodes.json',
        'live-2025-example-instances/100-nodes.json',
    ]
    
    output_dir = r'D:\D_backup\2025\tum\25W\hackthon\Hackathon-Nov-25-Heilbronn43\results'
    
    all_results = []
    
    for instance_path in instances:
        if not Path(instance_path).exists():
            print(f"\n⚠️  文件不存在: {instance_path}")
            continue
        
        # 優化K值 (運行5次)
        result = optimize_for_k(instance_path, num_runs=5, iterations=5000)
        
        # 保存結果
        output_file = save_result(instance_path, result, output_dir)
        
        all_results.append({
            'instance': Path(instance_path).name,
            'k': result['k'],
            'total': result['total'],
            'time': result['time'],
            'output': str(output_file)
        })
    
    # 最終總結
    print(f"\n{'='*80}")
    print("總結")
    print(f"{'='*80}")
    print(f"\n{'實例':<25} {'K值':<8} {'總交叉':<10} {'時間(s)':<10}")
    print("-"*55)
    for r in all_results:
        print(f"{r['instance']:<25} {r['k']:<8} {r['total']:<10} {r['time']:<10.2f}")
    
    print(f"\n✅ Benchmark完成!")
    print(f"結果保存在: {output_dir}")


if __name__ == "__main__":
    main()
