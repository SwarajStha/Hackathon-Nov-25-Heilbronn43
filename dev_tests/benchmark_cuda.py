#!/usr/bin/env python3
"""
CUDA Benchmark - 使用GPU加速運行benchmark
"""
import json
import sys
import os
import time
from pathlib import Path

# 添加CUDA DLL路徑
os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'build_artifacts'))

import planar_cuda


def count_crossings_simple(nodes, edges):
    """簡單計算交叉數 (用於驗證)"""
    crossings = 0
    for i in range(len(edges)):
        for j in range(i + 1, len(edges)):
            e1 = edges[i]
            e2 = edges[j]
            
            # 跳過共享端點
            if (e1['source'] == e2['source'] or e1['source'] == e2['target'] or
                e1['target'] == e2['source'] or e1['target'] == e2['target']):
                continue
            
            # 檢查交叉 (簡化版)
            src1 = nodes[e1['source']]
            tgt1 = nodes[e1['target']]
            src2 = nodes[e2['source']]
            tgt2 = nodes[e2['target']]
            
            # 這裡簡化處理,實際應該用幾何計算
            crossings += 1 if _simple_intersect(src1, tgt1, src2, tgt2) else 0
    
    return crossings


def _simple_intersect(p1, p2, q1, q2):
    """簡化的交叉檢測"""
    # 使用叉積檢測
    def cross(o, a, b):
        return (a['x'] - o['x']) * (b['y'] - o['y']) - (a['y'] - o['y']) * (b['x'] - o['x'])
    
    d1 = cross(p1, p2, q1)
    d2 = cross(p1, p2, q2)
    d3 = cross(q1, q2, p1)
    d4 = cross(q1, q2, p2)
    
    return d1 * d2 < 0 and d3 * d4 < 0


def save_result_to_json(solver, instance_path, output_dir, k_value):
    """
    保存求解結果到JSON文件
    
    Args:
        solver: CUDA求解器實例
        instance_path: 原始實例路徑
        output_dir: 輸出目錄
        k_value: K值(如果可用)
    """
    # 獲取當前座標
    coords = solver.get_coordinates()
    x_coords, y_coords = coords
    
    # 載入原始數據以獲取邊
    with open(instance_path, 'r') as f:
        original_data = json.load(f)
    
    # 構建輸出數據
    result = {
        'nodes': [
            {'id': i, 'x': int(x_coords[i]), 'y': int(y_coords[i])}
            for i in range(len(x_coords))
        ],
        'edges': original_data['edges'],
        'width': original_data.get('width', 1000),
        'height': original_data.get('height', 1000)
    }
    
    # 生成輸出文件名
    from datetime import datetime
    now = datetime.now()
    time_dir = now.strftime("%m-%d-%H")  # HH:TT 格式改為 MM-DD-HH
    
    # 獲取節點數
    num_nodes = len(x_coords)
    
    # 構建文件名: number-nodes-cu-k
    if k_value != "N/A":
        filename = f"{num_nodes}-nodes-cu-{k_value}.json"
    else:
        filename = f"{num_nodes}-nodes-cu.json"
    
    # 創建目錄
    output_path = Path(output_dir) / time_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 保存文件
    output_file = output_path / filename
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n[SAVED] Result saved to: {output_file}")
    return output_file


def benchmark_instance(instance_path, iterations=5000, cell_size=100, save_results=True, output_dir='results'):
    """
    使用CUDA求解器benchmark單個實例
    
    Args:
        instance_path: 實例文件路徑
        iterations: SA迭代次數
        cell_size: 空間哈希網格大小
        save_results: 是否保存結果
        output_dir: 輸出目錄
    """
    print(f"\n{'='*80}")
    print(f"測試: {Path(instance_path).name}")
    print(f"{'='*80}")
    
    # 載入數據
    with open(instance_path, 'r') as f:
        data = json.load(f)
    
    nodes = {n['id']: n for n in data['nodes']}
    edges = data['edges']
    
    print(f"節點數: {len(nodes)}")
    print(f"邊數: {len(edges)}")
    
    # 準備CUDA求解器輸入
    x_coords = [nodes[i]['x'] for i in range(len(nodes))]
    y_coords = [nodes[i]['y'] for i in range(len(nodes))]
    edge_list = [(e['source'], e['target']) for e in edges]
    
    # 創建求解器
    print(f"\n初始化CUDA求解器 (cell_size={cell_size})...")
    solver = planar_cuda.PlanarSolver(x_coords, y_coords, edge_list, cell_size=cell_size)
    
    # 獲取初始狀態
    initial_crossings = solver.calculate_total_crossings()
    print(f"初始交叉數: {initial_crossings}")
    
    # 運行SA優化
    print(f"\n運行模擬退火 ({iterations} 迭代)...")
    start_time = time.time()
    
    # 使用run_sa_optimization方法 (參數: iterations, start_temp, cooling_rate)
    stats = solver.run_sa_optimization(
        iterations,
        100.0,  # start_temp
        0.95    # cooling_rate
    )
    
    elapsed_time = time.time() - start_time
    
    # 獲取最終結果
    final_crossings = solver.calculate_total_crossings()
    
    print(f"\n結果:")
    print(f"  初始交叉數: {initial_crossings}")
    print(f"  最終交叉數: {final_crossings}")
    print(f"  改善: {initial_crossings - final_crossings} ({(1 - final_crossings/max(initial_crossings, 1))*100:.1f}%)")
    print(f"  運行時間: {elapsed_time:.2f}秒")
    print(f"  速度: {iterations/elapsed_time:.0f} iter/s")
    
    # 統計信息
    print(f"\nSA統計:")
    print(f"  接受移動: {stats['accepted_moves']}")
    print(f"  拒絕移動: {stats['rejected_moves']}")
    print(f"  接受率: {stats['accepted_moves']/(stats['accepted_moves']+stats['rejected_moves'])*100:.1f}%")
    
    # 計算K值 (使用總交叉數)
    k = "N/A"  # CUDA版本需要額外實現K值計算
    total = final_crossings
    print(f"\n最終交叉數: {total}")
    
    # 保存結果
    output_file = None
    if save_results:
        output_file = save_result_to_json(solver, instance_path, output_dir, k)
    
    return {
        'file': Path(instance_path).name,
        'nodes': len(nodes),
        'edges': len(edges),
        'initial_crossings': initial_crossings,
        'final_crossings': final_crossings,
        'k_value': k,
        'total_crossings': total,
        'time': elapsed_time,
        'iterations': iterations,
        'accepted_rate': stats['accepted_moves']/(stats['accepted_moves']+stats['rejected_moves']),
        'output_file': str(output_file) if output_file else None
    }


def main():
    """主函數"""
    print("="*80)
    print("CUDA加速Benchmark測試")
    print("="*80)
    
    # 輸出目錄
    output_dir = r'D:\D_backup\2025\tum\25W\hackthon\Hackathon-Nov-25-Heilbronn43\results'
    
    # 測試實例列表
    instances = [
        'live-2025-example-instances/15-nodes.json',
        'live-2025-example-instances/70-nodes.json',
        'live-2025-example-instances/100-nodes.json',
    ]
    
    results = []
    
    for instance_path in instances:
        if not os.path.exists(instance_path):
            print(f"\n⚠️ 文件不存在: {instance_path}")
            continue
        
        try:
            result = benchmark_instance(
                instance_path, 
                iterations=5000, 
                cell_size=100,
                save_results=True,
                output_dir=output_dir
            )
            results.append(result)
        except Exception as e:
            print(f"\n[ERROR] {e}")
            import traceback
            traceback.print_exc()
    
    # 總結
    print(f"\n{'='*80}")
    print("總結")
    print(f"{'='*80}")
    
    if results:
        print(f"\n{'文件':<25} {'節點':<8} {'邊':<8} {'初始':<8} {'最終':<8} {'K值':<6} {'時間(s)':<10}")
        print("-"*80)
        
        for r in results:
            print(f"{r['file']:<25} {r['nodes']:<8} {r['edges']:<8} "
                  f"{r['initial_crossings']:<8} {r['final_crossings']:<8} "
                  f"{r['k_value']:<6} {r['time']:<10.2f}")
    
    print("\n✅ Benchmark完成!")


if __name__ == "__main__":
    main()
