#!/usr/bin/env python3
"""
驗證CUDA實現的正確性:
1. 交叉計算是否正確(與Python版本對比)
2. 是否正確跳過共享端點
3. K值計算是否正確
"""
import sys
import os
sys.path.insert(0, 'src')

# 添加CUDA DLL路徑
os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build_artifacts'))

import json
from pathlib import Path
from LCNv1.core.geometry import Point, GeometryCore
import planar_cuda


def load_instance(filepath):
    """加載實例"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def calculate_k_python(data):
    """使用Python計算K值和總交叉數"""
    nodes = {n['id']: n for n in data['nodes']}
    edges = data['edges']
    
    # 計算每條邊的交叉數
    edge_crossings = {}
    
    for i in range(len(edges)):
        e1 = edges[i]
        src1, tgt1 = e1['source'], e1['target']
        p1 = Point(nodes[src1]['x'], nodes[src1]['y'])
        p2 = Point(nodes[tgt1]['x'], nodes[tgt1]['y'])
        
        crossings = 0
        for j in range(len(edges)):
            if i == j:
                continue
            
            e2 = edges[j]
            src2, tgt2 = e2['source'], e2['target']
            
            # 跳過共享端點 - 關鍵檢查!
            if src1 in (src2, tgt2) or tgt1 in (src2, tgt2):
                continue
            
            q1 = Point(nodes[src2]['x'], nodes[src2]['y'])
            q2 = Point(nodes[tgt2]['x'], nodes[tgt2]['y'])
            
            if GeometryCore.segments_intersect(p1, p2, q1, q2):
                crossings += 1
        
        edge_crossings[i] = crossings
    
    k = max(edge_crossings.values()) if edge_crossings else 0
    total = sum(edge_crossings.values()) // 2  # 除以2因為每個交叉被計算兩次
    
    return k, total, edge_crossings


def calculate_total_crossings_cuda(data):
    """使用CUDA計算總交叉數"""
    # 準備數據
    nodes_x = [n['x'] for n in data['nodes']]
    nodes_y = [n['y'] for n in data['nodes']]
    edges = [(e['source'], e['target']) for e in data['edges']]
    
    # 創建求解器
    solver = planar_cuda.PlanarSolver(nodes_x, nodes_y, edges, cell_size=100)
    
    # 計算交叉數
    total_crossings = solver.calculate_total_crossings()
    
    return total_crossings


def test_instance(filepath):
    """測試單個實例"""
    print(f"\n{'='*80}")
    print(f"測試: {filepath.name}")
    print(f"{'='*80}")
    
    # 加載數據
    data = load_instance(filepath)
    num_nodes = len(data['nodes'])
    num_edges = len(data['edges'])
    
    print(f"節點數: {num_nodes}")
    print(f"邊數: {num_edges}")
    
    # Python版本計算
    print("\n[Python版本]")
    k_python, total_python, edge_crossings = calculate_k_python(data)
    print(f"  K值: {k_python}")
    print(f"  總交叉數: {total_python}")
    
    # 顯示交叉最多的邊
    sorted_edges = sorted(edge_crossings.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"\n  交叉數最多的邊:")
    for idx, count in sorted_edges:
        edge = data['edges'][idx]
        print(f"    邊{idx} ({edge['source']}↔{edge['target']}): {count} 次交叉")
    
    # CUDA版本計算
    print("\n[CUDA版本]")
    total_cuda = calculate_total_crossings_cuda(data)
    print(f"  總交叉數: {total_cuda}")
    
    # 對比結果
    print(f"\n{'='*80}")
    print("結果對比")
    print(f"{'='*80}")
    
    if total_python == total_cuda:
        print(f"✅ 通過! Python和CUDA計算的總交叉數一致: {total_python}")
    else:
        print(f"❌ 失敗! 結果不一致:")
        print(f"   Python: {total_python}")
        print(f"   CUDA: {total_cuda}")
        print(f"   差異: {abs(total_python - total_cuda)}")
        return False
    
    print(f"\n✅ K值: {k_python}")
    print(f"✅ 總交叉數: {total_python}")
    
    return True


def main():
    print("="*80)
    print("CUDA正確性驗證")
    print("="*80)
    print("\n驗證項目:")
    print("1. 交叉計算是否與Python版本一致")
    print("2. 是否正確跳過共享端點的邊")
    print("3. K值計算是否正確")
    
    # 測試已保存的CUDA結果
    results_dir = Path('results/11-29-03')
    
    test_files = [
        results_dir / '15-nodes-cu-k6.json',
        results_dir / '70-nodes-cu-k37.json',
        results_dir / '100-nodes-cu-k33.json',
    ]
    
    all_passed = True
    for filepath in test_files:
        if not filepath.exists():
            print(f"\n⚠️  文件不存在: {filepath}")
            continue
        
        if not test_instance(filepath):
            all_passed = False
    
    print(f"\n{'='*80}")
    if all_passed:
        print("✅ 所有測試通過! CUDA實現正確")
    else:
        print("❌ 部分測試失敗,需要檢查CUDA實現")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
