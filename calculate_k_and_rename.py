#!/usr/bin/env python3
"""
計算CUDA結果的K值並更新文件名
"""
import sys
sys.path.insert(0, 'src')

import json
from pathlib import Path
from LCNv1.core.geometry import Point, GeometryCore


def calculate_k_value(filepath):
    """計算文件的K值"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
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
            
            # 跳過共享端點
            if src1 in (src2, tgt2) or tgt1 in (src2, tgt2):
                continue
            
            q1 = Point(nodes[src2]['x'], nodes[src2]['y'])
            q2 = Point(nodes[tgt2]['x'], nodes[tgt2]['y'])
            
            if GeometryCore.segments_intersect(p1, p2, q1, q2):
                crossings += 1
        
        edge_crossings[i] = crossings
    
    k = max(edge_crossings.values()) if edge_crossings else 0
    total = sum(edge_crossings.values()) // 2  # 除以2因為每個交叉被計算兩次
    
    return k, total


def rename_with_k_value(filepath):
    """重命名文件以包含K值"""
    k, total = calculate_k_value(filepath)
    
    # 解析當前文件名
    filepath = Path(filepath)
    parts = filepath.stem.split('-')
    
    # 構建新文件名: number-nodes-cu-k{k}
    num_nodes = parts[0]
    new_name = f"{num_nodes}-nodes-cu-k{k}.json"
    new_path = filepath.parent / new_name
    
    # 重命名
    if new_path.exists():
        new_path.unlink()  # 刪除已存在的文件
    filepath.rename(new_path)
    
    print(f"✅ {filepath.name} → {new_name}")
    print(f"   K值: {k}, 總交叉數: {total}")
    
    return new_path, k, total


if __name__ == "__main__":
    results_dir = Path(r'D:\D_backup\2025\tum\25W\hackthon\Hackathon-Nov-25-Heilbronn43\results\11-29-03')
    
    print("="*80)
    print("計算K值並更新文件名")
    print("="*80)
    
    results = []
    for file in results_dir.glob('*-nodes-cu.json'):
        print(f"\n處理: {file.name}")
        new_path, k, total = rename_with_k_value(file)
        results.append((file.stem.split('-')[0], k, total))
    
    print("\n" + "="*80)
    print("總結")
    print("="*80)
    print(f"\n{'節點數':<10} {'K值':<10} {'總交叉數':<15}")
    print("-"*35)
    for num_nodes, k, total in sorted(results, key=lambda x: int(x[0])):
        print(f"{num_nodes:<10} {k:<10} {total:<15}")
