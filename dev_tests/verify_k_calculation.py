#!/usr/bin/env python3
"""
驗證K值計算的正確性

根據題目:
- K = max{每條邊的交叉數}
- 邊是無向的
- 交叉 = proper intersection (不包括端點接觸、共享端點的邊)
"""

import sys
sys.path.insert(0, 'src')

import json
from collections import defaultdict
from LCNv1.core.geometry import Point, GeometryCore


def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def count_edge_crossings_detail(nodes, edges):
    """
    詳細計算每條邊的交叉數
    
    返回:
    - edge_crossings: dict {edge_idx: crossing_count}
    - crossing_pairs: list of (edge1, edge2, intersection_point)
    - max_k: 最大K值
    """
    edge_crossings = defaultdict(int)
    crossing_pairs = []
    
    for i in range(len(edges)):
        src1, tgt1 = edges[i]['source'], edges[i]['target']
        p1 = Point(nodes[src1]['x'], nodes[src1]['y'])
        p2 = Point(nodes[tgt1]['x'], nodes[tgt1]['y'])
        
        for j in range(i + 1, len(edges)):
            src2, tgt2 = edges[j]['source'], edges[j]['target']
            
            # 跳過共享端點的邊 (無向邊,任意端點相同都算共享)
            shared_endpoints = {src1, tgt1} & {src2, tgt2}
            if shared_endpoints:
                continue
            
            q1 = Point(nodes[src2]['x'], nodes[src2]['y'])
            q2 = Point(nodes[tgt2]['x'], nodes[tgt2]['y'])
            
            # 檢查交叉
            if GeometryCore.segments_intersect(p1, p2, q1, q2):
                edge_crossings[i] += 1
                edge_crossings[j] += 1
                crossing_pairs.append((i, j))
    
    max_k = max(edge_crossings.values()) if edge_crossings else 0
    
    return edge_crossings, crossing_pairs, max_k


def analyze_file(filepath, name):
    """分析文件的K值"""
    print(f"\n{'='*80}")
    print(f"分析: {name}")
    print(f"{'='*80}")
    
    data = load_json(filepath)
    nodes = {n['id']: n for n in data['nodes']}
    edges = data['edges']
    
    print(f"節點數: {len(nodes)}")
    print(f"邊數: {len(edges)}")
    
    # 計算K值
    edge_crossings, crossing_pairs, max_k = count_edge_crossings_detail(nodes, edges)
    
    print(f"\n交叉統計:")
    print(f"  總交叉對數: {len(crossing_pairs)}")
    print(f"  K值 (max crossings per edge): {max_k}")
    
    # 顯示K值最高的邊
    if edge_crossings:
        print(f"\n交叉數最多的邊:")
        sorted_edges = sorted(edge_crossings.items(), key=lambda x: x[1], reverse=True)
        
        for edge_idx, count in sorted_edges[:10]:
            src = edges[edge_idx]['source']
            tgt = edges[edge_idx]['target']
            print(f"  邊{edge_idx} ({src}↔{tgt}): {count} 次交叉")
        
        # 統計交叉數分佈
        print(f"\n交叉數分佈:")
        crossing_dist = defaultdict(int)
        for count in edge_crossings.values():
            crossing_dist[count] += 1
        
        for count in sorted(crossing_dist.keys(), reverse=True):
            num_edges = crossing_dist[count]
            print(f"  {count} 次交叉: {num_edges} 條邊")
    
    return {
        'k': max_k,
        'total_crossings': len(crossing_pairs),
        'edge_crossings': edge_crossings
    }


if __name__ == "__main__":
    files = [
        ("live-2025-example-instances/sol-15-nodes-5-planar.json", "官方標準答案 (聲稱K=5)"),
        ("results/11-29-02/15-nodes-o5.json", "我們的結果"),
    ]
    
    print("="*80)
    print("K值計算驗證工具")
    print("="*80)
    print("\n題目定義:")
    print("  K = max{每條邊的交叉數}")
    print("  交叉 = proper intersection (邊在內部相交)")
    print("  不計算:")
    print("    - 共享端點的邊")
    print("    - 端點接觸")
    print("    - 共線重疊")
    
    results = {}
    for filepath, name in files:
        try:
            results[name] = analyze_file(filepath, name)
        except Exception as e:
            print(f"\n❌ 錯誤: {e}")
            import traceback
            traceback.print_exc()
    
    # 對比
    print(f"\n{'='*80}")
    print("結果對比")
    print(f"{'='*80}")
    
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"  K值: {result['k']}")
        print(f"  總交叉數: {result['total_crossings']}")
