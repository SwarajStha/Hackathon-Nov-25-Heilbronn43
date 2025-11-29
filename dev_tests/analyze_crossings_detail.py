#!/usr/bin/env python3
"""
详细分析每条边的交叉数
"""

import sys
sys.path.insert(0, 'src')

import json
from LCNv1.core.geometry import Point, GeometryCore
from LCNv1.core.graph import GraphData, GridState

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_crossings_per_edge(filepath):
    print(f"\n{'='*70}")
    print(f"详细交叉分析: {filepath}")
    print(f"{'='*70}")
    
    data = load_json(filepath)
    nodes = {n['id']: Point(n['x'], n['y']) for n in data['nodes']}
    edges = [(e['source'], e['target']) for e in data['edges']]
    
    graph = GraphData(len(nodes), edges)
    state = GridState(nodes, data.get('width', 1000), data.get('height', 1000))
    
    # 计算每条边的交叉数
    edge_crossings = []
    
    for i in range(len(edges)):
        src1, tgt1 = edges[i]
        p1 = state.get_position(src1)
        p2 = state.get_position(tgt1)
        
        crossings = 0
        crossing_with = []
        
        for j in range(len(edges)):
            if i == j:
                continue
            
            src2, tgt2 = edges[j]
            
            # 跳过共享端点
            if src1 in (src2, tgt2) or tgt1 in (src2, tgt2):
                continue
            
            q1 = state.get_position(src2)
            q2 = state.get_position(tgt2)
            
            if GeometryCore.segments_intersect(p1, p2, q1, q2):
                crossings += 1
                crossing_with.append(j)
        
        edge_crossings.append({
            'edge_idx': i,
            'endpoints': (src1, tgt1),
            'crossings': crossings,
            'crossing_with': crossing_with
        })
    
    # 排序：交叉数从高到低
    edge_crossings.sort(key=lambda x: x['crossings'], reverse=True)
    
    # 显示前10条交叉最多的边
    print(f"\n前20条交叉最多的边:")
    print(f"{'边#':<6} {'端点':<12} {'交叉数':<8} 交叉的边")
    print("-" * 70)
    
    for ec in edge_crossings[:20]:
        crossing_str = ', '.join(f"#{x}" for x in ec['crossing_with'][:5])
        if len(ec['crossing_with']) > 5:
            crossing_str += f", ... (+{len(ec['crossing_with'])-5})"
        
        print(f"{ec['edge_idx']:<6} {str(ec['endpoints']):<12} {ec['crossings']:<8} {crossing_str}")
    
    # K值
    max_crossings = max(ec['crossings'] for ec in edge_crossings)
    total_crossings = sum(ec['crossings'] for ec in edge_crossings) // 2
    
    print(f"\n{'='*70}")
    print(f"K-value (最大交叉数): {max_crossings}")
    print(f"Total crossings: {total_crossings}")
    print(f"{'='*70}")
    
    # 找出 K 值对应的边
    k_edges = [ec for ec in edge_crossings if ec['crossings'] == max_crossings]
    print(f"\n达到 K={max_crossings} 的边:")
    for ec in k_edges:
        src, tgt = ec['endpoints']
        p1 = state.get_position(src)
        p2 = state.get_position(tgt)
        print(f"  边#{ec['edge_idx']} ({src}->{tgt}): {p1} -> {p2}")
        print(f"    与这些边交叉: {ec['crossing_with'][:10]}")
    
    return max_crossings, total_crossings

# 分析
analyze_crossings_per_edge("results/11-29-02/15-nodes-ours-v2.json")
