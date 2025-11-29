#!/usr/bin/env python3
"""
分析K值計算的正確性

根據題目定義:
k = max{crossings per edge}

但我們需要考慮違規情況:
1. 點與點重疊 (Overlaps)
2. 點壓在線上 (Node on edge interior)
3. 邊與邊重疊 (Edge overlaps)
4. 向下的邊 (Not upward)

這些違規都會影響K值的計算
"""

import sys
sys.path.insert(0, 'src')

import json
from collections import defaultdict
from LCNv1.core.geometry import Point, GeometryCore
from LCNv1.core.graph import GraphData, GridState


def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_edge_crossings(graph: GraphData, state: GridState, verbose=True):
    """
    詳細分析每條邊的交叉情況
    
    返回:
    - edge_crossing_counts: dict {edge_idx: crossing_count}
    - crossing_pairs: list of (edge1, edge2) that cross
    - max_k: maximum crossings on any single edge
    """
    edge_crossings = defaultdict(int)
    crossing_pairs = []
    
    edges = graph.edges
    
    for i in range(len(edges)):
        src1, tgt1 = edges[i]
        p1 = state.get_position(src1)
        p2 = state.get_position(tgt1)
        
        for j in range(i + 1, len(edges)):
            src2, tgt2 = edges[j]
            
            # 跳過共享端點的邊
            if src1 in (src2, tgt2) or tgt1 in (src2, tgt2):
                continue
            
            q1 = state.get_position(src2)
            q2 = state.get_position(tgt2)
            
            if GeometryCore.segments_intersect(p1, p2, q1, q2):
                edge_crossings[i] += 1
                edge_crossings[j] += 1
                crossing_pairs.append((i, j))
    
    max_k = max(edge_crossings.values()) if edge_crossings else 0
    
    if verbose:
        print(f"\n交叉分析:")
        print(f"  總交叉對數: {len(crossing_pairs)}")
        print(f"  最大K值: {max_k}")
        print(f"\n每條邊的交叉數:")
        
        # 按交叉數排序
        sorted_edges = sorted(edge_crossings.items(), key=lambda x: x[1], reverse=True)
        
        for edge_idx, count in sorted_edges[:10]:  # 顯示前10條
            src, tgt = edges[edge_idx]
            print(f"    邊{edge_idx} ({src}->{tgt}): {count} crossings")
        
        if len(sorted_edges) > 10:
            print(f"    ... 還有 {len(sorted_edges) - 10} 條邊")
    
    return edge_crossings, crossing_pairs, max_k


def check_violations(graph: GraphData, state: GridState, verbose=True):
    """
    檢查所有違規情況
    
    返回違規統計
    """
    violations = {
        'duplicate_coords': [],
        'nodes_on_edges': [],
        'overlapping_edges': [],
        'downward_edges': []
    }
    
    nodes = state.get_all_positions()
    edges = graph.edges
    
    # 1. 重複座標
    coords_map = defaultdict(list)
    for node_id, pos in nodes.items():
        coords_map[(pos.x, pos.y)].append(node_id)
    
    for coord, node_list in coords_map.items():
        if len(node_list) > 1:
            violations['duplicate_coords'].append((coord, node_list))
    
    # 2. 節點在邊內部
    for node_id, node_pos in nodes.items():
        for edge_idx in range(len(edges)):
            src, tgt = edges[edge_idx]
            if node_id == src or node_id == tgt:
                continue
            
            edge_start = state.get_position(src)
            edge_end = state.get_position(tgt)
            
            if GeometryCore.point_on_segment_interior(node_pos, edge_start, edge_end):
                violations['nodes_on_edges'].append((node_id, edge_idx))
    
    # 3. 邊重疊
    for i in range(len(edges)):
        src1, tgt1 = edges[i]
        p1 = state.get_position(src1)
        p2 = state.get_position(tgt1)
        
        for j in range(i + 1, len(edges)):
            src2, tgt2 = edges[j]
            
            # 跳過共享端點
            if src1 in (src2, tgt2) or tgt1 in (src2, tgt2):
                continue
            
            q1 = state.get_position(src2)
            q2 = state.get_position(tgt2)
            
            if GeometryCore.segments_overlap(p1, p2, q1, q2):
                violations['overlapping_edges'].append((i, j))
    
    # 4. 向下的邊
    for edge_idx in range(len(edges)):
        src, tgt = edges[edge_idx]
        src_pos = state.get_position(src)
        tgt_pos = state.get_position(tgt)
        
        if tgt_pos.y < src_pos.y:
            violations['downward_edges'].append(edge_idx)
        elif tgt_pos.y == src_pos.y:
            # 水平邊也是違規
            violations['downward_edges'].append(edge_idx)
    
    total = (len(violations['duplicate_coords']) +
             len(violations['nodes_on_edges']) +
             len(violations['overlapping_edges']) +
             len(violations['downward_edges']))
    
    if verbose:
        print(f"\n違規檢測:")
        print(f"  總違規數: {total}")
        
        if violations['duplicate_coords']:
            print(f"\n  ❌ 重複座標: {len(violations['duplicate_coords'])} 組")
            for coord, nodes in violations['duplicate_coords'][:3]:
                print(f"     {coord}: 節點 {nodes}")
        
        if violations['nodes_on_edges']:
            print(f"\n  ❌ 節點在邊內部: {len(violations['nodes_on_edges'])} 處")
            for node_id, edge_idx in violations['nodes_on_edges'][:3]:
                src, tgt = edges[edge_idx]
                print(f"     節點{node_id} 在邊{edge_idx}({src}->{tgt})內部")
        
        if violations['overlapping_edges']:
            print(f"\n  ❌ 邊重疊: {len(violations['overlapping_edges'])} 對")
            for e1, e2 in violations['overlapping_edges'][:3]:
                s1, t1 = edges[e1]
                s2, t2 = edges[e2]
                print(f"     邊{e1}({s1}->{t1}) × 邊{e2}({s2}->{t2})")
        
        if violations['downward_edges']:
            print(f"\n  ❌ 向下/水平邊: {len(violations['downward_edges'])} 條")
            for edge_idx in violations['downward_edges'][:3]:
                src, tgt = edges[edge_idx]
                src_pos = state.get_position(src)
                tgt_pos = state.get_position(tgt)
                direction = "水平" if src_pos.y == tgt_pos.y else "向下"
                print(f"     邊{edge_idx}({src}->{tgt}): {direction}")
    
    return violations, total


def analyze_file(filepath, name):
    """完整分析一個文件"""
    print(f"\n{'='*80}")
    print(f"分析: {name}")
    print(f"文件: {filepath}")
    print(f"{'='*80}")
    
    data = load_json(filepath)
    
    # 構建圖
    nodes = {n['id']: Point(n['x'], n['y']) for n in data['nodes']}
    edges = [(e['source'], e['target']) for e in data['edges']]
    
    graph = GraphData(len(nodes), edges)
    state = GridState(
        nodes,
        width=data.get('width', 1000),
        height=data.get('height', 1000),
        graph_data=graph,
        enable_constraints=False
    )
    
    print(f"\n基本信息:")
    print(f"  節點數: {len(nodes)}")
    print(f"  邊數: {len(edges)}")
    print(f"  畫布: {state.width} × {state.height}")
    
    # 檢查違規
    violations, total_violations = check_violations(graph, state)
    
    # 分析交叉
    edge_crossings, crossing_pairs, max_k = analyze_edge_crossings(graph, state)
    
    # 判斷結果是否有效
    is_valid = total_violations == 0
    
    print(f"\n{'='*80}")
    print(f"結論:")
    print(f"  K值: {max_k}")
    print(f"  總交叉數: {len(crossing_pairs)}")
    print(f"  有效性: {'✅ VALID' if is_valid else '❌ INVALID (有違規)'}")
    
    if not is_valid:
        print(f"\n  ⚠️ 注意: 此K值可能不準確,因為存在違規!")
    
    return {
        'k': max_k,
        'total_crossings': len(crossing_pairs),
        'violations': total_violations,
        'valid': is_valid,
        'edge_crossings': edge_crossings
    }


if __name__ == "__main__":
    files = [
        ("live-2025-example-instances/sol-15-nodes-5-planar.json", "官方標準答案 (聲稱K=5)"),
        ("results/11-29-02/15-nodes-o5.json", "當前打開的文件"),
    ]
    
    print("="*80)
    print("K值計算分析工具")
    print("="*80)
    print("\n題目定義:")
    print("  k = max{每條邊的交叉數}")
    print("  目標: 最小化 k")
    print("\n違規情況:")
    print("  1. 點與點重疊 → RED HALO")
    print("  2. 點壓在線上 → RED HALO")
    print("  3. 邊重疊 → 違規")
    print("  4. 向下/水平邊 → 違規 (必須向上)")
    
    results = {}
    for filepath, name in files:
        try:
            results[name] = analyze_file(filepath, name)
        except FileNotFoundError:
            print(f"\n⚠️ 文件不存在: {filepath}")
        except Exception as e:
            print(f"\n❌ 錯誤: {e}")
            import traceback
            traceback.print_exc()
    
    # 對比結果
    print(f"\n{'='*80}")
    print("結果對比")
    print(f"{'='*80}")
    
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"  K值: {result['k']}")
        print(f"  總交叉數: {result['total_crossings']}")
        print(f"  違規數: {result['violations']}")
        print(f"  狀態: {'✅ 有效' if result['valid'] else '❌ 無效'}")
