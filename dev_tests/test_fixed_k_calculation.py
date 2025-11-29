#!/usr/bin/env python3
"""
測試修復後的K值計算
"""

import sys
sys.path.insert(0, 'src')

import json
from LCNv1.core.geometry import Point
from LCNv1.core.graph import GraphData, GridState
from LCNv1.core.cost import SoftMaxCost


def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def test_file(filepath, name, expected_k=None):
    """測試文件的K值計算"""
    print(f"\n{'='*80}")
    print(f"測試: {name}")
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
    
    # 使用cost function計算
    cost_fn = SoftMaxCost(w_cross=100.0, w_len=1.0, power=2)
    
    # 獲取交叉統計
    k, total_crossings = cost_fn.get_crossing_stats(graph, state)
    
    print(f"節點數: {len(nodes)}")
    print(f"邊數: {len(edges)}")
    print(f"\n計算結果:")
    print(f"  K值: {k}")
    print(f"  總交叉數: {total_crossings}")
    
    if expected_k is not None:
        if k == expected_k:
            print(f"  ✅ K值正確 (expected={expected_k})")
        else:
            print(f"  ❌ K值錯誤! expected={expected_k}, got={k}")
    
    # 計算能量
    total_cost = cost_fn.calculate(graph, state)
    crossing_energy = cost_fn._calculate_crossing_energy(graph, state)
    length_energy = cost_fn.get_length_energy(graph, state)
    
    print(f"\n能量:")
    print(f"  交叉能量: {crossing_energy:.2f}")
    print(f"  長度能量: {length_energy:.2f}")
    print(f"  總能量: {total_cost:.2f}")
    
    return k, total_crossings


if __name__ == "__main__":
    print("="*80)
    print("測試修復後的K值計算")
    print("="*80)
    
    # 測試文件
    test_file(
        "live-2025-example-instances/sol-15-nodes-5-planar.json",
        "官方標準答案",
        expected_k=5
    )
    
    test_file(
        "results/11-29-02/15-nodes-o5.json",
        "我們的結果",
        expected_k=5
    )
    
    print(f"\n{'='*80}")
    print("測試完成!")
    print(f"{'='*80}")
