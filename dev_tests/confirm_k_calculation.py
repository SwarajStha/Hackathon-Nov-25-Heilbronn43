#!/usr/bin/env python3
"""
重新确认：根据问题定义正确计算 K-value

问题定义：
"To count crossings we take the maximum number of crossings over all edges"
"Crossings will always be counted pairwise"

K-value = max(每条边的交叉数)
"""

import json
from collections import defaultdict

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def ccw(A, B, C):
    """Counter-clockwise test."""
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def segments_intersect(A, B, C, D):
    """
    Check if line segments AB and CD intersect properly.
    Proper intersection = they cross in their interiors (not at endpoints).
    """
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def calculate_k_value_correct(data, verbose=False):
    """
    根据问题定义正确计算 K-value：
    1. 对于每条边，计算它与多少其他边有交叉
    2. K = max(所有边的交叉数)
    3. 交叉计数规则：成对计数（pairwise）
    """
    nodes = {n['id']: (n['x'], n['y']) for n in data['nodes']}
    edges = [(e['source'], e['target']) for e in data['edges']]
    
    # 计算每条边的交叉数
    edge_crossing_count = defaultdict(int)
    
    for i, (s1, t1) in enumerate(edges):
        if s1 == t1:
            continue
        
        A = nodes[s1]
        B = nodes[t1]
        
        for j, (s2, t2) in enumerate(edges):
            if i == j:  # 同一条边
                continue
            if s2 == t2:
                continue
            
            # 跳过共享端点的边（不算交叉）
            if s1 == s2 or s1 == t2 or t1 == s2 or t1 == t2:
                continue
            
            C = nodes[s2]
            D = nodes[t2]
            
            # 检查是否有proper intersection
            if segments_intersect(A, B, C, D):
                edge_crossing_count[i] += 1
    
    # K-value = 最大交叉数
    k_value = max(edge_crossing_count.values()) if edge_crossing_count else 0
    
    if verbose:
        # 显示详细信息
        print(f"\n总边数: {len(edges)}")
        print(f"K-value (最大交叉数): {k_value}")
        
        # 找出K值最大的边
        max_edges = [i for i, cnt in edge_crossing_count.items() if cnt == k_value]
        print(f"\n有{k_value}个交叉的边:")
        for edge_idx in max_edges:
            s, t = edges[edge_idx]
            print(f"  边 #{edge_idx}: 节点{s} -> 节点{t}")
        
        # 交叉分布
        distribution = defaultdict(int)
        for cnt in edge_crossing_count.values():
            distribution[cnt] += 1
        
        # 计算有0个交叉的边数
        edges_with_zero = len(edges) - len(edge_crossing_count)
        if edges_with_zero > 0:
            distribution[0] = edges_with_zero
        
        print(f"\n交叉数分布:")
        for cnt in sorted(distribution.keys()):
            print(f"  {distribution[cnt]} 条边有 {cnt} 个交叉")
    
    return k_value

def main():
    print("="*70)
    print("根据问题定义重新计算 K-value")
    print("="*70)
    print("\n问题定义：")
    print("  'To count crossings we take the maximum number of")
    print("   crossings over all edges'")
    print("\n计算方法：")
    print("  K = max(每条边与其他边的交叉数)")
    print("="*70)
    
    files = [
        ("15-node-others.json", "应该是10"),
        ("15-node-o2.json", "应该是11"),
        ("15-nodes-ours.json", "我们的结果")
    ]
    
    results = {}
    
    for filename, description in files:
        filepath = f"results/29-01-22/{filename}"
        print(f"\n{'='*70}")
        print(f"文件: {filename} ({description})")
        print(f"{'='*70}")
        
        data = load_json(filepath)
        k = calculate_k_value_correct(data, verbose=True)
        results[filename] = k
    
    print("\n" + "="*70)
    print("最终结果汇总")
    print("="*70)
    print(f"15-node-others.json:  K = {results['15-node-others.json']} (期望: 10)")
    print(f"15-node-o2.json:      K = {results['15-node-o2.json']} (期望: 11)")
    print(f"15-nodes-ours.json:   K = {results['15-nodes-ours.json']} (期望: ?)")
    print("="*70)
    
    # 验证
    if results['15-node-o2.json'] == 11:
        print("\n✓ 15-node-o2.json 的 K=11 正确匹配!")
    
    print("\n结论:")
    print(f"  我们的优化结果 K={results['15-nodes-ours.json']}")
    print(f"  参考结果 o2 的 K={results['15-node-o2.json']}")
    
    if results['15-nodes-ours.json'] < results['15-node-o2.json']:
        print(f"  ✓ 我们的结果更好! (K值更小)")
    elif results['15-nodes-ours.json'] == results['15-node-o2.json']:
        print(f"  = 我们的结果相同")
    else:
        print(f"  ✗ 我们的结果较差 (K值更大)")

if __name__ == "__main__":
    main()
