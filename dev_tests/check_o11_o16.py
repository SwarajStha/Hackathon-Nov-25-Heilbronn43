#!/usr/bin/env python3
"""
详细检查 o11 和 o16 文件
特别注意：
1. 重复的节点坐标（问题定义说不允许！）
2. 节点是否在边的内部
"""

import json
from collections import defaultdict

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def check_duplicate_coordinates(data):
    """检查是否有重复坐标"""
    coords = {}
    duplicates = []
    
    for node in data['nodes']:
        coord = (node['x'], node['y'])
        if coord in coords:
            duplicates.append({
                'coord': coord,
                'nodes': [coords[coord], node['id']]
            })
        else:
            coords[coord] = node['id']
    
    return duplicates

def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def segments_intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def calculate_k_value(data, verbose=True):
    """计算 K 值"""
    nodes = {n['id']: (n['x'], n['y']) for n in data['nodes']}
    edges = [(e['source'], e['target']) for e in data['edges']]
    
    edge_crossings = defaultdict(int)
    
    for i, (s1, t1) in enumerate(edges):
        A = nodes[s1]
        B = nodes[t1]
        for j, (s2, t2) in enumerate(edges):
            if i == j:
                continue
            if s1 == s2 or s1 == t2 or t1 == s2 or t1 == t2:
                continue
            C = nodes[s2]
            D = nodes[t2]
            if segments_intersect(A, B, C, D):
                edge_crossings[i] += 1
    
    k = max(edge_crossings.values()) if edge_crossings else 0
    
    if verbose:
        max_edges = [i for i, cnt in edge_crossings.items() if cnt == k]
        print(f"  K-value: {k}")
        print(f"  有{k}个交叉的边:")
        for idx in max_edges[:5]:  # 只显示前5条
            s, t = edges[idx]
            print(f"    边 #{idx}: {s} -> {t}")
    
    return k

def analyze_file(filepath, expected_k):
    """完整分析文件"""
    print(f"\n{'='*70}")
    print(f"文件: {filepath}")
    print(f"预期 K 值: {expected_k}")
    print(f"{'='*70}")
    
    data = load_json(filepath)
    
    # 检查重复坐标
    duplicates = check_duplicate_coordinates(data)
    if duplicates:
        print(f"\n❌ 发现重复坐标 ({len(duplicates)} 组):")
        for dup in duplicates:
            print(f"  坐标 {dup['coord']}: 节点 {dup['nodes']}")
        print(f"\n  问题定义要求: 'no two vertices are placed on the same coordinate'")
        print(f"  这违反了规则！")
    else:
        print(f"\n✓ 所有节点坐标唯一")
    
    # 计算 K 值
    print(f"\n计算结果:")
    k = calculate_k_value(data, verbose=True)
    
    match = "✓" if k == expected_k else "✗"
    print(f"\n结果: K={k}, 预期={expected_k} {match}")
    
    if duplicates and k != expected_k:
        print(f"\n⚠️  重复坐标可能导致计算错误！")
        print(f"  当两个节点在同一位置时，从它们出发的边会重叠")
        print(f"  这可能导致额外的交叉！")
    
    return k, duplicates

print("="*70)
print("详细分析 o11 和 o16 文件")
print("="*70)

k11, dup11 = analyze_file("results/29-01-22/15-node-o11.json", 11)
k16, dup16 = analyze_file("results/29-01-22/15-node-o16.json", 16)

print(f"\n{'='*70}")
print("总结")
print(f"{'='*70}")
print(f"o11: K={k11} (预期11) {'✓' if k11==11 else '✗'}, 重复坐标: {len(dup11)}组")
print(f"o16: K={k16} (预期16) {'✓' if k16==16 else '✗'}, 重复坐标: {len(dup16)}组")

if dup11 or dup16:
    print(f"\n⚠️  发现重复坐标！这违反了问题定义！")
    print(f"   'no two vertices are placed on the same coordinate'")
