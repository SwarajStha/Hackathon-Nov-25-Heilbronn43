#!/usr/bin/env python3
"""
重新检查计算方法
对比官方标准答案 sol-15-nodes-5-planar.json
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
    """检查线段 AB 和 CD 是否相交（不包括端点）"""
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def analyze_detailed(filepath, name):
    """详细分析文件"""
    print(f"\n{'='*70}")
    print(f"分析: {name}")
    print(f"文件: {filepath}")
    print(f"{'='*70}")
    
    data = load_json(filepath)
    nodes = {n['id']: (n['x'], n['y']) for n in data['nodes']}
    edges = [(e['source'], e['target']) for e in data['edges']]
    
    print(f"节点数: {len(nodes)}")
    print(f"边数: {len(edges)}")
    
    # 检查重复坐标
    coords = {}
    duplicates = []
    for nid, coord in nodes.items():
        if coord in coords.values():
            existing = [k for k, v in coords.items() if v == coord]
            duplicates.append((coord, existing + [nid]))
        coords[nid] = coord
    
    if duplicates:
        print(f"\n❌ 重复坐标:")
        for coord, node_ids in duplicates:
            print(f"  {coord}: 节点 {node_ids}")
    else:
        print(f"\n✓ 所有坐标唯一")
    
    # 计算每条边的交叉数
    edge_crossings = defaultdict(list)  # 存储交叉的边对
    
    print(f"\n开始计算交叉...")
    for i, (s1, t1) in enumerate(edges):
        A = nodes[s1]
        B = nodes[t1]
        
        for j, (s2, t2) in enumerate(edges):
            if i >= j:  # 只计算一次
                continue
            
            # 检查是否共享端点
            shared_endpoint = (s1 == s2 or s1 == t2 or t1 == s2 or t1 == t2)
            
            C = nodes[s2]
            D = nodes[t2]
            
            # 检查是否相交
            if segments_intersect(A, B, C, D):
                if shared_endpoint:
                    print(f"  ⚠️  边{i}({s1}->{t1}) × 边{j}({s2}->{t2}) 相交但共享端点！")
                else:
                    edge_crossings[i].append(j)
                    edge_crossings[j].append(i)
    
    # 计算 K 值
    k_value = 0
    for edge_idx in range(len(edges)):
        count = len(edge_crossings[edge_idx])
        if count > k_value:
            k_value = count
    
    print(f"\nK-value: {k_value}")
    
    # 显示交叉数最多的边
    max_edges = [i for i in range(len(edges)) if len(edge_crossings[i]) == k_value]
    if max_edges:
        print(f"\n有 {k_value} 个交叉的边:")
        for idx in max_edges[:3]:
            s, t = edges[idx]
            crossing_with = edge_crossings[idx]
            print(f"  边 #{idx}: {s}->{t}, 与边 {crossing_with[:5]} 相交")
    
    # 分布
    distribution = defaultdict(int)
    for i in range(len(edges)):
        count = len(edge_crossings[i])
        distribution[count] += 1
    
    print(f"\n交叉分布:")
    for count in sorted(distribution.keys()):
        print(f"  {distribution[count]} 条边有 {count} 个交叉")
    
    return k_value

# 分析官方答案
k_sol = analyze_detailed(
    "live-2025-example-instances/sol-15-nodes-5-planar.json",
    "官方标准答案 (应该是 K=5)"
)

# 分析我们的结果
k_ours = analyze_detailed(
    "results/29-01-22/15-nodes-ours.json",
    "我们的结果 (我们计算出 K=8)"
)

print(f"\n{'='*70}")
print(f"对比结果")
print(f"{'='*70}")
print(f"官方答案: K = {k_sol}")
print(f"我们的结果: K = {k_ours}")

if k_sol == 5:
    print(f"\n✓ 官方答案确认是 K=5")
    if k_ours > k_sol:
        print(f"❌ 我们的结果 K={k_ours} 比官方答案差！")
        print(f"   需要优化算法！")
    elif k_ours == k_sol:
        print(f"✓ 我们达到了官方答案的水平！")
    else:
        print(f"⚠️  我们的结果 K={k_ours} 比官方答案好？这不太可能...")
        print(f"   可能我们的计算方法有误！")
