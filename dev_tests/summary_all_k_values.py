#!/usr/bin/env python3
"""汇总所有测试文件的 K 值"""

import json
from collections import defaultdict

def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def segments_intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def get_k_value(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
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
    
    return max(edge_crossings.values()) if edge_crossings else 0

print("="*70)
print("所有文件的 K 值汇总")
print("="*70)

files = [
    ("15-node-others.json", 10, "官方说10"),
    ("15-node-o2.json", 11, "官方说11"),
    ("15-node-o3.json", 8, "官方说8"),
    ("15-nodes-ours.json", 16, "官方说16?"),
]

results = []
for filename, expected, note in files:
    filepath = f"results/29-01-22/{filename}"
    try:
        k = get_k_value(filepath)
        match = "✓" if k == expected else "✗"
        results.append((filename, k, expected, match, note))
        print(f"{filename:25} K={k:2d}  期望={expected:2d}  {match}  ({note})")
    except Exception as e:
        print(f"{filename:25} ERROR: {e}")

print("="*70)
print("\n关键发现:")
print("-" * 70)

our_k = next((k for f, k, _, _, _ in results if "ours" in f), None)
if our_k:
    print(f"我们的结果: K = {our_k}")
    print(f"官方声称: K = 16")
    print(f"\n这不匹配！")
    print(f"\n可能原因:")
    print(f"  1. 你上传的不是 15-nodes-ours.json 文件")
    print(f"  2. 文件在上传后被修改")
    print(f"  3. 官方评分系统有bug")
    print(f"  4. 我们的计算方法有误")
    
print("\n验证我们的计算:")
print("-" * 70)
matching = [f for f, k, e, m, _ in results if m == "✓"]
print(f"匹配官方评分的文件数: {len(matching)}/{len(results)}")
for f, k, e, m, n in results:
    if m == "✓":
        print(f"  ✓ {f}: K={k}")

if len(matching) >= 2:
    print(f"\n结论: 我们的计算方法是正确的！")
    print(f"       (至少 {len(matching)} 个文件的 K 值与官方评分匹配)")
