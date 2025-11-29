#!/usr/bin/env python3
"""
检查所有可能的违规情况：
1. 重复坐标
2. 节点在边的内部
3. 边的重叠
"""

import json
from collections import defaultdict

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def point_on_segment(p, a, b, epsilon=1e-9):
    """检查点p是否在线段ab上"""
    # 检查共线
    cross = (p[1] - a[1]) * (b[0] - a[0]) - (p[0] - a[0]) * (b[1] - a[1])
    if abs(cross) > epsilon:
        return False
    
    # 检查p是否在a和b之间（不包括端点）
    if p[0] != b[0]:
        t = (p[0] - a[0]) / (b[0] - a[0]) if b[0] != a[0] else 0
    else:
        t = (p[1] - a[1]) / (b[1] - a[1]) if b[1] != a[1] else 0
    
    return 0 < t < 1  # 严格在内部

def segments_overlap(a1, b1, a2, b2, epsilon=1e-9):
    """检查两条线段是否重叠（部分或完全）"""
    # 检查是否平行/共线
    dx1, dy1 = b1[0] - a1[0], b1[1] - a1[1]
    dx2, dy2 = b2[0] - a2[0], b2[1] - a2[1]
    
    cross = dx1 * dy2 - dy1 * dx2
    if abs(cross) > epsilon:
        return False  # 不平行，不可能重叠
    
    # 检查是否共线
    dx3, dy3 = a2[0] - a1[0], a2[1] - a1[1]
    cross2 = dx1 * dy3 - dy1 * dx3
    if abs(cross2) > epsilon:
        return False  # 平行但不共线
    
    # 共线，检查是否有重叠部分
    # 将两条线段投影到主轴上
    if abs(dx1) > abs(dy1):
        # 使用x轴
        seg1_min, seg1_max = min(a1[0], b1[0]), max(a1[0], b1[0])
        seg2_min, seg2_max = min(a2[0], b2[0]), max(a2[0], b2[0])
    else:
        # 使用y轴
        seg1_min, seg1_max = min(a1[1], b1[1]), max(a1[1], b1[1])
        seg2_min, seg2_max = min(a2[1], b2[1]), max(a2[1], b2[1])
    
    # 检查区间是否有重叠（不包括端点接触）
    return seg2_min < seg1_max and seg1_min < seg2_max

def check_violations(filepath, name):
    """检查所有违规情况"""
    print(f"\n{'='*70}")
    print(f"检查: {name}")
    print(f"文件: {filepath}")
    print(f"{'='*70}")
    
    data = load_json(filepath)
    nodes = {n['id']: (n['x'], n['y']) for n in data['nodes']}
    edges = [(e['source'], e['target']) for e in data['edges']]
    
    violations = {
        'duplicate_coords': [],
        'node_on_edge': [],
        'overlapping_edges': []
    }
    
    # 1. 检查重复坐标
    coords = {}
    for nid, coord in nodes.items():
        if coord in coords.values():
            existing = [k for k, v in coords.items() if v == coord]
            violations['duplicate_coords'].append((coord, existing + [nid]))
        coords[nid] = coord
    
    # 2. 检查节点是否在边的内部
    for node_id, node_pos in nodes.items():
        for edge_idx, (s, t) in enumerate(edges):
            if node_id == s or node_id == t:
                continue  # 跳过端点
            
            edge_start = nodes[s]
            edge_end = nodes[t]
            
            if point_on_segment(node_pos, edge_start, edge_end):
                violations['node_on_edge'].append({
                    'node': node_id,
                    'edge': edge_idx,
                    'edge_endpoints': (s, t)
                })
    
    # 3. 检查边的重叠
    for i, (s1, t1) in enumerate(edges):
        a1 = nodes[s1]
        b1 = nodes[t1]
        
        for j, (s2, t2) in enumerate(edges):
            if i >= j:
                continue
            
            # 跳过共享端点的边
            if s1 == s2 or s1 == t2 or t1 == s2 or t1 == t2:
                continue
            
            a2 = nodes[s2]
            b2 = nodes[t2]
            
            if segments_overlap(a1, b1, a2, b2):
                violations['overlapping_edges'].append({
                    'edge1': i,
                    'edge1_endpoints': (s1, t1),
                    'edge2': j,
                    'edge2_endpoints': (s2, t2)
                })
    
    # 报告
    total_violations = (len(violations['duplicate_coords']) + 
                       len(violations['node_on_edge']) + 
                       len(violations['overlapping_edges']))
    
    print(f"\n节点数: {len(nodes)}")
    print(f"边数: {len(edges)}")
    print(f"\n违规总数: {total_violations}")
    
    if violations['duplicate_coords']:
        print(f"\n❌ 重复坐标: {len(violations['duplicate_coords'])}组")
        for coord, node_ids in violations['duplicate_coords']:
            print(f"  坐标{coord}: 节点{node_ids}")
    else:
        print(f"\n✅ 无重复坐标")
    
    if violations['node_on_edge']:
        print(f"\n❌ 节点在边内部: {len(violations['node_on_edge'])}处")
        for v in violations['node_on_edge'][:5]:  # 只显示前5个
            print(f"  节点{v['node']} 在边{v['edge']}({v['edge_endpoints'][0]}->{v['edge_endpoints'][1]})内部")
        if len(violations['node_on_edge']) > 5:
            print(f"  ... 还有{len(violations['node_on_edge'])-5}处")
    else:
        print(f"\n✅ 无节点在边内部")
    
    if violations['overlapping_edges']:
        print(f"\n❌ 边重叠: {len(violations['overlapping_edges'])}对")
        for v in violations['overlapping_edges'][:5]:
            print(f"  边{v['edge1']}({v['edge1_endpoints'][0]}->{v['edge1_endpoints'][1]}) × "
                  f"边{v['edge2']}({v['edge2_endpoints'][0]}->{v['edge2_endpoints'][1]})")
        if len(violations['overlapping_edges']) > 5:
            print(f"  ... 还有{len(violations['overlapping_edges'])-5}对")
    else:
        print(f"\n✅ 无边重叠")
    
    return violations, total_violations

# 检查所有文件
files = [
    ("live-2025-example-instances/sol-15-nodes-5-planar.json", "官方标准答案 K=5"),
    ("results/29-02-18/15-nodes-ours.json", "我们的旧结果 K=4 (有违规)"),
    ("results/11-29-02/15-nodes-ours-v2.json", "我们的新结果 K=5 (带约束)")
]

print("="*70)
print("检查所有可能的违规情况")
print("="*70)

for filepath, name in files:
    violations, total = check_violations(filepath, name)
    
print("\n" + "="*70)
print("结论")
print("="*70)
print("如果我们的结果有违规（节点在边内部、边重叠等），")
print("官方评分系统可能会计算这些违规，导致crossing数值不同。")
