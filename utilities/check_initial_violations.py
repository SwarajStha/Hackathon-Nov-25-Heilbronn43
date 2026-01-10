#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查初始文件是否有几何约束违反
"""
import json
import sys
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')


def check_collinear_nodes(nodes_x, nodes_y, nodes):
    """检查是否有三个或更多节点共线"""
    n = len(nodes)
    violations = []
    
    # 检查垂直线（相同 x）
    x_groups = {}
    for i, x in enumerate(nodes_x):
        if x not in x_groups:
            x_groups[x] = []
        x_groups[x].append(i)
    
    for x, node_ids in x_groups.items():
        if len(node_ids) >= 3:
            violations.append(f"垂直线 x={x}: {len(node_ids)} 个节点 {node_ids}")
    
    # 检查水平线（相同 y）
    y_groups = {}
    for i, y in enumerate(nodes_y):
        if y not in y_groups:
            y_groups[y] = []
        y_groups[y].append(i)
    
    for y, node_ids in y_groups.items():
        if len(node_ids) >= 3:
            violations.append(f"水平线 y={y}: {len(node_ids)} 个节点 {node_ids}")
    
    return violations


def check_collinear_edges(nodes_x, nodes_y, edges):
    """检查是否有边共享同一条线段（共线且重叠）"""
    violations = []
    
    for i in range(len(edges)):
        for j in range(i+1, len(edges)):
            s1, t1 = edges[i]
            s2, t2 = edges[j]
            
            # 跳过共享端点的边
            if s1 in (s2, t2) or t1 in (s2, t2):
                continue
            
            # 获取坐标
            p1x, p1y = nodes_x[s1], nodes_y[s1]
            p2x, p2y = nodes_x[t1], nodes_y[t1]
            p3x, p3y = nodes_x[s2], nodes_y[s2]
            p4x, p4y = nodes_x[t2], nodes_y[t2]
            
            # 检查是否共线
            cross1 = (p2x - p1x) * (p3y - p1y) - (p2y - p1y) * (p3x - p1x)
            cross2 = (p2x - p1x) * (p4y - p1y) - (p2y - p1y) * (p4x - p1x)
            
            if cross1 == 0 and cross2 == 0:
                # 共线，检查是否重叠
                dx = abs(p2x - p1x)
                dy = abs(p2y - p1y)
                
                if dx >= dy:
                    # 使用 x 轴
                    seg1_min, seg1_max = min(p1x, p2x), max(p1x, p2x)
                    seg2_min, seg2_max = min(p3x, p4x), max(p3x, p4x)
                else:
                    # 使用 y 轴
                    seg1_min, seg1_max = min(p1y, p2y), max(p1y, p2y)
                    seg2_min, seg2_max = min(p3y, p4y), max(p3y, p4y)
                
                overlap_start = max(seg1_min, seg2_min)
                overlap_end = min(seg1_max, seg2_max)
                
                if overlap_start < overlap_end:
                    violations.append(
                        f"边 {i} ({edges[i]}) 和边 {j} ({edges[j]}) 共线且重叠: "
                        f"({p1x},{p1y})-({p2x},{p2y}) vs ({p3x},{p3y})-({p4x},{p4y})"
                    )
    
    return violations


def main():
    file_path = r'results\06-18-34\15-nodes-cu-k1.json'
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    nodes = data['nodes']
    edges = data['edges']
    nodes_x = [n['x'] for n in nodes]
    nodes_y = [n['y'] for n in nodes]
    edge_pairs = [(e['source'], e['target']) for e in edges]
    
    print("="*80)
    print(f"检查文件: {file_path}")
    print("="*80)
    print(f"节点数: {len(nodes)}, 边数: {len(edges)}\n")
    
    # 统计坐标分布
    print("X 坐标分布:")
    x_counter = Counter(nodes_x)
    for x, count in sorted(x_counter.items()):
        marker = " ⚠️" if count > 2 else ""
        print(f"  x={x:3d}: {count} 个节点{marker}")
    
    print("\nY 坐标分布:")
    y_counter = Counter(nodes_y)
    for y, count in sorted(y_counter.items()):
        marker = " ⚠️" if count > 2 else ""
        print(f"  y={y:3d}: {count} 个节点{marker}")
    
    # 检查共线节点
    print("\n" + "="*80)
    print("检查: 三个或更多节点共线")
    print("="*80)
    node_violations = check_collinear_nodes(nodes_x, nodes_y, nodes)
    if node_violations:
        print("❌ 发现违反约束!")
        for v in node_violations:
            print(f"  {v}")
    else:
        print("✅ 没有发现三个或更多节点共线")
    
    # 检查共线边
    print("\n" + "="*80)
    print("检查: 边共享线段（共线且重叠）")
    print("="*80)
    edge_violations = check_collinear_edges(nodes_x, nodes_y, edge_pairs)
    if edge_violations:
        print("❌ 发现违反约束!")
        for v in edge_violations:
            print(f"  {v}")
    else:
        print("✅ 没有发现边共享线段")
    
    # 总结
    print("\n" + "="*80)
    print("总结")
    print("="*80)
    total_violations = len(node_violations) + len(edge_violations)
    if total_violations == 0:
        print("✅ 没有发现几何约束违反")
    else:
        print(f"❌ 共发现 {total_violations} 个约束违反:")
        print(f"   - 共线节点: {len(node_violations)} 个")
        print(f"   - 共线边: {len(edge_violations)} 个")


if __name__ == "__main__":
    main()
