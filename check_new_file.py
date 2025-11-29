#!/usr/bin/env python3
"""
快速检查新生成的文件是否有违规
"""
import json
from collections import Counter

# 检查新文件
filepath = 'results/06-55-34/225-nodes-cu-k482.json'
print(f"检查文件: {filepath}")
print("=" * 70)

with open(filepath, 'r') as f:
    data = json.load(f)

nodes = data['nodes']
x_coords = [n['x'] for n in nodes]
y_coords = [n['y'] for n in nodes]

print(f"节点数: {len(nodes)}")
print()

# 检查垂直共线 (相同 x)
x_counter = Counter(x_coords)
x_violations = [(x, count) for x, count in x_counter.items() if count >= 3]
x_violations.sort(key=lambda item: item[1], reverse=True)

# 检查水平共线 (相同 y)
y_counter = Counter(y_coords)
y_violations = [(y, count) for y, count in y_counter.items() if count >= 3]
y_violations.sort(key=lambda item: item[1], reverse=True)

print(f"垂直共线违规 (≥3节点同x): {len(x_violations)} 处")
if x_violations:
    print(f"  最严重的5处:")
    for x, count in x_violations[:5]:
        print(f"    x={x}: {count}个节点")

print()
print(f"水平共线违规 (≥3节点同y): {len(y_violations)} 处")
if y_violations:
    print(f"  最严重的5处:")
    for y, count in y_violations[:5]:
        print(f"    y={y}: {count}个节点")

print()
print("=" * 70)
total_violations = len(x_violations) + len(y_violations)
print(f"总违规数: {total_violations}")

if total_violations == 0:
    print("✅ 文件满足所有约束条件！")
else:
    print(f"❌ 文件有 {total_violations} 处违规！")
    print()
    print("这说明:")
    print("  1. 原始文件本身就有违规")
    print("  2. SA从违规状态开始，优先减少K值而非消除违规")
    print("  3. 需要实现「初始修复」或「绝对拒绝」策略")

# 检查重复坐标
coord_counter = Counter(zip(x_coords, y_coords))
duplicates = [(coord, count) for coord, count in coord_counter.items() if count > 1]
if duplicates:
    print(f"\n⚠️ 重复坐标: {len(duplicates)} 处")
    for coord, count in duplicates[:3]:
        print(f"  {coord}: {count}个节点")
