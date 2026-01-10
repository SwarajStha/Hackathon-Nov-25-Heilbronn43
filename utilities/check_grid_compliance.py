#!/usr/bin/env python3
"""检查文件是否符合Grid granularity约束"""
import json
import sys

def check_grid_compliance(filepath, grid_size):
    """检查文件是否符合grid约束"""
    with open(filepath) as f:
        data = json.load(f)
    
    violations = []
    for n in data['nodes']:
        if n['x'] % grid_size != 0 or n['y'] % grid_size != 0:
            violations.append((n['id'], n['x'], n['y']))
    
    return data, violations

# 检查我们的结果
print("="*70)
print("检查Grid Granularity约束")
print("="*70)
print()

filepath = sys.argv[1] if len(sys.argv) > 1 else 'results/07-06-39/225-nodes-NEW-k303.json'
grid_size = int(sys.argv[2]) if len(sys.argv) > 2 else 4

print(f"文件: {filepath}")
print(f"Grid granularity: {grid_size}")
print()

data, violations = check_grid_compliance(filepath, grid_size)

print(f"节点数: {len(data['nodes'])}")

if violations:
    print(f"❌ 违反grid约束的节点: {len(violations)}/{len(data['nodes'])}")
    print()
    print("前10个违规节点:")
    for nid, x, y in violations[:10]:
        print(f"  节点{nid:3d}: ({x:5d}, {y:5d}) - x%{grid_size}={x%grid_size}, y%{grid_size}={y%grid_size}")
    
    if len(violations) > 10:
        print(f"  ... 还有{len(violations)-10}个违规节点")
else:
    print(f"✅ 所有节点坐标都符合grid={grid_size}的要求")

# 检查原始输入文件
print()
print("="*70)
print("对比: 原始输入文件")
print("="*70)
original_path = 'live-2025-example-instances/225-nodes.json'
try:
    orig_data, orig_violations = check_grid_compliance(original_path, grid_size)
    
    if orig_violations:
        print(f"原始输入也有 {len(orig_violations)} 个违规节点")
    else:
        print(f"✅ 原始输入符合grid={grid_size}")
except FileNotFoundError:
    print(f"文件不存在: {original_path}")

# 检查官方标准答案
print()
print("="*70)
print("对比: 官方标准答案 (sol-625-nodes)")
print("="*70)
official_path = 'live-2025-example-instances/sol-625-nodes-5-planar.json'
try:
    official_data, official_violations = check_grid_compliance(official_path, grid_size)
    
    if official_violations:
        print(f"官方文件有 {len(official_violations)} 个违规节点")
    else:
        print(f"✅ 官方文件符合grid={grid_size}")
except FileNotFoundError:
    print(f"文件不存在: {official_path}")
