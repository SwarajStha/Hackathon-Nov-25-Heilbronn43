#!/usr/bin/env python3
"""测试新的违规修复策略"""
import json
import sys
import os
from pathlib import Path
from collections import Counter

# 添加CUDA DLL路径
os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin')
sys.path.insert(0, 'build_artifacts')
sys.path.insert(0, 'src')

import planar_cuda

def count_violations(nodes):
    """计算违规数"""
    x = [n['x'] for n in nodes]
    y = [n['y'] for n in nodes]
    
    x_viol = len([c for c in Counter(x).values() if c >= 3])
    y_viol = len([c for c in Counter(y).values() if c >= 3])
    
    return x_viol, y_viol, x_viol + y_viol

print("=" * 80)
print("测试新的违规修复策略 (before/after comparison)")
print("=" * 80)
print()

# 加载225-nodes.json
input_file = "live-2025-example-instances/225-nodes.json"
print(f"输入文件: {input_file}")

with open(input_file) as f:
    data = json.load(f)

nodes_x = [n['x'] for n in data['nodes']]
nodes_y = [n['y'] for n in data['nodes']]
edges = [[e['source'], e['target']] for e in data['edges']]

# 检查初始违规
x_v, y_v, total_v = count_violations(data['nodes'])
print(f"初始违规: 垂直={x_v}, 水平={y_v}, 总计={total_v}")
print()

# 创建solver
solver = planar_cuda.PlanarSolver(nodes_x, nodes_y, edges)
k_before = solver.calculate_k_value()
print(f"初始 K值: {k_before}")
print()

print("运行SA优化 (30000迭代, p=3, 新策略)...")
print("  策略: 增加违规→拒绝, 减少违规→-100B奖励, 违规不变→正常优化")
print()

import time
start = time.time()

solver.run_sa_optimization(
    iterations=30000,
    start_temp=100.0,
    cooling_rate=0.97,
    cost_function="bottleneck_p3"
)

elapsed = time.time() - start

k_after = solver.calculate_k_value()
print(f"\n优化完成 (用时 {elapsed:.1f}秒)")
print(f"K值: {k_before} → {k_after} (Δ={k_after - k_before})")

# 获取结果
result_x, result_y = solver.get_coordinates()
result_nodes = [{'id': i, 'x': x, 'y': y} for i, (x, y) in enumerate(zip(result_x, result_y))]

# 检查违规
x_v2, y_v2, total_v2 = count_violations(result_nodes)
print(f"\n违规检查:")
print(f"  初始: {total_v} (垂直={x_v}, 水平={y_v})")
print(f"  优化后: {total_v2} (垂直={x_v2}, 水平={y_v2})")
print(f"  改善: {total_v - total_v2} 个违规被修复 ({(total_v - total_v2)/total_v*100:.1f}%)")

print()
if total_v2 == 0:
    print("✅✅✅ 完全成功！所有违规已修复！")
elif total_v2 < total_v:
    print(f"⚠️ 部分成功：违规从 {total_v} 减少到 {total_v2}")
    print(f"   建议：增加迭代次数或运行多次选最佳结果")
else:
    print(f"❌ 未改善：违规数量未减少")

# 保存结果
import datetime
timestamp = datetime.datetime.now().strftime("%H-%M-%S")
os.makedirs("results", exist_ok=True)
os.makedirs(f"results/{timestamp}", exist_ok=True)

output = {
    "nodes": result_nodes,
    "edges": data['edges']
}

outfile = f"results/{timestamp}/225-nodes-NEW-k{k_after}.json"
with open(outfile, "w") as f:
    json.dump(output, f, indent=2)

print(f"\n结果已保存到: {outfile}")
