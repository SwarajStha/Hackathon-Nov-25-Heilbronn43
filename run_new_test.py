#!/usr/bin/env python3
"""
直接运行测试：用新编译的代码重新优化225-nodes
期待：违规数应该持续减少直到0
"""
import json
import subprocess
import sys

# 创建Python脚本字符串
test_script = '''
import sys
sys.path.insert(0, "src")
import planar_cuda
import json

# 加载225-nodes.json
with open("live-2025-example-instances/225-nodes.json") as f:
    data = json.load(f)

nodes_x = [n["x"] for n in data["nodes"]]
nodes_y = [n["y"] for n in data["nodes"]]
edges = [[e["source"], e["target"]] for e in data["edges"]]

print("初始状态:")
print(f"  节点数: {len(nodes_x)}")
print(f"  边数: {len(edges)}")

# 创建solver
solver = planar_cuda.PlanarGraphSolver(nodes_x, nodes_y, edges)
k_before = solver.calculate_k_value()
print(f"  K值: {k_before}")

# 运行SA优化（瓶颈模式，新的违规修复策略）
print()
print("运行SA优化 (30000迭代, p=3, 新违规修复策略)...")
solver.run_sa_bottleneck(
    iterations=30000,
    start_temp=100.0,
    cooling_rate=0.97,
    power=3
)

k_after = solver.calculate_k_value()
print(f"\\n优化后 K值: {k_before} → {k_after} (Δ={k_after - k_before})")

# 获取结果
result_x, result_y = solver.get_coordinates()

# 保存结果
import datetime
timestamp = datetime.datetime.now().strftime("%H-%M-%S")
output = {
    "nodes": [{"id": i, "x": x, "y": y} for i, (x, y) in enumerate(zip(result_x, result_y))],
    "edges": data["edges"]
}

import os
os.makedirs("results", exist_ok=True)
os.makedirs(f"results/{timestamp}", exist_ok=True)
outfile = f"results/{timestamp}/225-nodes-NEW-k{k_after}.json"
with open(outfile, "w") as f:
    json.dump(output, f, indent=2)

print(f"\\n结果已保存到: {outfile}")
print("\\n运行 simple_check.py 检查违规...")
'''

# 写入临时文件
with open("temp_test_new.py", "w", encoding="utf-8") as f:
    f.write(test_script)

# 运行
print("=" * 80)
print("测试新的违规修复策略")
print("=" * 80)
print()

result = subprocess.run(
    ["heilbron-43/Scripts/python.exe", "temp_test_new.py"],
    capture_output=False,
    text=True
)

sys.exit(result.returncode)
