#!/usr/bin/env python3
"""
对比CUDA计算和官方几何计算的差异
找出为什么CUDA报告K=303但实际是456
"""
import json
import sys
import os

# 官方几何方法
def ccw(A, B, C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

def segments_intersect(A, B, C, D):
    # 共享端点不算
    if A == C or A == D or B == C or B == D:
        return False
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

# 加载CUDA
sys.path.insert(0, 'src')
os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin')
import planar_cuda

# 加载文件
filepath = sys.argv[1] if len(sys.argv) > 1 else 'results/07-06-39/225-nodes-NEW-k303.json'
print(f"对比文件: {filepath}")
print("="*70)
print()

with open(filepath) as f:
    data = json.load(f)

nodes = {n['id']: (n['x'], n['y']) for n in data['nodes']}
edges = [(e['source'], e['target']) for e in data['edges']]

# 方法1: CUDA计算
print("方法1: CUDA calculate_k_value()")
nodes_x = [n['x'] for n in data['nodes']]
nodes_y = [n['y'] for n in data['nodes']]
solver = planar_cuda.PlanarSolver(nodes_x, nodes_y, edges)
cuda_k = solver.calculate_k_value()
cuda_crossings = solver.get_edge_crossings()
print(f"  K值: {cuda_k}")
cuda_max_idx = cuda_crossings.index(max(cuda_crossings))
print(f"  最大边: 边{cuda_max_idx} ({edges[cuda_max_idx][0]}->{edges[cuda_max_idx][1]})")
print()

# 方法2: 官方几何
print("方法2: 官方几何方法")
official_crossings = []
for i, (src1, tgt1) in enumerate(edges):
    p1, p2 = nodes[src1], nodes[tgt1]
    count = sum(1 for j, (src2, tgt2) in enumerate(edges) 
                if i != j and segments_intersect(p1, p2, nodes[src2], nodes[tgt2]))
    official_crossings.append(count)
    if (i+1) % 100 == 0:
        print(f"  进度: {i+1}/{len(edges)}", end='\r')

official_k = max(official_crossings)
official_max_idx = official_crossings.index(official_k)
print(f"\n  K值: {official_k}")
print(f"  最大边: 边{official_max_idx} ({edges[official_max_idx][0]}->{edges[official_max_idx][1]})")
print()

# 对比
print("="*70)
print("对比分析:")
print("="*70)
print(f"CUDA K值:   {cuda_k}")
print(f"官方 K值:   {official_k}")
print(f"差异:       {cuda_k - official_k} ({abs(cuda_k - official_k) / official_k * 100:.1f}%)")
print()

# 找出差异最大的边
print("差异最大的边 (前10):")
diffs = [(i, official_crossings[i] - cuda_crossings[i], official_crossings[i], cuda_crossings[i]) 
         for i in range(len(edges))]
diffs.sort(key=lambda x: abs(x[1]), reverse=True)

for i, diff, official, cuda in diffs[:10]:
    src, tgt = edges[i]
    print(f"  边{i:4d} ({src:3d}->{tgt:3d}): 官方={official:3d}, CUDA={cuda:3d}, 差={diff:+4d}")

print()
print("可能原因:")
if cuda_k < official_k:
    print("  ❌ CUDA漏算了某些交叉")
    print("  可能是:")
    print("    - 浮点精度问题")
    print("    - 边界情况处理不当")
    print("    - 共享端点判断错误")
elif cuda_k > official_k:
    print("  ❌ CUDA多算了某些交叉")
    print("  可能是:")
    print("    - 重复计数")
    print("    - 共享端点也算作交叉")
else:
    print("  ✅ 完全一致！")
