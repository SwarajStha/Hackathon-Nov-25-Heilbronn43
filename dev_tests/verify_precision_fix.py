"""
验证修复后的CUDA K值计算 - 精度问题修复
"""
import sys
import os
# 添加父目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from solver_framework import SolverRegistry
import json

# 加载数据
with open('results/07-06-39/225-nodes-NEW-k303.json', 'r') as f:
    data = json.load(f)

nodes = [(n['x'], n['y']) for n in data['nodes']]
edges = [(e['source'], e['target']) for e in data['edges']]

print("=" * 80)
print("CUDA精度修复验证 - 225节点文件")
print("=" * 80)
print(f"节点数: {len(nodes)}")
print(f"边数: {len(edges)}")
print(f"坐标范围: {max(max(x, y) for x, y in nodes)}")
print()

# 使用CUDA计算
try:
    solver = SolverRegistry.create_solver("cuda", nodes, edges)
    k_cuda = solver.calculate_k_value()
    
    print(f"CUDA计算K值: {k_cuda}")
    print(f"官方几何K值: 456")
    print(f"差异: {k_cuda - 456}")
    print()
    
    if k_cuda == 456:
        print("✅ 精度修复成功！")
        print("   修复方法: 避免d1*d2溢出，改用符号比较")
        print("   问题原因: 6位数坐标导致cross product乘积超过long long范围")
    else:
        print(f"❌ 仍有差异: {abs(k_cuda - 456)}")
        
except Exception as e:
    print(f"❌ CUDA计算失败: {e}")
    import traceback
    traceback.print_exc()
