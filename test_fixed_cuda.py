"""
测试修复后的CUDA K值计算
"""
import json
import planar_cuda

# 加载数据
with open('results/07-06-39/225-nodes-NEW-k303.json', 'r') as f:
    data = json.load(f)

nodes_x = [n['x'] for n in data['nodes']]
nodes_y = [n['y'] for n in data['nodes']]
edges = [(e['source'], e['target']) for e in data['edges']]

# 使用修复后的CUDA计算K值
k_cuda = planar_cuda.calculate_k_value(nodes_x, nodes_y, edges)

print(f"修复后的CUDA K值: {k_cuda}")
print(f"官方几何计算K值: 456")
print(f"差异: {k_cuda - 456}")
print()

if k_cuda == 456:
    print("✅ CUDA计算修复成功！精度问题已解决！")
else:
    print(f"❌ 仍有{abs(k_cuda - 456)}的差异")
