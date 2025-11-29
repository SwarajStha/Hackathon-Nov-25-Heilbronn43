"""
简单的评分测试 - 快速验证求解器
"""
import json
import sys
import os
from pathlib import Path

# 添加路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from LCNv1.core.geometry import Point, GeometryCore


def count_crossings(nodes_dict, edges):
    """计算交叉数"""
    crossings = 0
    edge_list = list(edges)
    
    for i in range(len(edge_list)):
        for j in range(i + 1, len(edge_list)):
            e1 = edge_list[i]
            e2 = edge_list[j]
            
            # 跳过共享端点
            if (e1['source'] == e2['source'] or e1['source'] == e2['target'] or
                e1['target'] == e2['source'] or e1['target'] == e2['target']):
                continue
            
            # 检查交叉
            p1 = Point(nodes_dict[e1['source']]['x'], nodes_dict[e1['source']]['y'])
            p2 = Point(nodes_dict[e1['target']]['x'], nodes_dict[e1['target']]['y'])
            p3 = Point(nodes_dict[e2['source']]['x'], nodes_dict[e2['source']]['y'])
            p4 = Point(nodes_dict[e2['target']]['x'], nodes_dict[e2['target']]['y'])
            
            if GeometryCore.segments_intersect(p1, p2, p3, p4):
                crossings += 1
    
    return crossings


def main():
    # 加载 15-nodes 测试
    instance_file = Path(__file__).parent / 'live-2025-example-instances' / '15-nodes.json'
    solution_file = Path(__file__).parent / 'live-2025-example-instances' / 'sol-15-nodes-5-planar.json'
    
    with open(instance_file, 'r') as f:
        instance = json.load(f)
    
    with open(solution_file, 'r') as f:
        solution = json.load(f)
    
    # 检查输入
    input_nodes = {n['id']: n for n in instance['nodes']}
    input_crossings = count_crossings(input_nodes, instance['edges'])
    
    # 检查解答
    sol_nodes = {n['id']: n for n in solution['nodes']}
    sol_crossings = count_crossings(sol_nodes, solution['edges'])
    
    print(f"15-nodes 测试:")
    print(f"  输入初始交叉数: {input_crossings}")
    print(f"  标准解答交叉数: {sol_crossings}")
    print(f"  节点数: {len(instance['nodes'])}")
    print(f"  边数: {len(instance['edges'])}")
    
    # 尝试使用求解器
    print(f"\n尝试加载求解器...")
    try:
        from LCNv1.api import LCNSolver
        print("✅ LCNSolver 导入成功")
        
        solver = LCNSolver(strategy='numba')
        print("✅ 求解器创建成功")
        
        solver.load_from_json(str(instance_file))
        print("✅ 数据加载成功")
        
        print("\n运行优化 (1000次迭代)...")
        result = solver.optimize(iterations=1000)
        
        print(f"\n结果:")
        print(f"  最终交叉数: {result.total_crossings}")
        print(f"  初始交叉数: {result.initial_crossings}")
        print(f"  改进: {result.initial_crossings - result.total_crossings}")
        print(f"  vs 标准解答 ({sol_crossings}): 差距 {result.total_crossings - sol_crossings}")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
