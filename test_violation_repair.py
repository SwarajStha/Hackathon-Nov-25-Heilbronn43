#!/usr/bin/env python3
"""
测试新的违规修复策略
策略：优先减少违规，拒绝增加违规，违规不变时正常优化
"""
import json
import sys
from pathlib import Path
from collections import Counter

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

import planar_cuda

def count_violations(nodes):
    """计算垂直+水平共线违规数"""
    x_coords = [n['x'] for n in nodes]
    y_coords = [n['y'] for n in nodes]
    
    x_counter = Counter(x_coords)
    y_counter = Counter(y_coords)
    
    x_viol = sum(1 for count in x_counter.values() if count >= 3)
    y_viol = sum(1 for count in y_counter.values() if count >= 3)
    
    return x_viol, y_viol, x_viol + y_viol

def test_repair_from_violated_input():
    """测试从违规输入文件开始能否修复"""
    
    print("=" * 80)
    print("测试：从违规输入修复到无违规状态")
    print("=" * 80)
    print()
    
    # 使用已知有违规的225-nodes.json
    input_file = "live-2025-example-instances/225-nodes.json"
    
    print(f"输入文件: {input_file}")
    
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    nodes = data['nodes']
    edges = data['edges']
    
    # 检查初始违规
    x_viol_before, y_viol_before, total_viol_before = count_violations(nodes)
    print(f"初始违规: 垂直={x_viol_before}, 水平={y_viol_before}, 总计={total_viol_before}")
    
    if total_viol_before == 0:
        print("⚠️ 输入文件没有违规，无法测试修复功能！")
        return
    
    print()
    print("新策略:")
    print("  - 增加违规的移动 → 完全拒绝 (LLONG_MAX)")
    print("  - 减少违规的移动 → 强烈奖励 (-100B per violation)")
    print("  - 违规不变的移动 → 正常优化交叉数")
    print()
    print("运行5次优化，看能否修复违规...")
    print()
    
    best_result = None
    best_violations = float('inf')
    
    for run in range(1, 6):
        print(f"运行 {run}/5...")
        
        # 重新加载数据
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        nodes_x = [n['x'] for n in data['nodes']]
        nodes_y = [n['y'] for n in data['nodes']]
        edge_list = [[e['source'], e['target']] for e in data['edges']]
        
        # 创建solver并优化
        solver = planar_cuda.PlanarGraphSolver(nodes_x, nodes_y, edge_list)
        
        # 使用瓶颈模式 (p=3)
        k_before = solver.calculate_k_value()
        solver.run_sa_bottleneck(
            iterations=20000,  # 增加迭代次数以有更多机会修复
            start_temp=100.0,
            cooling_rate=0.97,
            power=3
        )
        k_after = solver.calculate_k_value()
        
        # 获取结果
        result_x, result_y = solver.get_coordinates()
        result_nodes = [{'id': i, 'x': x, 'y': y} for i, (x, y) in enumerate(zip(result_x, result_y))]
        
        # 检查违规
        x_viol, y_viol, total_viol = count_violations(result_nodes)
        
        print(f"  K: {k_before} → {k_after} (Δ={k_after - k_before})")
        print(f"  违规: {total_viol_before} → {total_viol} (Δ={total_viol - total_viol_before})")
        
        if total_viol == 0:
            print(f"  ✅ 成功修复所有违规！")
        elif total_viol < best_violations:
            best_violations = total_viol
            best_result = {
                'nodes': result_nodes,
                'edges': data['edges'],
                'k': k_after,
                'violations': total_viol
            }
        
        print()
    
    print("=" * 80)
    print("测试结果总结:")
    print("=" * 80)
    print(f"初始违规数: {total_viol_before}")
    print(f"最佳结果违规数: {best_violations}")
    print(f"违规减少: {total_viol_before - best_violations} ({(total_viol_before - best_violations) / total_viol_before * 100:.1f}%)")
    
    if best_violations == 0:
        print()
        print("✅✅✅ 完全成功！所有违规已修复！")
        
        # 保存结果
        output_file = f"results/repaired-225-nodes-k{best_result['k']}.json"
        Path("results").mkdir(exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(best_result, f, indent=2)
        print(f"已保存到: {output_file}")
    elif best_violations < total_viol_before:
        print()
        print(f"⚠️ 部分成功：违规从 {total_viol_before} 减少到 {best_violations}")
        print(f"   可能需要:")
        print(f"   - 更多迭代次数")
        print(f"   - 更高的初始温度")
        print(f"   - 多次运行并选择最佳结果")
    else:
        print()
        print(f"❌ 失败：无法减少违规数")
        print(f"   这表明约束检查可能有问题")

if __name__ == "__main__":
    test_repair_from_violated_input()
