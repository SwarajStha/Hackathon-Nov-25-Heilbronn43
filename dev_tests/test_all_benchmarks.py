"""
完整的评分测试 - 测试所有数据集
使用新的初始化策略并保存结果
"""
import json
import time
from pathlib import Path
import sys
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from LCNv1.core.geometry import Point, GeometryCore
from LCNv1.core.k_plane_cost import KPlaneCost
from LCNv1.api import LCNSolver
from LCNv1.initialization import FMMEInitializer, PlanarizationInitializer
from LCNv1.strategies.enhanced import EnhancedSolverStrategy


def count_crossings(nodes_dict, edges):
    """
    计算交叉数（按题目要求的K-Plane定义）
    
    Returns:
        tuple: (total_crossings, k)
            - total_crossings: 总交叉数（所有边对的交叉总数）
            - k: K值（任一边的最大交叉数）- 这是K-Plane的定义指标
    """
    # 为每条边计数交叉数
    edge_list = list(edges)
    edge_crossing_counts = [0] * len(edge_list)
    
    for i in range(len(edge_list)):
        for j in range(i + 1, len(edge_list)):
            e1 = edge_list[i]
            e2 = edge_list[j]
            
            # 跳过共享端点的边
            if (e1['source'] == e2['source'] or e1['source'] == e2['target'] or
                e1['target'] == e2['source'] or e1['target'] == e2['target']):
                continue
            
            p1 = Point(nodes_dict[e1['source']]['x'], nodes_dict[e1['source']]['y'])
            p2 = Point(nodes_dict[e1['target']]['x'], nodes_dict[e1['target']]['y'])
            p3 = Point(nodes_dict[e2['source']]['x'], nodes_dict[e2['source']]['y'])
            p4 = Point(nodes_dict[e2['target']]['x'], nodes_dict[e2['target']]['y'])
            
            if GeometryCore.segments_intersect(p1, p2, p3, p4):
                # 成对计数：两条边都增加交叉数
                edge_crossing_counts[i] += 1
                edge_crossing_counts[j] += 1
    
    # 计算总交叉数（每个交叉被计数两次，所以除以2）
    total_crossings = sum(edge_crossing_counts) // 2
    
    # 计算K值（任一边的最大交叉数）
    k = max(edge_crossing_counts) if edge_crossing_counts else 0
    
    return total_crossings, k


def save_solution_json(solver_or_strategy, instance_name_or_dir, output_dir_or_base=None, timestamp=None):
    """
    保存求解结果为JSON格式（兼容新旧接口）
    
    新接口:
        save_solution_json(strategy, output_dir, base_name, timestamp)
    旧接口:
        save_solution_json(solver, instance_name, output_dir)
    
    Returns:
        str: 保存的文件路径
    """
    # 判断使用哪个接口
    if timestamp is not None:
        # 新接口：(strategy, output_dir, base_name, timestamp)
        strategy = solver_or_strategy
        output_dir = Path(instance_name_or_dir)
        base_name = output_dir_or_base
        
        output_name = f"{base_name}-ours.json"
        output_path = output_dir / output_name
        
        # 从strategy导出
        strategy.export_to_json(str(output_path))
        
    else:
        # 旧接口：(solver, instance_name, output_dir)
        solver = solver_or_strategy
        instance_name = instance_name_or_dir
        output_dir = Path(output_dir_or_base)
        
        # 构建输出文件名：15-nodes.json -> 15-nodes-ours.json
        base_name = instance_name.replace('.json', '')
        output_name = f"{base_name}-ours.json"
        output_path = output_dir / output_name
        
        # 获取当前解
        result_data = {
            "nodes": [],
            "edges": []
        }
        
        # 从solver获取节点位置
        if hasattr(solver, '_graph_data') and hasattr(solver, '_grid_state'):
            graph = solver._graph_data
            grid = solver._grid_state
            
            # 添加节点
            for node_id in range(graph.num_nodes):
                pos = grid.get_position(node_id)
                result_data["nodes"].append({
                    "id": node_id,
                    "x": pos.x,
                    "y": pos.y
                })
            
            # 添加边
            for edge in graph.edges:
                result_data["edges"].append({
                    "source": edge[0],
                    "target": edge[1]
                })
        
        # 保存文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
    
    return str(output_path)
    
    return str(output_path)


def test_instance(instance_name, solution_name, iterations=5000, use_init_strategy=True):
    """测试单个实例"""
    base_dir = Path(__file__).parent / 'live-2025-example-instances'
    instance_file = base_dir / instance_name
    solution_file = base_dir / solution_name
    
    # 创建输出目录：results/dd-HH-MM/
    now = datetime.now()
    timestamp = now.strftime("%d-%H-%M")
    output_dir = Path(__file__).parent / 'results' / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    with open(instance_file, 'r') as f:
        instance = json.load(f)
    
    with open(solution_file, 'r') as f:
        solution = json.load(f)
    
    # 计算标准解答的K值和总交叉数
    sol_nodes = {n['id']: n for n in solution['nodes']}
    sol_crossings, sol_k = count_crossings(sol_nodes, solution['edges'])
    
    # 运行求解器
    print(f"\n{'='*70}")
    print(f"测试: {instance_name}")
    print(f"  节点数: {len(instance['nodes'])}, 边数: {len(instance['edges'])}")
    print(f"  标准解答: K={sol_k}, 总交叉数={sol_crossings}")
    
    if use_init_strategy:
        print(f"  使用增强策略: FMME初始化 + K-Plane成本函数")
        print(f"{'='*70}")
        
        # 创建初始化策略
        init_strategy = FMMEInitializer(
            spring_iterations=100  # NetworkX spring layout 迭代次数
        )
        
        # 创建K-Plane成本函数（优先优化K值）
        cost_func = KPlaneCost(
            w_k=10000.0,      # K值权重（非常大）
            w_cross=100.0,    # 总交叉数权重
            w_len=1.0         # 边长度权重
        )
        
        # 创建增强求解器（使用统一接口）
        strategy = EnhancedSolverStrategy(
            w_cross=100.0,
            w_len=1.0,
            power=2,
            init_strategy=init_strategy,
            cost_function=cost_func  # 使用K-Plane成本函数
        )
        
        # 加载JSON
        strategy.load_from_json(str(instance_file))
        
        # 运行求解
        start = time.time()
        result = strategy.solve(iterations=iterations)
        elapsed = time.time() - start
        
        # 获取交叉数
        our_crossings = result['total_crossings']
        k_planes = result['k']
        
        # 保存结果JSON
        base_name = instance_name.replace('.json', '')
        output_file = save_solution_json(
            strategy,
            output_dir,
            base_name,
            timestamp
        )
        
        diff_crossings = our_crossings - sol_crossings
        diff_k = k_planes - sol_k
        status = "✅ 超越!" if diff_crossings < 0 else ("✨ 相等!" if diff_crossings == 0 else "⚠️  待改进")
        k_status = "✅" if diff_k <= 0 else "⚠️"
        
        print(f"\n结果:")
        print(f"  我们的结果: K={k_planes}, 总交叉数={our_crossings}")
        print(f"  标准解答:   K={sol_k}, 总交叉数={sol_crossings}")
        print(f"  K值对比: {diff_k:+d} {k_status}")
        print(f"  交叉数对比: {diff_crossings:+d} {status}")
        print(f"  用时: {elapsed:.2f}s")
        print(f"  结果已保存: {output_file}")
        
        return {
            'instance': instance_name,
            'nodes': len(instance['nodes']),
            'edges': len(instance['edges']),
            'standard_crossings': sol_crossings,
            'standard_k': sol_k,
            'our_crossings': our_crossings,
            'our_k': k_planes,
            'diff_crossings': diff_crossings,
            'diff_k': diff_k,
            'time': elapsed,
            'output_file': output_file
        }
    else:
        print(f"  使用默认 Numba 策略")
        print(f"{'='*70}")
        
        # 使用默认numba策略
        solver = LCNSolver(strategy='numba')
        solver.load_from_json(str(instance_file))
        
        start = time.time()
        opt_result = solver.optimize(iterations=iterations)
        elapsed = time.time() - start
        
        # 保存结果JSON
        base_name = instance_name.replace('.json', '')
        output_file = save_solution_json(
            solver._strategy,
            output_dir,
            base_name,
            timestamp
        )
        
        diff = opt_result.total_crossings - sol_crossings
        diff_k = opt_result.k - sol_k
        status = "✅ 超越!" if diff < 0 else ("✨ 相等!" if diff == 0 else "⚠️  待改进")
        k_status = "✅" if diff_k <= 0 else "⚠️"
        
        print(f"\n结果:")
        print(f"  我们的结果: K={opt_result.k}, 总交叉数={opt_result.total_crossings}")
        print(f"  标准解答:   K={sol_k}, 总交叉数={sol_crossings}")
        print(f"  K值对比: {diff_k:+d} {k_status}")
        print(f"  交叉数对比: {diff:+d} {status}")
        print(f"  改进: {opt_result.initial_crossings} → {opt_result.total_crossings}")
        print(f"  用时: {elapsed:.2f}s")
        print(f"  结果已保存: {output_file}")
        
        return {
            'instance': instance_name,
            'nodes': len(instance['nodes']),
            'edges': len(instance['edges']),
            'standard_crossings': sol_crossings,
            'standard_k': sol_k,
            'our_crossings': opt_result.total_crossings,
            'our_k': opt_result.k,
            'diff_crossings': diff,
            'diff_k': diff_k,
            'initial': opt_result.initial_crossings,
            'time': elapsed,
            'output_file': output_file
        }
    
    return result


def main():
    """运行所有测试"""
    # 只测试15和70节点
    test_cases = [
        ('15-nodes.json', 'sol-15-nodes-5-planar.json', 5000),
        ('70-nodes.json', 'sol-70-nodes-16-planar.json', 10000),
    ]
    
    results = []
    
    print(f"\n{'='*70}")
    print(f"开始测试 - 使用 FMME 初始化 + K-Plane成本函数")
    print(f"{'='*70}")
    
    for instance, solution, iters in test_cases:
        try:
            result = test_instance(instance, solution, iters, use_init_strategy=True)
            results.append(result)
        except Exception as e:
            print(f"❌ 错误: {instance} - {e}")
            import traceback
            traceback.print_exc()
    
    # 总结
    print(f"\n{'='*70}")
    print(f"总结报告")
    print(f"{'='*70}\n")
    print(f"{'实例':<20} {'节点':<8} {'边':<8} {'标准':<8} {'我们':<8} {'差距':<8} {'状态':<8}")
    print(f"{'-'*70}")
    
    better = 0
    equal = 0
    worse = 0
    
    for r in results:
        if r['diff'] < 0:
            status = '✅'
            better += 1
        elif r['diff'] == 0:
            status = '✨'
            equal += 1
        else:
            status = '⚠️'
            worse += 1
        
        print(f"{r['instance']:<20} {r['nodes']:<8} {r['edges']:<8} "
              f"{r['standard']:<8} {r['ours']:<8} {r['diff']:+<8} {status:<8}")
    
    print(f"{'-'*70}")
    print(f"\n统计:")
    print(f"  超越标准解答: {better}/{len(results)}")
    print(f"  等于标准解答: {equal}/{len(results)}")
    print(f"  低于标准解答: {worse}/{len(results)}")
    print(f"  成功率: {(better + equal) / len(results) * 100:.1f}%")
    
    # 保存详细结果
    output_file = Path(__file__).parent / 'benchmark_results_enhanced.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n详细结果已保存: {output_file}")
    
    # 显示所有保存的结果文件
    if results and 'output_file' in results[0]:
        print(f"\n💾 结果文件位置:")
        for r in results:
            print(f"  {r['instance']:<20} -> {r['output_file']}")


if __name__ == '__main__':
    main()
    
    # 显示所有保存的结果文件
    if results and 'output_file' in results[0]:
        print(f"\n💾 结果文件位置:")
        for r in results:
            print(f"  {r['instance']:<20} -> {r['output_file']}")
        json.dump(results, f, indent=2)
    print(f"\n详细结果已保存: {output_file}")


if __name__ == '__main__':
    main()
