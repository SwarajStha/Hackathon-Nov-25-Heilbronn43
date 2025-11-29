"""
Benchmark Script - 评估求解器性能
比较当前求解器的结果与标准解答
"""
import json
import sys
import os
import time
from pathlib import Path

# 添加 src 路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from LCNv1.api import LCNSolver
from LCNv1.core.geometry import Point, segments_intersect


def count_crossings(nodes, edges):
    """计算交叉数量"""
    crossings = 0
    edge_list = list(edges)
    
    for i in range(len(edge_list)):
        for j in range(i + 1, len(edge_list)):
            e1 = edge_list[i]
            e2 = edge_list[j]
            
            # 跳过共享端点的边
            if (e1['source'] == e2['source'] or e1['source'] == e2['target'] or
                e1['target'] == e2['source'] or e1['target'] == e2['target']):
                continue
            
            # 获取坐标
            p1 = Point(nodes[e1['source']]['x'], nodes[e1['source']]['y'])
            p2 = Point(nodes[e1['target']]['x'], nodes[e1['target']]['y'])
            p3 = Point(nodes[e2['source']]['x'], nodes[e2['source']]['y'])
            p4 = Point(nodes[e2['target']]['x'], nodes[e2['target']]['y'])
            
            if segments_intersect(p1, p2, p3, p4):
                crossings += 1
    
    return crossings


def calculate_total_edge_length(nodes, edges):
    """计算总边长"""
    total_length = 0.0
    for edge in edges:
        src = nodes[edge['source']]
        tgt = nodes[edge['target']]
        dx = src['x'] - tgt['x']
        dy = src['y'] - tgt['y']
        total_length += (dx * dx + dy * dy) ** 0.5
    return total_length


def load_json(filepath):
    """加载JSON文件"""
    with open(filepath, 'r') as f:
        return json.load(f)


def evaluate_solution(instance_file, solution_file, strategy='numba', iterations=10000):
    """
    评估求解器性能
    
    Args:
        instance_file: 问题实例文件
        solution_file: 标准解答文件
        strategy: 求解策略
        iterations: 迭代次数
    
    Returns:
        dict: 评估结果
    """
    print(f"\n{'='*80}")
    print(f"测试实例: {Path(instance_file).name}")
    print(f"{'='*80}")
    
    # 加载数据
    instance = load_json(instance_file)
    solution = load_json(solution_file)
    
    # 标准解答指标
    sol_nodes = {n['id']: n for n in solution['nodes']}
    sol_crossings = count_crossings(sol_nodes, solution['edges'])
    sol_length = calculate_total_edge_length(sol_nodes, solution['edges'])
    
    print(f"\n📊 标准解答:")
    print(f"  - 交叉数: {sol_crossings}")
    print(f"  - 总边长: {sol_length:.2f}")
    
    # 使用我们的求解器
    print(f"\n🚀 运行求解器 (strategy={strategy}, iterations={iterations})...")
    solver = LCNSolver(strategy=strategy)
    solver.load_from_json(instance_file)
    
    start_time = time.time()
    result = solver.optimize(iterations=iterations)
    elapsed_time = time.time() - start_time
    
    # 我们的结果
    our_crossings = result['final_crossings']
    our_length = result['final_edge_length']
    
    print(f"\n✅ 我们的结果:")
    print(f"  - 交叉数: {our_crossings}")
    print(f"  - 总边长: {our_length:.2f}")
    print(f"  - 运行时间: {elapsed_time:.2f}s")
    print(f"  - 初始交叉数: {result['initial_crossings']}")
    print(f"  - 改进: {result['initial_crossings'] - our_crossings} 个交叉")
    
    # 对比
    crossing_diff = our_crossings - sol_crossings
    length_diff = our_length - sol_length
    crossing_ratio = (our_crossings / sol_crossings * 100) if sol_crossings > 0 else 0
    
    print(f"\n📈 性能对比:")
    print(f"  - 交叉数差异: {crossing_diff:+d} ({crossing_ratio:.1f}% of optimal)")
    print(f"  - 边长差异: {length_diff:+.2f}")
    
    if our_crossings <= sol_crossings:
        print(f"  ✨ 优秀! 达到或超越标准解答")
    elif our_crossings <= sol_crossings * 1.5:
        print(f"  👍 良好! 接近标准解答")
    else:
        print(f"  ⚠️  需要改进")
    
    return {
        'instance': Path(instance_file).name,
        'solution': Path(solution_file).name,
        'standard': {
            'crossings': sol_crossings,
            'edge_length': sol_length
        },
        'our_result': {
            'crossings': our_crossings,
            'edge_length': our_length,
            'time': elapsed_time,
            'initial_crossings': result['initial_crossings'],
            'improvement': result['initial_crossings'] - our_crossings
        },
        'comparison': {
            'crossing_diff': crossing_diff,
            'crossing_ratio': crossing_ratio,
            'length_diff': length_diff,
            'is_better': our_crossings <= sol_crossings
        }
    }


def run_all_benchmarks(strategy='numba', iterations=10000):
    """运行所有测试用例"""
    instance_dir = Path(__file__).parent / 'live-2025-example-instances'
    
    test_cases = [
        ('15-nodes.json', 'sol-15-nodes-5-planar.json'),
        ('70-nodes.json', 'sol-70-nodes-16-planar.json'),
        ('100-nodes.json', 'sol-100-nodes-10-planar.json'),
        ('150-nodes.json', 'sol-150-nodes-50-planar.json'),
        ('225-nodes.json', 'sol-225-nodes-16-planar.json'),
        ('625-nodes.json', 'sol-625-nodes-5-planar.json'),
    ]
    
    results = []
    
    for instance_name, solution_name in test_cases:
        instance_file = instance_dir / instance_name
        solution_file = instance_dir / solution_name
        
        if not instance_file.exists() or not solution_file.exists():
            print(f"⚠️  跳过: {instance_name} (文件不存在)")
            continue
        
        try:
            result = evaluate_solution(
                str(instance_file),
                str(solution_file),
                strategy=strategy,
                iterations=iterations
            )
            results.append(result)
        except Exception as e:
            print(f"❌ 错误: {instance_name} - {e}")
            import traceback
            traceback.print_exc()
    
    # 总结报告
    print(f"\n{'='*80}")
    print(f"📊 总结报告")
    print(f"{'='*80}\n")
    
    print(f"{'实例':<20} {'标准交叉':<12} {'我们的交叉':<12} {'差异':<10} {'状态':<10}")
    print(f"{'-'*80}")
    
    total_better = 0
    total_tests = len(results)
    
    for r in results:
        status = '✅' if r['comparison']['is_better'] else '⚠️'
        if r['comparison']['is_better']:
            total_better += 1
        
        print(f"{r['instance']:<20} "
              f"{r['standard']['crossings']:<12} "
              f"{r['our_result']['crossings']:<12} "
              f"{r['comparison']['crossing_diff']:+<10} "
              f"{status:<10}")
    
    print(f"{'-'*80}")
    print(f"\n成功率: {total_better}/{total_tests} ({total_better/total_tests*100:.1f}%)")
    
    # 保存详细结果
    output_file = Path(__file__).parent / 'benchmark_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n详细结果已保存到: {output_file}")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='评估求解器性能')
    parser.add_argument('--strategy', default='numba', 
                       choices=['legacy', 'new', 'numba'],
                       help='求解策略')
    parser.add_argument('--iterations', type=int, default=10000,
                       help='迭代次数')
    parser.add_argument('--single', type=str, default=None,
                       help='只测试单个实例 (例如: 15-nodes.json)')
    
    args = parser.parse_args()
    
    if args.single:
        # 单个测试
        instance_dir = Path(__file__).parent / 'live-2025-example-instances'
        instance_file = instance_dir / args.single
        
        # 找到对应的解答文件
        solution_name = args.single.replace('.json', '')
        solution_files = list(instance_dir.glob(f'sol-{solution_name}*.json'))
        
        if not solution_files:
            print(f"❌ 找不到解答文件: sol-{solution_name}*.json")
            sys.exit(1)
        
        solution_file = solution_files[0]
        
        evaluate_solution(
            str(instance_file),
            str(solution_file),
            strategy=args.strategy,
            iterations=args.iterations
        )
    else:
        # 运行所有测试
        run_all_benchmarks(
            strategy=args.strategy,
            iterations=args.iterations
        )
