"""
在真實測試點上評估自適應 FMME 的效果

測試數據：
- 15-nodes.json
- 70-nodes.json
- 100-nodes.json
- 150-nodes.json
- 225-nodes.json
- 625-nodes.json

比較：
1. 固定 50 次迭代
2. 固定 100 次迭代
3. 自適應迭代（50-200，patience=5）
"""

import time
import json
import os
from pathlib import Path
from src.LCNv1.initialization import FMMEInitializer
from src.LCNv1.strategies import EnhancedSolverStrategy


def benchmark_single_instance(instance_file, fmme_config, sa_iterations=1000):
    """
    測試單個實例
    
    Returns:
        dict: 測試結果
    """
    print(f"\n{'='*80}")
    print(f"Testing: {os.path.basename(instance_file)}")
    print(f"FMME Config: {fmme_config['name']}")
    print(f"{'='*80}")
    
    # 創建初始化器
    initializer = FMMEInitializer(**fmme_config['params'])
    
    # 創建 solver
    solver = EnhancedSolverStrategy(init_strategy=initializer)
    
    # 測量初始化時間
    start_init = time.time()
    try:
        solver.load_from_json(instance_file)
    except Exception as e:
        print(f"  [ERROR] Failed to load: {e}")
        return None
    init_time = time.time() - start_init
    
    # 獲取初始統計
    initial_stats = solver.get_current_stats()
    actual_iters = initializer.last_actual_iterations
    convergence = initializer.last_convergence_reason
    
    print(f"\n[Initialization Results]")
    print(f"  Time: {init_time:.3f}s")
    print(f"  Actual FMME iterations: {actual_iters}")
    print(f"  Convergence: {convergence}")
    print(f"  Initial K: {initial_stats['k']}")
    print(f"  Initial crossings: {initial_stats['total_crossings']}")
    print(f"  Initial energy: {initial_stats['energy']:.0f}")
    
    # 運行 SA 優化
    print(f"\n[SA Optimization] {sa_iterations} iterations...")
    start_sa = time.time()
    try:
        result = solver.solve(iterations=sa_iterations)
    except Exception as e:
        print(f"  [ERROR] Optimization failed: {e}")
        return None
    sa_time = time.time() - start_sa
    
    print(f"\n[Final Results]")
    print(f"  SA time: {sa_time:.3f}s")
    print(f"  Final K: {result['k']}")
    print(f"  Final crossings: {result['total_crossings']}")
    print(f"  Final energy: {result['energy']:.0f}")
    print(f"  Total time: {init_time + sa_time:.3f}s")
    print(f"  K improvement: {initial_stats['k']} → {result['k']} (-{initial_stats['k'] - result['k']})")
    
    return {
        'instance': os.path.basename(instance_file),
        'fmme_config': fmme_config['name'],
        'num_nodes': initial_stats['num_nodes'],
        'num_edges': initial_stats['num_edges'],
        'fmme_iterations': actual_iters,
        'convergence_reason': convergence,
        'init_time': init_time,
        'initial_k': initial_stats['k'],
        'initial_crossings': initial_stats['total_crossings'],
        'initial_energy': initial_stats['energy'],
        'sa_time': sa_time,
        'final_k': result['k'],
        'final_crossings': result['total_crossings'],
        'final_energy': result['energy'],
        'total_time': init_time + sa_time,
        'k_improvement': initial_stats['k'] - result['k'],
        'acceptance_rate': result['stats']['acceptance_rate']
    }


def run_full_benchmark():
    """運行完整基準測試"""
    
    # 測試實例
    test_instances = [
        'live-2025-example-instances/15-nodes.json',
        'live-2025-example-instances/70-nodes.json',
        'live-2025-example-instances/100-nodes.json',
        'live-2025-example-instances/150-nodes.json',
        'live-2025-example-instances/225-nodes.json',
        # 'live-2025-example-instances/625-nodes.json',  # 太大，可選
    ]
    
    # FMME 配置
    fmme_configs = [
        {
            'name': 'Fixed-50',
            'params': {'spring_iterations': 50}
        },
        {
            'name': 'Fixed-100',
            'params': {'spring_iterations': 100}
        },
        {
            'name': 'Adaptive-Default',
            'params': {
                'enable_adaptive': True,
                'min_iterations': 50,
                'max_iterations': 200,
                'patience': 5
            }
        },
        {
            'name': 'Adaptive-Aggressive',
            'params': {
                'enable_adaptive': True,
                'min_iterations': 50,
                'max_iterations': 300,
                'patience': 3,
                'improvement_threshold': 0.0005
            }
        },
    ]
    
    # SA 迭代次數（根據圖大小調整）
    sa_iterations_map = {
        '15-nodes.json': 500,
        '70-nodes.json': 1000,
        '100-nodes.json': 1000,
        '150-nodes.json': 1500,
        '225-nodes.json': 2000,
        '625-nodes.json': 3000,
    }
    
    all_results = []
    
    print("\n" + "="*80)
    print("ADAPTIVE FMME BENCHMARK - REAL TEST INSTANCES")
    print("="*80)
    
    for instance_file in test_instances:
        if not os.path.exists(instance_file):
            print(f"\n[SKIP] File not found: {instance_file}")
            continue
        
        instance_name = os.path.basename(instance_file)
        sa_iters = sa_iterations_map.get(instance_name, 1000)
        
        for fmme_config in fmme_configs:
            result = benchmark_single_instance(
                instance_file, 
                fmme_config, 
                sa_iterations=sa_iters
            )
            
            if result:
                all_results.append(result)
            
            # 短暫暫停
            time.sleep(1)
    
    # 生成報告
    generate_report(all_results)
    
    # 保存結果
    output_file = 'benchmark_adaptive_fmme_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n\nResults saved to: {output_file}")
    
    return all_results


def generate_report(results):
    """生成詳細報告"""
    
    print("\n\n" + "="*100)
    print("BENCHMARK REPORT - SUMMARY")
    print("="*100)
    
    # 按實例分組
    instances = sorted(set(r['instance'] for r in results))
    
    for instance in instances:
        instance_results = [r for r in results if r['instance'] == instance]
        
        print(f"\n{'─'*100}")
        print(f"Instance: {instance} ({instance_results[0]['num_nodes']} nodes, {instance_results[0]['num_edges']} edges)")
        print(f"{'─'*100}")
        print(f"{'Config':<25} {'FMME Iter':<12} {'Init(s)':<10} {'Init K':<8} {'Final K':<10} {'SA(s)':<10} {'Total(s)':<10}")
        print(f"{'─'*100}")
        
        for r in instance_results:
            print(f"{r['fmme_config']:<25} {r['fmme_iterations']:<12} {r['init_time']:<10.2f} "
                  f"{r['initial_k']:<8} {r['final_k']:<10} {r['sa_time']:<10.2f} {r['total_time']:<10.2f}")
        
        # 分析最佳配置
        best_final_k = min(instance_results, key=lambda x: x['final_k'])
        best_init_k = min(instance_results, key=lambda x: x['initial_k'])
        fastest = min(instance_results, key=lambda x: x['total_time'])
        
        print(f"\n  Best final K: {best_final_k['fmme_config']} (K={best_final_k['final_k']})")
        print(f"  Best initial K: {best_init_k['fmme_config']} (K={best_init_k['initial_k']})")
        print(f"  Fastest: {fastest['fmme_config']} ({fastest['total_time']:.2f}s)")
    
    # 整體分析
    print("\n\n" + "="*100)
    print("OVERALL ANALYSIS")
    print("="*100)
    
    configs = sorted(set(r['fmme_config'] for r in results))
    
    for config in configs:
        config_results = [r for r in results if r['fmme_config'] == config]
        
        avg_init_time = sum(r['init_time'] for r in config_results) / len(config_results)
        avg_fmme_iters = sum(r['fmme_iterations'] for r in config_results) / len(config_results)
        avg_initial_k = sum(r['initial_k'] for r in config_results) / len(config_results)
        avg_final_k = sum(r['final_k'] for r in config_results) / len(config_results)
        avg_total_time = sum(r['total_time'] for r in config_results) / len(config_results)
        
        print(f"\n{config}:")
        print(f"  Avg FMME iterations: {avg_fmme_iters:.1f}")
        print(f"  Avg init time: {avg_init_time:.2f}s")
        print(f"  Avg initial K: {avg_initial_k:.1f}")
        print(f"  Avg final K: {avg_final_k:.1f}")
        print(f"  Avg total time: {avg_total_time:.2f}s")
    
    # 效率比較（以 Fixed-50 為基準）
    print("\n\n" + "="*100)
    print("EFFICIENCY COMPARISON (vs Fixed-50)")
    print("="*100)
    
    baseline_results = [r for r in results if r['fmme_config'] == 'Fixed-50']
    
    for config in configs:
        if config == 'Fixed-50':
            continue
        
        config_results = [r for r in results if r['fmme_config'] == config]
        
        # 配對比較
        time_diffs = []
        k_improvements = []
        
        for baseline in baseline_results:
            matching = [r for r in config_results if r['instance'] == baseline['instance']]
            if matching:
                comp = matching[0]
                time_diffs.append(comp['init_time'] - baseline['init_time'])
                k_improvements.append(baseline['final_k'] - comp['final_k'])
        
        if time_diffs:
            avg_time_diff = sum(time_diffs) / len(time_diffs)
            avg_k_improvement = sum(k_improvements) / len(k_improvements)
            
            print(f"\n{config}:")
            print(f"  Avg init time difference: {avg_time_diff:+.2f}s")
            print(f"  Avg K improvement: {avg_k_improvement:+.1f}")
            if avg_time_diff > 0:
                efficiency = avg_k_improvement / avg_time_diff
                print(f"  Efficiency: {efficiency:.2f} K-improvement per second")


def quick_test():
    """快速測試（只測試小圖）"""
    print("Quick Test: Adaptive FMME on 15-nodes")
    print("="*80)
    
    configs = [
        {'name': 'Fixed-50', 'params': {'spring_iterations': 50}},
        {'name': 'Adaptive', 'params': {'enable_adaptive': True, 'min_iterations': 50, 'max_iterations': 200}},
    ]
    
    results = []
    
    for config in configs:
        result = benchmark_single_instance(
            'live-2025-example-instances/15-nodes.json',
            config,
            sa_iterations=500
        )
        if result:
            results.append(result)
    
    # 簡單比較
    if len(results) == 2:
        print("\n" + "="*80)
        print("COMPARISON")
        print("="*80)
        print(f"Fixed-50:  Init K={results[0]['initial_k']}, Final K={results[0]['final_k']}, Time={results[0]['total_time']:.2f}s")
        print(f"Adaptive:  Init K={results[1]['initial_k']}, Final K={results[1]['final_k']}, Time={results[1]['total_time']:.2f}s")
        print(f"\nImprovement: K {results[0]['final_k']} → {results[1]['final_k']} ({results[0]['final_k'] - results[1]['final_k']:+d})")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'quick':
        # 快速測試
        quick_test()
    else:
        # 完整基準測試
        run_full_benchmark()
