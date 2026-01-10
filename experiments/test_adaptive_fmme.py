"""
測試自適應 FMME 的效果

比較：
1. 固定迭代（50次）
2. 固定迭代（100次）
3. 自適應迭代（50-200次，patience=5）
4. 自適應迭代（30-150次，patience=10）
"""

import time
import json
from src.LCNv1.initialization import FMMEInitializer
from src.LCNv1.strategies import EnhancedSolverStrategy


def test_adaptive_vs_fixed(input_file='sample.json'):
    """比較自適應 vs 固定迭代"""
    
    print("=" * 80)
    print("ADAPTIVE FMME TEST")
    print("=" * 80)
    print(f"Input: {input_file}\n")
    
    test_configs = [
        {
            'name': 'Fixed 50 iters',
            'init': FMMEInitializer(spring_iterations=50),
        },
        {
            'name': 'Fixed 100 iters',
            'init': FMMEInitializer(spring_iterations=100),
        },
        {
            'name': 'Adaptive (50-200, patience=5)',
            'init': FMMEInitializer(
                enable_adaptive=True,
                min_iterations=50,
                max_iterations=200,
                patience=5,
                improvement_threshold=0.001
            ),
        },
        {
            'name': 'Adaptive (30-150, patience=10)',
            'init': FMMEInitializer(
                enable_adaptive=True,
                min_iterations=30,
                max_iterations=150,
                patience=10,
                improvement_threshold=0.001
            ),
        },
        {
            'name': 'Adaptive Aggressive (50-300, patience=3)',
            'init': FMMEInitializer(
                enable_adaptive=True,
                min_iterations=50,
                max_iterations=300,
                patience=3,
                improvement_threshold=0.0005
            ),
        },
    ]
    
    results = []
    
    for config in test_configs:
        print("\n" + "-" * 80)
        print(f"Testing: {config['name']}")
        print("-" * 80)
        
        # 創建 solver
        solver = EnhancedSolverStrategy(init_strategy=config['init'])
        
        # 測量初始化時間
        start_init = time.time()
        solver.load_from_json(input_file)
        init_time = time.time() - start_init
        
        # 獲取初始統計
        initial_stats = solver.get_current_stats()
        
        # 獲取實際迭代次數（如果是自適應）
        actual_iters = config['init'].last_actual_iterations
        convergence_reason = config['init'].last_convergence_reason
        
        print(f"\nInitialization Results:")
        print(f"  Time: {init_time:.3f}s")
        print(f"  Actual iterations: {actual_iters}")
        print(f"  Convergence reason: {convergence_reason}")
        print(f"  Initial K: {initial_stats['k']}")
        print(f"  Initial crossings: {initial_stats['total_crossings']}")
        
        # 運行 SA 優化（固定迭代次數以公平比較）
        print(f"\nRunning SA optimization (1000 iterations)...")
        start_sa = time.time()
        result = solver.solve(iterations=1000)
        sa_time = time.time() - start_sa
        
        print(f"\nFinal Results:")
        print(f"  SA time: {sa_time:.3f}s")
        print(f"  Final K: {result['k']}")
        print(f"  Final crossings: {result['total_crossings']}")
        print(f"  Total time: {init_time + sa_time:.3f}s")
        
        # 記錄結果
        results.append({
            'name': config['name'],
            'actual_iterations': actual_iters,
            'convergence_reason': convergence_reason,
            'init_time': init_time,
            'initial_k': initial_stats['k'],
            'initial_crossings': initial_stats['total_crossings'],
            'sa_time': sa_time,
            'final_k': result['k'],
            'final_crossings': result['total_crossings'],
            'total_time': init_time + sa_time,
        })
    
    # 生成比較報告
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'Strategy':<35} {'Iters':<8} {'Init(s)':<10} {'Init K':<8} {'Final K':<10} {'Total(s)':<10}")
    print("-" * 80)
    
    for r in results:
        print(f"{r['name']:<35} {r['actual_iterations']:<8} {r['init_time']:<10.3f} "
              f"{r['initial_k']:<8} {r['final_k']:<10} {r['total_time']:<10.3f}")
    
    # 分析最佳配置
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    
    best_final_k = min(results, key=lambda x: x['final_k'])
    best_init_k = min(results, key=lambda x: x['initial_k'])
    fastest_total = min(results, key=lambda x: x['total_time'])
    fastest_init = min(results, key=lambda x: x['init_time'])
    
    print(f"\nBest final K: {best_final_k['name']}")
    print(f"  K={best_final_k['final_k']}, Total time={best_final_k['total_time']:.3f}s")
    
    print(f"\nBest initial K: {best_init_k['name']}")
    print(f"  K={best_init_k['initial_k']}, Init time={best_init_k['init_time']:.3f}s")
    
    print(f"\nFastest total time: {fastest_total['name']}")
    print(f"  Time={fastest_total['total_time']:.3f}s, Final K={fastest_total['final_k']}")
    
    print(f"\nFastest initialization: {fastest_init['name']}")
    print(f"  Time={fastest_init['init_time']:.3f}s, Iterations={fastest_init['actual_iterations']}")
    
    # 效率分析
    print("\n" + "=" * 80)
    print("EFFICIENCY ANALYSIS (vs Fixed 50)")
    print("=" * 80)
    
    baseline = results[0]  # Fixed 50
    
    for r in results[1:]:
        time_diff = r['init_time'] - baseline['init_time']
        k_diff = baseline['final_k'] - r['final_k']
        
        efficiency = f"{k_diff / time_diff:.2f} K/sec" if time_diff > 0 else "N/A"
        
        print(f"\n{r['name']}:")
        print(f"  Time difference: {time_diff:+.3f}s ({r['init_time']/baseline['init_time']*100:.1f}% of baseline)")
        print(f"  K improvement: {k_diff:+d}")
        print(f"  Efficiency: {efficiency}")
        print(f"  Iterations used: {r['actual_iterations']} ({r['convergence_reason']})")
    
    # 保存結果
    output_file = 'adaptive_fmme_test_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n\nResults saved to: {output_file}")
    
    return results


def quick_demo():
    """快速演示自適應 FMME"""
    print("Quick Demo: Adaptive FMME")
    print("=" * 80)
    
    # 創建自適應初始化器
    adaptive_init = FMMEInitializer(
        enable_adaptive=True,
        min_iterations=50,
        max_iterations=200,
        patience=5
    )
    
    # 創建 solver
    solver = EnhancedSolverStrategy(init_strategy=adaptive_init)
    
    # 載入並觀察自適應過程
    print("\nLoading graph with adaptive FMME...")
    solver.load_from_json('sample.json')
    
    print(f"\n✓ Initialization completed!")
    print(f"  Actual iterations: {adaptive_init.last_actual_iterations}")
    print(f"  Reason: {adaptive_init.last_convergence_reason}")
    
    stats = solver.get_current_stats()
    print(f"  Initial K: {stats['k']}")
    print(f"  Initial crossings: {stats['total_crossings']}")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'demo':
        # 快速演示
        quick_demo()
    else:
        # 完整測試
        test_adaptive_vs_fixed('sample.json')
