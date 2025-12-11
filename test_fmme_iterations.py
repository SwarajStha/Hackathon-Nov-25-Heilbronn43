"""
測試不同 FMME 迭代次數的影響

比較：
- 初始化時間
- 初始 K 值
- SA 後的最終 K 值
- 總運行時間
"""

import time
import json
from src.LCNv1.initialization import FMMEInitializer
from src.LCNv1.strategies import EnhancedSolverStrategy


def test_fmme_iterations(input_file, iterations_list):
    """
    測試不同 FMME 迭代次數
    
    Args:
        input_file: 輸入 JSON 文件
        iterations_list: FMME 迭代次數列表，例如 [20, 50, 100, 200, 500]
    """
    results = []
    
    print(f"Testing FMME iterations on: {input_file}")
    print(f"Iterations to test: {iterations_list}\n")
    print("-" * 80)
    
    for fmme_iters in iterations_list:
        print(f"\n[FMME Iterations: {fmme_iters}]")
        
        # 1. 創建帶有不同迭代次數的初始化器
        initializer = FMMEInitializer(spring_iterations=fmme_iters)
        
        # 2. 創建 solver
        solver = EnhancedSolverStrategy(init_strategy=initializer)
        
        # 3. 測量初始化時間
        start_init = time.time()
        solver.load_from_json(input_file)
        init_time = time.time() - start_init
        
        # 4. 獲取初始統計
        initial_stats = solver.get_current_stats()
        initial_k = initial_stats['k']
        initial_crossings = initial_stats['total_crossings']
        
        print(f"  Initialization time: {init_time:.3f}s")
        print(f"  Initial K: {initial_k}")
        print(f"  Initial crossings: {initial_crossings}")
        
        # 5. 運行 SA 優化（固定迭代次數以公平比較）
        start_sa = time.time()
        result = solver.solve(iterations=1000)  # 固定 1000 次迭代
        sa_time = time.time() - start_sa
        
        # 6. 獲取最終結果
        final_k = result['k']
        final_crossings = result['total_crossings']
        total_time = init_time + sa_time
        
        print(f"  SA time: {sa_time:.3f}s")
        print(f"  Final K: {final_k}")
        print(f"  Final crossings: {final_crossings}")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Improvement: {initial_k - final_k} (K), {initial_crossings - final_crossings} (crossings)")
        
        # 7. 記錄結果
        results.append({
            'fmme_iterations': fmme_iters,
            'init_time': init_time,
            'initial_k': initial_k,
            'initial_crossings': initial_crossings,
            'sa_time': sa_time,
            'final_k': final_k,
            'final_crossings': final_crossings,
            'total_time': total_time,
            'k_improvement': initial_k - final_k,
            'crossing_improvement': initial_crossings - final_crossings
        })
    
    # 8. 生成比較報告
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'FMME Iter':<12} {'Init(s)':<10} {'Init K':<8} {'Final K':<10} {'K Improve':<12} {'Total(s)':<10}")
    print("-" * 80)
    
    for r in results:
        print(f"{r['fmme_iterations']:<12} {r['init_time']:<10.3f} {r['initial_k']:<8} "
              f"{r['final_k']:<10} {r['k_improvement']:<12} {r['total_time']:<10.3f}")
    
    # 9. 找出最佳配置
    best_final = min(results, key=lambda x: x['final_k'])
    best_total = min(results, key=lambda x: x['total_time'])
    best_init = min(results, key=lambda x: x['initial_k'])
    
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print(f"Best final K: {best_final['fmme_iterations']} iterations (K={best_final['final_k']})")
    print(f"Best initial K: {best_init['fmme_iterations']} iterations (K={best_init['initial_k']})")
    print(f"Fastest total time: {best_total['fmme_iterations']} iterations ({best_total['total_time']:.3f}s)")
    
    # 10. 成本效益分析
    print("\n" + "=" * 80)
    print("COST-BENEFIT ANALYSIS")
    print("=" * 80)
    
    baseline = results[0]  # 假設第一個是基準
    for r in results[1:]:
        extra_time = r['init_time'] - baseline['init_time']
        k_benefit = baseline['final_k'] - r['final_k']
        
        if extra_time > 0:
            efficiency = k_benefit / extra_time if extra_time > 0 else 0
            print(f"{r['fmme_iterations']} iters vs {baseline['fmme_iterations']}: "
                  f"+{extra_time:.3f}s → K improvement: {k_benefit} "
                  f"(efficiency: {efficiency:.2f} K/sec)")
    
    return results


def test_multiple_instances():
    """測試多個實例"""
    test_cases = [
        ('sample.json', [20, 50, 100, 200]),
        # 如果有其他測試文件，可以添加：
        # ('live-2025-example-instances/o01.json', [20, 50, 100, 200, 500]),
        # ('live-2025-example-instances/o02.json', [20, 50, 100]),
    ]
    
    all_results = {}
    
    for input_file, iterations in test_cases:
        try:
            print(f"\n\n{'#' * 80}")
            print(f"# Testing: {input_file}")
            print(f"{'#' * 80}")
            results = test_fmme_iterations(input_file, iterations)
            all_results[input_file] = results
        except Exception as e:
            print(f"Error testing {input_file}: {e}")
            continue
    
    # 保存結果
    output_file = 'fmme_iterations_test_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n\nResults saved to: {output_file}")


if __name__ == '__main__':
    # 快速測試
    print("Quick Test: FMME Iterations Impact")
    print("=" * 80)
    
    # 測試範圍：20 到 200 次迭代
    # 選擇對數分布以涵蓋廣泛範圍
    test_fmme_iterations(
        input_file='sample.json',
        iterations_list=[20, 50, 100, 200]
    )
    
    # 如果要測試多個實例，取消註釋：
    # test_multiple_instances()
