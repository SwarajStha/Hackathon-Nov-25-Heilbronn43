"""
測試智能移動生成器 vs 傳統檢測拒絕方法
Test Smart Move Generator vs Traditional Detect-and-Reject

比較：
1. 違規生成次數
2. 計算時間
3. 最終解質量
"""
import sys
import os
import time
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from LCNv1.core.geometry import Point
from LCNv1.core.graph import GraphData, GridState
from LCNv1.core.cost import SoftMaxCost
from LCNv1.strategies.new import NewArchitectureSolverStrategy


def test_smart_vs_traditional():
    """比較智能生成 vs 傳統檢測"""
    
    # 載入測試數據
    test_file = 'live-2025-example-instances/15-nodes.json'
    if not os.path.exists(test_file):
        print(f"❌ {test_file} 不存在")
        return
    
    print("=" * 70)
    print("智能移動生成器 vs 傳統檢測拒絕 - 性能對比")
    print("=" * 70)
    
    iterations = 1000
    
    # 測試1：傳統方法（檢測後拒絕）
    print("\n【測試1】傳統方法 - 檢測後拒絕違規移動")
    print("-" * 70)
    
    solver_traditional = NewArchitectureSolverStrategy(
        w_cross=100.0,
        w_len=1.0,
        power=2,
        use_smart_moves=False  # 不使用智能生成
    )
    solver_traditional.load_from_json(test_file)
    
    start = time.time()
    result_traditional = solver_traditional.solve(
        iterations=iterations,
        initial_temp=50.0,
        cooling_rate=0.995
    )
    time_traditional = time.time() - start
    
    print(f"[OK] 完成")
    print(f"   時間: {time_traditional:.3f}s")
    print(f"   K-value: {result_traditional['k']}")
    print(f"   總交叉數: {result_traditional['crossings']}")
    print(f"   最終能量: {result_traditional['final_energy']:.1f}")
    
    # 測試2：智能生成（源頭避免違規）
    print("\n【測試2】智能方法 - 源頭避免違規移動")
    print("-" * 70)
    
    solver_smart = NewArchitectureSolverStrategy(
        w_cross=100.0,
        w_len=1.0,
        power=2,
        use_smart_moves=True,  # 使用智能生成
        smart_threshold=10.0   # 溫度 <= 10 使用智能生成
    )
    solver_smart.load_from_json(test_file)
    
    start = time.time()
    result_smart = solver_smart.solve(
        iterations=iterations,
        initial_temp=50.0,
        cooling_rate=0.995
    )
    time_smart = time.time() - start
    
    print(f"✅ 完成")
    print(f"   時間: {time_smart:.3f}s")
    print(f"   K-value: {result_smart['k']}")
    print(f"   總交叉數: {result_smart['crossings']}")
    print(f"   最終能量: {result_smart['final_energy']:.1f}")
    
    # 對比分析
    print("\n" + "=" * 70)
    print("【性能對比】")
    print("=" * 70)
    
    speedup = time_traditional / time_smart if time_smart > 0 else 1.0
    k_improvement = result_traditional['k'] - result_smart['k']
    
    print(f"\n時間效率:")
    print(f"  傳統方法: {time_traditional:.3f}s")
    print(f"  智能方法: {time_smart:.3f}s")
    print(f"  加速比: {speedup:.2f}x {'✅' if speedup > 1 else '❌'}")
    
    print(f"\n解質量:")
    print(f"  傳統 K-value: {result_traditional['k']}")
    print(f"  智能 K-value: {result_smart['k']}")
    print(f"  改進: {k_improvement} {'✅' if k_improvement >= 0 else '❌'}")
    
    print(f"\n計算資源:")
    if speedup > 1.1:
        print(f"  ✅ 智能生成節省 {(1 - 1/speedup)*100:.1f}% 計算時間")
    elif speedup < 0.9:
        print(f"  ❌ 智能生成額外消耗 {(speedup - 1)*100:.1f}% 計算時間")
    else:
        print(f"  ➖ 兩者性能相當")
    
    # 測試違規檢查
    print("\n" + "=" * 70)
    print("【違規檢測】")
    print("=" * 70)
    
    # 檢查傳統方法結果
    print("\n傳統方法最終解:")
    violations_traditional = check_violations(solver_traditional.state, solver_traditional.graph)
    
    # 檢查智能方法結果
    print("\n智能方法最終解:")
    violations_smart = check_violations(solver_smart.state, solver_smart.graph)
    
    # 總結
    print("\n" + "=" * 70)
    print("【總結】")
    print("=" * 70)
    
    if violations_smart == 0 and violations_traditional > 0:
        print("✅ 智能生成成功避免違規！")
    elif violations_smart == 0 and violations_traditional == 0:
        print("✅ 兩種方法都沒有違規")
    else:
        print("❌ 智能生成仍有違規，需要進一步優化")
    
    if speedup > 1.1 and k_improvement >= 0:
        print("✅ 智能生成在速度和質量上都優於傳統方法！")
    elif speedup > 1.1:
        print("➖ 智能生成速度更快，但解質量稍差")
    elif k_improvement > 0:
        print("➖ 智能生成解質量更好，但速度稍慢")
    else:
        print("➖ 兩種方法表現接近")


def check_violations(state: GridState, graph: GraphData) -> int:
    """檢查違規數量"""
    from LCNv1.core.cost import SoftMaxCost
    
    cost_func = SoftMaxCost()
    violations = 0
    
    # 檢查重複坐標
    positions = set()
    for nid in range(graph.num_nodes):
        pos = state.get_position(nid)
        if pos in positions:
            print(f"  ❌ 重複坐標: 節點 {nid} 位於 {pos}")
            violations += 1
        positions.add(pos)
    
    # 檢查節點在邊上
    for nid in range(graph.num_nodes):
        pos = state.get_position(nid)
        for edge in graph.edges:
            src, tgt = edge
            if src == nid or tgt == nid:
                continue
            
            src_pos = state.get_position(src)
            tgt_pos = state.get_position(tgt)
            
            if cost_func._point_on_segment_strict(pos, src_pos, tgt_pos):
                print(f"  ❌ 節點在邊上: 節點 {nid} 在邊 ({src}, {tgt}) 上")
                violations += 1
    
    if violations == 0:
        print("  ✅ 無違規")
    else:
        print(f"  ❌ 共 {violations} 個違規")
    
    return violations


if __name__ == '__main__':
    test_smart_vs_traditional()
