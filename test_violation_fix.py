"""
快速測試：驗證違規檢測是否有效
"""

from src.LCNv1.strategies import EnhancedSolverStrategy
from src.LCNv1.initialization import FMMEInitializer

# 創建 solver
initializer = FMMEInitializer(spring_iterations=50)
solver = EnhancedSolverStrategy(init_strategy=initializer)

# 載入測試數據
print("Loading 15-nodes.json...")
solver.load_from_json('live-2025-example-instances/15-nodes.json')

# 獲取初始狀態
initial_stats = solver.get_current_stats()
print(f"\n[Initial State]")
print(f"  K: {initial_stats['k']}")
print(f"  Total crossings: {initial_stats['total_crossings']}")
print(f"  Energy: {initial_stats['energy']:.0f}")

# 運行短時間 SA
print(f"\n[Running SA - 200 iterations]")
result = solver.solve(iterations=200)

# 檢查最終狀態
print(f"\n[Final State]")
print(f"  K: {result['k']}")
print(f"  Total crossings: {result['total_crossings']}")
print(f"  Energy: {result['energy']:.0f}")

# 最終驗證
print(f"\n[Final Validation]")
violations = solver._check_violations()

if violations['total'] == 0:
    print("  ✅ NO VIOLATIONS! Solution is valid.")
else:
    print(f"  ❌ FOUND {violations['total']} VIOLATIONS:")
    print(f"     - Duplicate coords: {violations['duplicate_coords']}")
    print(f"     - Nodes on edges: {violations['nodes_on_edges']}")
    print(f"     - Overlapping edges: {violations['overlapping_edges']}")

print("\nTest complete!")
