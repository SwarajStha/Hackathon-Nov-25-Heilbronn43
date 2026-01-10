"""
Final Comparison: Python vs CUDA Smart Moves
"""
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Python version
from LCNv1.strategies.new import NewArchitectureSolverStrategy

# CUDA version
os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin')
build_dir = os.path.join(os.path.dirname(__file__), 'build_artifacts')
sys.path.insert(0, build_dir)
import planar_cuda
import json

print("=" * 80)
print("FINAL COMPARISON: Python vs CUDA Smart Move Generation")
print("=" * 80)

test_file = 'live-2025-example-instances/15-nodes.json'
iterations = 2000
runs = 3

# Load data for CUDA
with open(test_file, 'r') as f:
    data = json.load(f)

# Use FIXED coordinates from file (not random!)
nodes_x = [node['x'] for node in data['nodes']]
nodes_y = [node['y'] for node in data['nodes']]
edges = [(edge['source'], edge['target']) for edge in data['edges']]

print(f"\n📊 Test: {test_file}")
print(f"   Nodes: {len(nodes_x)}, Edges: {len(edges)}")
print(f"   Iterations: {iterations}, Runs: {runs}")
print(f"   Using SAME initial state for fair comparison\n")

# =============================================================================
# Python Version
# =============================================================================
print("🐍 Python Version (with smart moves)")
print("-" * 80)

python_times = []
python_k_values = []

for run in range(runs):
    solver = NewArchitectureSolverStrategy(use_smart_moves=True, smart_threshold=10.0)
    solver.load_from_json(test_file)
    
    start = time.time()
    result = solver.solve(iterations=iterations, initial_temp=100.0, cooling_rate=0.995)
    elapsed = time.time() - start
    
    python_times.append(elapsed)
    python_k_values.append(result['k'])
    print(f"  Run {run+1}: K={result['k']}, time={elapsed:.3f}s")

# =============================================================================
# CUDA Version
# =============================================================================
print("\n⚡ CUDA Version (with smart moves)")
print("-" * 80)

cuda_times = []
cuda_k_values = []

for run in range(runs):
    solver_cuda = planar_cuda.PlanarSolver(nodes_x, nodes_y, edges)
    
    start = time.time()
    stats = solver_cuda.run_sa_optimization(
        iterations=iterations,
        start_temp=100.0,
        cooling_rate=0.995,
        cost_function="total_crossings",
        use_smart_moves=True,
        smart_threshold=10.0
    )
    elapsed = time.time() - start
    
    cuda_times.append(elapsed)
    cuda_k_values.append(stats['final_k'])
    print(f"  Run {run+1}: K={stats['final_k']:.0f}, time={elapsed:.3f}s")

# =============================================================================
# Comparison
# =============================================================================
print("\n" + "=" * 80)
print("📈 COMPARISON")
print("=" * 80)

avg_python_time = sum(python_times) / len(python_times)
avg_cuda_time = sum(cuda_times) / len(cuda_times)
best_python_k = min(python_k_values)
best_cuda_k = min(cuda_k_values)

speedup = avg_python_time / avg_cuda_time
k_improvement = best_python_k - best_cuda_k

print(f"\n⏱️ Speed:")
print(f"  Python avg: {avg_python_time:.3f}s")
print(f"  CUDA avg:   {avg_cuda_time:.3f}s")
print(f"  Speedup:    {speedup:.2f}x {'✅' if speedup < 1.0 else '⚠️'}")

print(f"\n🎯 Best K-Value:")
print(f"  Python: {best_python_k}")
print(f"  CUDA:   {best_cuda_k:.0f}")
print(f"  Diff:   {k_improvement:+.0f} {'✅' if abs(k_improvement) <= 2 else '⚠️'}")

print(f"\n📊 Verdict:")
if speedup < 1.0 and abs(k_improvement) <= 2:
    print("  🎉 SUCCESS! CUDA is faster with similar quality")
elif speedup < 1.0:
    print("  ⚡ CUDA is faster but quality differs")
elif abs(k_improvement) <= 2:
    print("  ✅ Quality is similar but speed is comparable")
else:
    print("  ⚠️ Mixed results - needs further investigation")

print("\n" + "=" * 80)
