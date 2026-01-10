"""
Debug CUDA Smart Move - Check if moves are actually smart
"""
import sys
import json
import os

# Import CUDA module
try:
    os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin')
    
    build_dir = os.path.join(os.path.dirname(__file__), 'build_artifacts')
    sys.path.insert(0, build_dir)
    import planar_cuda
    print(f"✅ CUDA module loaded: {planar_cuda.__version__}\n")
except ImportError as e:
    print(f"❌ Failed to import planar_cuda: {e}")
    sys.exit(1)


# Load graph
with open("results/06-11-00/15-nodes-cu-k3.json", 'r') as f:
    data = json.load(f)

nodes_x = [node['x'] for node in data['nodes']]
nodes_y = [node['y'] for node in data['nodes']]
edges = [(edge['source'], edge['target']) for edge in data['edges']]

solver = planar_cuda.PlanarSolver(nodes_x, nodes_y, edges)

print("📍 Original positions:")
for i in range(min(5, len(nodes_x))):
    print(f"  Node {i}: ({nodes_x[i]}, {nodes_y[i]})")

print("\n🎲 Testing generate_smart_move for node 0:")
for test in range(10):
    new_pos = solver.generate_smart_move(0, step_size=50, max_retries=10)
    
    # Check if duplicate
    is_dup = any(nodes_x[i] == new_pos[0] and nodes_y[i] == new_pos[1] 
                 for i in range(len(nodes_x)) if i != 0)
    
    dup_marker = "❌ DUPLICATE!" if is_dup else "✅"
    print(f"  Try {test+1}: ({new_pos[0]}, {new_pos[1]}) {dup_marker}")

# Test SA with detailed stats
print("\n" + "="*60)
print("Running SA with smart moves...")
print("="*60)

stats = solver.run_sa_optimization(
    iterations=100,
    start_temp=100.0,
    cooling_rate=0.99,
    cost_function="total_crossings",
    use_smart_moves=True,
    smart_threshold=10.0
)

print(f"\nResults:")
print(f"  Initial K: {stats['initial_k']:.0f}")
print(f"  Final K: {stats['final_k']:.0f}")
print(f"  Accepted: {stats['accepted_moves']:.0f}")
print(f"  Rejected: {stats['rejected_moves']:.0f}")
print(f"  Acceptance rate: {stats['accepted_moves']/(stats['accepted_moves']+stats['rejected_moves'])*100:.1f}%")
