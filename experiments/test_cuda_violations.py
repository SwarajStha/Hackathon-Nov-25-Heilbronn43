"""
Diagnostic: Check violation penalties in CUDA
"""
import sys
import os
import json

os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin')
sys.path.insert(0, 'build_artifacts')
import planar_cuda

# Load simple graph
with open("live-2025-example-instances/15-nodes.json", 'r') as f:
    data = json.load(f)

nodes_x = [node['x'] for node in data['nodes']]
nodes_y = [node['y'] for node in data['nodes']]
edges = [(edge['source'], edge['target']) for edge in data['edges']]

solver = planar_cuda.PlanarSolver(nodes_x, nodes_y, edges)

print("="*60)
print("Violation Penalty Diagnostic")
print("="*60)

# Test 1: Try to move node to duplicate position
print("\n[Test 1] Moving to duplicate position")
print(f"Node 0 at: ({nodes_x[0]}, {nodes_y[0]})")
print(f"Node 1 at: ({nodes_x[1]}, {nodes_y[1]})")

# Try to move node 0 to node 1's position
delta = solver.compute_delta_e(0, nodes_x[1], nodes_y[1])
print(f"Delta for moving node 0 to node 1's position: {delta}")
if delta >= 1000000000:
    print("✅ Duplicate position penalty working (1B+)")
else:
    print(f"⚠️ Penalty too low: {delta}")

# Test 2: Normal move
print("\n[Test 2] Normal move")
delta_normal = solver.compute_delta_e(0, nodes_x[0] + 10, nodes_y[0] + 10)
print(f"Delta for normal move: {delta_normal}")
if delta_normal < 1000000:
    print("✅ Normal move has reasonable delta")
else:
    print(f"⚠️ Normal move has huge penalty: {delta_normal}")

# Test 3: Run short SA and check violations
print("\n[Test 3] Short SA run")
stats = solver.run_sa_optimization(
    iterations=100,
    start_temp=50.0,
    cooling_rate=0.99,
    use_smart_moves=False  # Use random to test violation detection
)

print(f"Initial K: {stats['initial_k']:.0f}")
print(f"Final K: {stats['final_k']:.0f}")
print(f"Accepted: {stats['accepted_moves']:.0f}/{100}")
print(f"Acceptance rate: {stats['accepted_moves']/100*100:.1f}%")

# Check final state for duplicates
final_coords = solver.get_coordinates()
fx, fy = final_coords

duplicates = 0
for i in range(len(fx)):
    for j in range(i+1, len(fx)):
        if fx[i] == fx[j] and fy[i] == fy[j]:
            duplicates += 1
            print(f"⚠️ DUPLICATE: Node {i} and {j} at ({fx[i]}, {fy[i]})")

if duplicates == 0:
    print("✅ No duplicate coordinates in final state")
else:
    print(f"❌ Found {duplicates} duplicate coordinate pairs!")

print("\n" + "="*60)
