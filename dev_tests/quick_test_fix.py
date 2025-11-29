import os
import sys
import json

os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin')
sys.path.insert(0, 'build_artifacts')

import planar_cuda

# Test with 15-nodes
data = json.load(open('live-2025-example-instances/15-nodes.json'))
solver = planar_cuda.PlanarSolver(
    [n['x'] for n in data['nodes']], 
    [n['y'] for n in data['nodes']], 
    [(e['source'], e['target']) for e in data['edges']], 
    100, 
    data['width'], 
    data['height']
)

print("Running quick test (5000 iterations)...")
stats = solver.run_sa_optimization(5000, 100, 0.95)
x, y = solver.get_coordinates()

print(f"Final crossings: {int(stats['final_crossings'])}")
print(f"Accepted moves: {int(stats['accepted_moves'])}")

# Save result for validation
result = {
    'nodes': [{'id': i, 'x': int(x[i]), 'y': int(y[i])} for i in range(len(x))],
    'edges': data['edges'],
    'width': data['width'],
    'height': data['height']
}

with open('results/test-fix.json', 'w') as f:
    json.dump(result, f, indent=2)

print("\n✅ Result saved to results/test-fix.json")
print("Now validating...")

# Validate
sys.path.insert(0, 'dev_tests')
from strict_violation_check import check_file
is_valid = check_file('results/test-fix.json')
if is_valid:
    print("\n🎉 SUCCESS: No violations detected!")
else:
    print("\n❌ FAIL: Violations still present")
