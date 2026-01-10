"""
Comprehensive test: Smart vs Traditional at different scales
"""
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from LCNv1.strategies.new import NewArchitectureSolverStrategy

def test_configuration(instance_file, iterations, label):
    """Test both methods on a configuration"""
    print(f"\n{'='*70}")
    print(f"{label}")
    print(f"File: {instance_file}, Iterations: {iterations}")
    print('='*70)
    
    # Traditional
    solver1 = NewArchitectureSolverStrategy(use_smart_moves=False)
    solver1.load_from_json(instance_file)
    start = time.time()
    result1 = solver1.solve(iterations=iterations)
    time1 = time.time() - start
    
    print(f"\n[Traditional] Time: {time1:.2f}s | K: {result1['k']} | X: {result1['total_crossings']}")
    
    # Smart
    solver2 = NewArchitectureSolverStrategy(use_smart_moves=True, smart_threshold=10.0)
    solver2.load_from_json(instance_file)
    start = time.time()
    result2 = solver2.solve(iterations=iterations)
    time2 = time.time() - start
    
    print(f"[Smart]       Time: {time2:.2f}s | K: {result2['k']} | X: {result2['total_crossings']}")
    
    speedup = time1 / time2 if time2 > 0 else 1.0
    k_diff = result1['k'] - result2['k']
    
    print(f"\n[Result] Speedup: {speedup:.2f}x | K improvement: {k_diff:+d}")
    
    return {
        'time1': time1,
        'time2': time2,
        'speedup': speedup,
        'k1': result1['k'],
        'k2': result2['k'],
        'k_diff': k_diff
    }

print("="*70)
print("Smart Move Generation - Comprehensive Benchmark")
print("="*70)

results = []

# Test 1: Small graph, few iterations (baseline)
results.append(test_configuration(
    'live-2025-example-instances/15-nodes.json',
    500,
    'Test 1: Small graph (15 nodes), 500 iterations'
))

# Test 2: Small graph, many iterations (where smart should win)
results.append(test_configuration(
    'live-2025-example-instances/15-nodes.json',
    2000,
    'Test 2: Small graph (15 nodes), 2000 iterations'
))

# Test 3: Medium graph
if os.path.exists('live-2025-example-instances/70-nodes.json'):
    results.append(test_configuration(
        'live-2025-example-instances/70-nodes.json',
        1000,
        'Test 3: Medium graph (70 nodes), 1000 iterations'
    ))

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

for i, r in enumerate(results, 1):
    status = "FASTER" if r['speedup'] > 1.1 else "SLOWER" if r['speedup'] < 0.9 else "SAME"
    quality = "BETTER" if r['k_diff'] > 0 else "WORSE" if r['k_diff'] < 0 else "SAME"
    print(f"Test {i}: Speedup {r['speedup']:.2f}x [{status}] | K {r['k_diff']:+d} [{quality}]")

avg_speedup = sum(r['speedup'] for r in results) / len(results)
print(f"\nAverage speedup: {avg_speedup:.2f}x")

if avg_speedup > 1.1:
    print("\n[CONCLUSION] Smart generation is FASTER on average!")
elif avg_speedup < 0.9:
    print("\n[CONCLUSION] Traditional is faster (smart has overhead)")
else:
    print("\n[CONCLUSION] Both methods perform similarly")
