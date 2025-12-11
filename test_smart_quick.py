"""
Quick test: Smart Move Generator vs Traditional Detect-and-Reject
"""
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from LCNv1.strategies.new import NewArchitectureSolverStrategy

print("=" * 70)
print("Smart Move Generation vs Traditional Detection")
print("=" * 70)

test_file = 'live-2025-example-instances/15-nodes.json'
iterations = 500

# Test 1: Traditional (with violation check)
print("\n[Test 1] Traditional - Detect and Reject")
print("-" * 70)

solver1 = NewArchitectureSolverStrategy(use_smart_moves=False)
solver1.load_from_json(test_file)

start = time.time()
result1 = solver1.solve(iterations=iterations)
time1 = time.time() - start

print(f"Time: {time1:.2f}s")
print(f"K-value: {result1['k']}")
print(f"Total crossings: {result1['total_crossings']}")

# Test 2: Smart Generation (prevent violations)
print("\n[Test 2] Smart - Prevent at Source")
print("-" * 70)

solver2 = NewArchitectureSolverStrategy(use_smart_moves=True, smart_threshold=10.0)
solver2.load_from_json(test_file)

start = time.time()
result2 = solver2.solve(iterations=iterations)
time2 = time.time() - start

print(f"Time: {time2:.2f}s")
print(f"K-value: {result2['k']}")
print(f"Total crossings: {result2['total_crossings']}")

# Comparison
print("\n" + "=" * 70)
print("COMPARISON")
print("=" * 70)

speedup = time1 / time2 if time2 > 0 else 1.0
k_improvement = result1['k'] - result2['k']

print(f"\nSpeed:")
print(f"  Traditional: {time1:.2f}s")
print(f"  Smart:       {time2:.2f}s")
print(f"  Speedup:     {speedup:.2f}x {'[FASTER]' if speedup > 1.1 else '[SAME]'}")

print(f"\nQuality:")
print(f"  Traditional K: {result1['k']}")
print(f"  Smart K:       {result2['k']}")
print(f"  Improvement:   {k_improvement} {'[BETTER]' if k_improvement > 0 else '[SAME]'}")

print(f"\nConclusion:")
if speedup > 1.1 and k_improvement >= 0:
    print("  [SUCCESS] Smart generation is FASTER and BETTER!")
elif speedup > 1.1:
    print("  [SUCCESS] Smart generation is FASTER (quality similar)")
elif k_improvement > 0:
    print("  [SUCCESS] Smart generation has BETTER quality (speed similar)")
else:
    print("  [INFO] Both methods perform similarly")

print("\n" + "=" * 70)
