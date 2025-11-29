# K-Plane Drawing Comparison - 15 Nodes

## Results Summary

| Solution | K-Value | Total Crossings | Performance |
|----------|---------|-----------------|-------------|
| **15-node-others.json** | **8** | **68** | ❌ Poor |
| **15-nodes-ours.json** | **4** | **29** | ✅ **BEST** |
| Standard baseline | 5 | 62 | Reference |

## Key Findings

### Our Solution (15-nodes-ours.json)
- **K-value: 4** ⭐ (Lower is better!)
- **Total crossings: 29**
- **Improvement over "others":**
  - K reduced by **50%** (8 → 4)
  - Total crossings reduced by **57%** (68 → 29)
- **Improvement over standard:**
  - K reduced by **20%** (5 → 4)
  - Total crossings reduced by **53%** (62 → 29)

### "Others" Solution (15-node-others.json)
- **K-value: 8** ❌ (Worst performer)
- **Total crossings: 68**
- Performs **worse** than standard baseline
- Edge #43 (10→12) has 8 crossings alone!

## Detailed Edge Distribution

### Our Solution
```
31 edges with 0 crossings
 6 edges with 1 crossing
13 edges with 2 crossings
 6 edges with 3 crossings
 2 edges with 4 crossings (max)
```
**Worst edges:** #51 (1→9), #54 (1→14) with 4 crossings each

### "Others" Solution
```
15 edges with 0 crossings
10 edges with 1 crossing
 8 edges with 2 crossings
 7 edges with 3 crossings
 9 edges with 4 crossings
 4 edges with 5 crossings
 3 edges with 6 crossings
 1 edge with 7 crossings
 1 edge with 8 crossings (max)
```
**Worst edge:** #43 (10→12) with 8 crossings

## Conclusion

**Our K-Plane optimization algorithm is working correctly and producing superior results!**

The "15-node-others.json" file you uploaded has:
- K=8 (worse than standard K=5)
- Total=68 crossings (worse than standard 62)

This confirms that our solution (K=4, Total=29) is significantly better than both the standard baseline and the "others" solution.

## Problem Definition Compliance

✅ Our calculation method is **100% correct** according to the problem statement:
- "To count crossings we take the maximum number of crossings over all edges" → K=4 ✓
- "k-plane drawing is one where each edge has at most k crossings" → 4-plane drawing ✓
- "Crossings will always be counted pairwise" → 29 total pairs ✓

---
**Generated:** 2025-01-29
**Algorithm:** K-Plane Cost Function with FMME Initialization
**Optimization:** Simulated Annealing (10,000 iterations, T₀=100)
