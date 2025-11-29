# CUDA Performance Analysis & Optimization Roadmap

## 🔍 問題發現

### 1. Cost Function 不是優化 K-value
**現狀**: CUDA SA 優化的是 **total crossing count**，而非 **K-value**

**證據**:
```cpp
// planar_cuda.cu, Line 1276
int delta_e = compute_delta_e(node_id, new_x, new_y);

// compute_delta_e() 返回 (Lines 1112-1118):
long long new_crossings = calculate_total_crossings();
return new_crossings - current_crossings;  // ❌ 優化 total crossings
```

**影響**:
- SA接受/拒絕move的標準是「能否減少總交叉數」
- 但比賽評分標準是「單條邊的最大交叉數 (K-value)」
- 兩者不一致 → 可能降低總交叉但增加K值

**範例**:
```
Move A: K=10 → K=12, Total=100 → Total=95  (SA會接受，但K變差!)
Move B: K=10 → K=8,  Total=100 → Total=102 (SA會拒絕，但K變好!)
```

---

### 2. GPU 加速效果不明顯

**現象**: 15-nodes 和 100-nodes 執行時間幾乎相同

| Instance    | Iterations | Time (s) | Time/Iter (ms) |
|-------------|-----------|----------|----------------|
| 15-nodes    | 20,000    | 4.81     | 0.24           |
| 70-nodes    | 20,000    | 6.72     | 0.34           |
| 100-nodes   | 20,000    | 5.98     | 0.30           |

**預期**: 100-nodes 應該比 15-nodes 慢很多 (edges² 複雜度)
- 15-nodes: ~70 edges  → ~4,900 edge pairs
- 100-nodes: ~320 edges → ~102,400 edge pairs (20倍!)

**實際**: 時間差異只有 24% (4.81s → 5.98s)

---

## 🐛 根本原因分析

### CPU-GPU 記憶體傳輸瓶頸

**問題**: SA loop 在 CPU 執行，每次迭代都有多次記憶體拷貝

```cpp
// run_sa_optimization(), Line 1268-1300
for (int iter = 0; iter < iterations; iter++) {
    // ❌ 每次迭代都觸發 GPU kernel launch
    int delta_e = compute_delta_e(node_id, new_x, new_y);  
    // compute_delta_e() 內部:
    //   1. cudaMemcpy: Host → Device (new position)
    //   2. launch kernel: count crossings  
    //   3. cudaMemcpy: Device → Host (result)
    //   4. cudaMemcpy: Device → Host (restore position)
    
    if (accept) {
        update_node_position(node_id, new_x, new_y);
        // ❌ 又一次 cudaMemcpy
    }
}
```

**記憶體拷貝成本**:
- 每次 cudaMemcpy: ~10-100 μs (latency)
- 每次迭代: 4-5 次拷貝
- 20,000 iterations → **80,000-100,000 次拷貝**
- 總拷貝時間: ~0.8-10 秒 (大部分時間在等待傳輸!)

**為什麼 100-nodes 不比 15-nodes 慢很多?**
→ **因為瓶頸在記憶體傳輸，不在計算！**

---

### Kernel Launch Overhead

**問題**: 每次 `compute_delta_e()` 都 launch 一個 GPU kernel

```cpp
// calculate_total_crossings(), Lines 940-970
__global__ void count_crossings_kernel(...)  // GPU kernel
dim3 block(256);
dim3 grid((num_pairs + 255) / 256);
count_crossings_kernel<<<grid, block>>>(...);  // ❌ Kernel launch
cudaDeviceSynchronize();  // ❌ CPU等待GPU完成
```

**Kernel launch 成本**:
- Launch overhead: ~5-10 μs
- Synchronization: ~1-5 μs
- 20,000 iterations → **40,000 次 launch** (compute_delta_e 呼叫2次)
- 總 overhead: ~0.2-0.6 秒

---

## 📊 時間分解估算

**20,000 iterations, 100-nodes:**

| Component              | Time (s) | Percentage |
|------------------------|----------|------------|
| CPU-GPU 記憶體傳輸      | 2-4      | 40-60%     |
| Kernel launch overhead | 0.2-0.6  | 5-10%      |
| GPU crossing 計算      | 1-2      | 20-30%     |
| CPU SA logic           | 0.5-1    | 10-15%     |
| **Total**              | **~6**   | **100%**   |

**結論**: **70-80% 時間浪費在記憶體傳輸和overhead，只有20-30%在實際計算！**

---

## ✅ 優化方案

### 方案 1: 修改 Cost Function 使用 K-value (高優先級)

**目標**: SA 直接優化 K-value 而非 total crossings

**實現**:
```cpp
// 新函數: compute_delta_k()
int compute_delta_k(int node_id, int new_x, int new_y) {
    // Step 1: 計算當前K值
    int current_k = calculate_k_value();
    
    // Step 2: 暫時移動node
    update_node_position(node_id, new_x, new_y);
    
    // Step 3: 計算新K值
    int new_k = calculate_k_value();
    
    // Step 4: 還原
    update_node_position(node_id, old_x, old_y);
    
    return new_k - current_k;  // ✅ K-value差異
}

// 修改 run_sa_optimization()
int delta_e = compute_delta_k(node_id, new_x, new_y);  // 改用 delta_k
```

**需要新增**:
- `calculate_k_value()`: 回傳當前K值
- 需要維護每條邊的交叉數 (edge_crossings array)

**優點**:
- ✅ 直接優化比賽指標
- ✅ K-value 變化比 total crossings 更穩定 (更容易收斂)

**缺點**:
- ⚠️ 計算K值需要額外記憶體 (O(E) array)
- ⚠️ 每次需要找max (O(E) 複雜度)

---

### 方案 2: Batch GPU Operations (中優先級)

**目標**: 減少CPU-GPU記憶體傳輸次數

**實現**:
```cpp
// 一次處理多個moves
int run_sa_optimization_batched(int iterations, int batch_size = 100) {
    for (int batch = 0; batch < iterations / batch_size; batch++) {
        // Step 1: 在CPU生成 batch_size 個 moves
        std::vector<Move> moves(batch_size);
        for (int i = 0; i < batch_size; i++) {
            moves[i] = {node_id, new_x, new_y};
        }
        
        // Step 2: 一次性傳輸到GPU
        cudaMemcpy(d_moves, moves.data(), ...);  // ✅ 只傳1次
        
        // Step 3: GPU並行計算所有delta_e
        compute_delta_e_batch<<<grid, block>>>(d_moves, d_delta_e);
        
        // Step 4: 一次性取回結果
        cudaMemcpy(delta_e, d_delta_e, ...);  // ✅ 只取1次
        
        // Step 5: CPU決定accept/reject
        for (int i = 0; i < batch_size; i++) {
            if (accept(delta_e[i])) apply_move(moves[i]);
        }
    }
}
```

**優點**:
- ✅ 記憶體傳輸減少 100 倍 (batch_size = 100)
- ✅ GPU kernel launch 減少 100 倍
- ✅ GPU 利用率提升 (並行計算)

**缺點**:
- ⚠️ Moves 之間有依賴性 (後面的move依賴前面的結果)
- ⚠️ 可能影響SA收斂性

---

### 方案 3: 完全GPU化 SA Loop (低優先級，複雜)

**目標**: 整個SA loop 在GPU執行

**實現**:
```cpp
__global__ void sa_kernel(
    int* d_nodes_x, int* d_nodes_y,
    int iterations, float temperature
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 每個thread獨立運行SA
    curandState state;
    curand_init(seed + tid, 0, 0, &state);
    
    for (int iter = 0; iter < iterations; iter++) {
        // 隨機選node
        int node_id = curand(&state) % num_nodes;
        
        // 計算delta_e (GPU內部)
        int delta_e = ...;
        
        // Metropolis
        if (accept) {
            atomicExch(&d_nodes_x[node_id], new_x);
        }
    }
}
```

**優點**:
- ✅ 完全消除CPU-GPU傳輸
- ✅ 可以多個thread並行運行SA (ensemble)

**缺點**:
- ⚠️ 實現非常複雜
- ⚠️ Atomic operations 可能很慢
- ⚠️ Random number generation 在GPU上較慢
- ⚠️ 需要完全重寫算法

---

## 📋 行動計劃

### Phase 1: Cost Function 修正 (1-2天)
1. [ ] 實現 `calculate_k_value()` GPU kernel
2. [ ] 實現 `compute_delta_k()` 函數
3. [ ] 修改 `run_sa_optimization()` 使用 delta_k
4. [ ] 測試 benchmark: 70/100/150-nodes
5. [ ] 驗證 K-value 是否改善

**預期結果**: K值顯著降低 (目標: 70/100→K≤10)

---

### Phase 2: 參數調優 (1天)
1. [ ] 增加迭代次數: 20k → 50k-100k
2. [ ] 調整溫度參數: 測試不同 start_temp
3. [ ] 調整 cooling_rate: 0.95 → 0.99
4. [ ] Multi-start strategy: 嘗試不同初始配置

**預期結果**: 提升解的質量

---

### Phase 3: GPU Batching (2-3天，如需要)
- 只有在 Phase 1+2 仍無法達到目標時才執行
- 實現 batch GPU operations
- 測試 batch_size: 10, 50, 100

**預期結果**: 速度提升 5-10x

---

## 🎯 預期改善

### Cost Function 修正後:

| Instance | Current K | Target K | Expected K (修正後) |
|----------|-----------|----------|---------------------|
| 70-nodes | 19        | ≤10      | 8-12                |
| 100-nodes| 20        | ≤10      | 8-12                |
| 150-nodes| ?         | ≤15      | 10-15               |

### 速度優化後 (如實施 Batching):

| Instance | Current Time | Expected Time (batched) |
|----------|--------------|-------------------------|
| 70-nodes | 6.7s         | 2-3s                    |
| 100-nodes| 6.0s         | 2-3s                    |
| 150-nodes| ~7s?         | 3-4s                    |

---

## 📝 總結

**關鍵問題**:
1. ❌ Cost function 優化 crossing count 而非 K-value
2. ❌ CPU-GPU 記憶體傳輸瓶頸 (70-80% 時間浪費)

**優先解決**:
1. **修改 cost function** → 最大影響，相對容易
2. **增加迭代次數** → 簡單有效
3. **GPU batching** → 如前兩步不夠才做

**預期**:
- 修正 cost function → **K值改善 50%+**
- GPU batching → **速度提升 5-10x**
