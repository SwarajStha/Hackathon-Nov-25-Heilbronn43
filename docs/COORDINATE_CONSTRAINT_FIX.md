# CUDA坐標約束修復方案

## 問題

當前CUDA SA優化**違反坐標約束**:

```cpp
// 錯誤: 添加margin導致超出邊界
min_x -= width / 4;  // 可能導致負值
max_x += width / 4;  // 超出width
min_y -= height / 4; // 可能導致負值
max_y += height / 4; // 超出height
```

**結果**:
- 原始: X∈[0,186], Y∈[0,201]
- 優化後: X∈[-45,222], Y∈[-36,219] ❌ 違規!

## 約束條件

根據題目要求:
```
0 ≤ x ≤ width  (默認: 1,000,000)
0 ≤ y ≤ height (默認: 1,000,000)
```

## 修復方案

### 方案1: 修改CUDA代碼 (推薦)

**步驟1**: 添加width/height參數到構造函數

```cpp
class PlanarSolver {
private:
    int max_width;   // 最大X坐標
    int max_height;  // 最大Y坐標
    
public:
    PlanarSolver(
        const std::vector<int>& nodes_x,
        const std::vector<int>& nodes_y,
        const std::vector<std::pair<int, int>>& edges,
        int cell_size = -1,
        int width = 1000000,   // 新參數
        int height = 1000000   // 新參數
    ) : max_width(width), max_height(height), ... {
        // ...
    }
```

**步驟2**: 修改SA中的坐標生成

```cpp
// 舊代碼 (錯誤):
int min_x = *std::min_element(nodes_x.begin(), nodes_x.end());
int max_x = *std::max_element(nodes_x.begin(), nodes_x.end());
min_x -= width / 4;  // ❌
max_x += width / 4;  // ❌

// 新代碼 (正確):
std::uniform_int_distribution<> x_dist(0, max_width);
std::uniform_int_distribution<> y_dist(0, max_height);
```

**優點**:
- 嚴格遵守約束
- 簡單直接
- 不需要後處理

### 方案2: Python後處理裁剪 (臨時方案)

如果暫時無法重新編譯CUDA:

```python
def clip_coordinates(data):
    """裁剪坐標到有效範圍"""
    width = data.get('width', 1000000)
    height = data.get('height', 1000000)
    
    for node in data['nodes']:
        node['x'] = max(0, min(node['x'], width))
        node['y'] = max(0, min(node['y'], height))
    
    return data
```

**缺點**:
- 改變優化結果
- 可能增加交叉數
- 不是根本解決方案

### 方案3: 使用約束條件的SA

更複雜但更優雅:

```cpp
// 生成滿足約束的候選位置
int new_x = x_dist(gen);
int new_y = y_dist(gen);

// 確保在邊界內
new_x = std::max(0, std::min(new_x, max_width));
new_y = std::max(0, std::min(new_y, max_height));
```

## 實現計劃

### 立即行動: 使用方案2修復當前結果

創建腳本修復已保存的結果文件。

### 長期方案: 實現方案1

修改CUDA源碼,重新編譯,確保從源頭避免違規。

## 影響評估

**K值變化**:
裁剪坐標可能影響K值,因為:
- 某些節點位置被強制移動
- 邊的角度/長度改變
- 可能增加新的交叉

**建議**:
修復後重新運行優化,使用正確的約束條件。
