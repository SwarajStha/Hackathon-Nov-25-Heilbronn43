"""
简单测试 - 验证约束enforcement
"""

import json
from src.LCNv1.strategies.enhanced import EnhancedSolverStrategy
from src.LCNv1.initialization.fmme import FMMEInitializer

def main():
    # 加载数据
    with open('live-2025-example-instances/15-nodes.json', 'r') as f:
        data = json.load(f)
    
    # 创建solver
    init_strategy = FMMEInitializer()
    solver = EnhancedSolverStrategy(init_strategy=init_strategy)
    
    # 加载数据（使用文件路径）
    solver.load_from_json('live-2025-example-instances/15-nodes.json')
    
    # 尝试优化 - 应该在初始验证时失败（如果FMME产生了违规）
    try:
        result = solver.solve(iterations=5000)
        print(f"Success! K={result['k']}, Crossings={result['total_crossings']}")
        
        # 保存结果
        output_data = {
            'nodes': result['nodes'],
            'edges': data['edges'],
            'width': data['width'],
            'height': data['height']
        }
        
        with open('results/11-29-02/15-nodes-valid-test.json', 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print("Saved to results/11-29-02/15-nodes-valid-test.json")
        
    except RuntimeError as e:
        print(f"Optimization failed: {e}")
        print("\nThis is expected if initialization produces violations.")
        print("We need a violation-free initialization strategy.")

if __name__ == '__main__':
    main()
