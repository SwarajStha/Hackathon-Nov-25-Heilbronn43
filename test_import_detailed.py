"""
Detailed import test with error diagnostics
"""
import sys
import os

# CRITICAL: Add CUDA DLL directory (Windows specific)
os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin')
# Add build_artifacts to Python path (where .pyd is located)
sys.path.insert(0, '.')

print("=" * 60)
print("Import Diagnostics")
print("=" * 60)
print(f"Python: {sys.version}")
print(f"CWD: {os.getcwd()}")
print(f"sys.path[0]: {sys.path[0]}")
print()

# Try importing with detailed error
try:
    import planar_cuda
    print("✓ SUCCESS: Module imported")
    print(f"  Version: {planar_cuda.__version__}")
    print(f"  CUDA enabled: {planar_cuda.cuda_enabled}")
    
    # Try creating solver
    try:
        solver = planar_cuda.PlanarSolver([0, 10], [0, 0], [(0, 1)])
        print("✓ SUCCESS: Solver created")
        
        # Try calling methods
        try:
            crossings = solver.calculate_total_crossings()
            print(f"✓ SUCCESS: calculate_total_crossings() = {crossings}")
        except Exception as e:
            print(f"✗ FAIL: calculate_total_crossings() error: {e}")
        
        try:
            k_value = solver.calculate_k_value()
            print(f"✓ SUCCESS: calculate_k_value() = {k_value}")
        except Exception as e:
            print(f"✗ FAIL: calculate_k_value() error: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"✗ FAIL: Solver creation error: {e}")
        import traceback
        traceback.print_exc()
        
except ImportError as e:
    print(f"✗ IMPORT FAILED: {e}")
    print()
    print("Possible causes:")
    print("1. Missing CUDA runtime DLL (cudart64_12.dll)")
    print("2. Missing Visual C++ runtime")
    print("3. Incompatible Python version")
    print("4. Module compiled with different settings")
    print()
    
    # Check if file exists
    pyd_path = os.path.join(os.getcwd(), 'planar_cuda.pyd')
    if os.path.exists(pyd_path):
        print(f"✓ File exists: {pyd_path}")
        print(f"  Size: {os.path.getsize(pyd_path)} bytes")
    else:
        print(f"✗ File not found: {pyd_path}")
    
    import traceback
    traceback.print_exc()
