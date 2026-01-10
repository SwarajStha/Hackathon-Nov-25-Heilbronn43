# Project Structure

This document describes the organized file structure of the K-Planar Graph Minimizer project.

## 📁 Directory Organization

### Root Directory (Clean!)
- `README.md` - Main project documentation
- `requirements.txt` - Python dependencies
- `sample.json` - Example input file
- `ALGORITHM_ARCHITECTURE.md` - Algorithm documentation
- `SYSTEM_SUMMARY.md` - Technical summary
- `planar_cuda.pyd` - Compiled CUDA module
- `.gitignore` - Git ignore rules

### 📁 Core Directories

#### `src/` - Source Code
Main application code:
- `app.py` - GUI application
- `solver.py` - Simulated annealing solver
- `strategies.py` - Optimization strategies
- `cuda_utils/` - CUDA acceleration modules
- `LCNv1/` - Legacy code version 1

#### `tests/` - Unit Tests
Formal test suite for the core modules.

#### `docs/` - Documentation
Additional documentation and guides.

---

### 📁 Development & Testing Directories

#### `benchmarks/` - Performance Benchmarks
All performance testing scripts:
- `benchmark_*.py` - Various benchmark tests
- Comparative analysis (CUDA vs Numba, strategies, etc.)

#### `analysis/` - Analysis Scripts
Data analysis and investigation:
- `analyze_*.py` - Graph analysis tools
- K-value calculations and validations

#### `experiments/` - Experimental Tests
Development and experimental test scripts:
- `test_*.py` - All experimental test files
- Feature testing and validation

#### `utilities/` - Utility Scripts
Helper scripts and tools:
- `check_*.py` - Validation scripts
- `calculate_*.py` - Calculation utilities
- `debug_*.py` - Debugging tools
- `plot_*.py` - Plotting utilities
- Other one-off utilities

---

### 📁 Data & Results Directories

#### `outputs/` - Generated Results
Benchmark results, plots, and analysis outputs:
- `.txt` - Text results
- `.csv` - Tabular data
- `.json` - JSON results
- `.png` - Generated plots

#### `results/` - Processed Results
Finalized and processed result data.

#### `live-2025-example-instances/` - Test Instances
Example graph instances for testing.

---

### 📁 Build & Scripts Directories

#### `build_scripts/` - Build Scripts
Build automation scripts:
- `*.bat` - Windows batch files for building

#### `build_artifacts/` - Build Artifacts
Compiled binaries and build outputs.

#### `build_cycle1/` - Build Configuration
CMake and build configuration files.

#### `scripts/` - General Scripts
Various utility scripts.

#### `dev_tests/` - Development Tests
Additional development testing files.

---

### 📁 Environment & Configuration

#### `heilbron-43/` - Virtual Environment
Python virtual environment (do not commit).

#### `.vscode/` - VS Code Settings
Editor configuration.

#### `.pytest_cache/` - Pytest Cache
Test framework cache (auto-generated).

---

## 🚀 Quick Navigation

**Want to...**
- Run the app? → `python src/app.py`
- Run tests? → `python -m unittest -v` (from `tests/`)
- Benchmark? → Check `benchmarks/`
- Analyze results? → Check `analysis/` or `outputs/`
- Debug issues? → Check `utilities/` for debug scripts
- Build CUDA? → Check `build_scripts/`

---

## 📝 Notes

- All experimental test files have been moved to `experiments/`
- Benchmark results are now in `outputs/`
- The root directory is now clean and focused on essential files
- Development workflow files are organized by purpose

**Last Updated:** January 10, 2026
