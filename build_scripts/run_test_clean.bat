@echo off
echo Cleaning Python cache files...
del /s /q "src\LCNv1\initialization\__pycache__\*.pyc" 2>nul
rmdir /s /q "src\LCNv1\initialization\__pycache__" 2>nul
echo Cache cleaned!
echo.
echo Running test...
python test_adaptive_fmme.py demo
