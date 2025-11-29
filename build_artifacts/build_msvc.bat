@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

set ROOT_DIR=%~dp0..
cd /d "%~dp0"

echo Compiling CUDA kernel...
for /f "tokens=*" %%i in ('D:/D_backup/2025/tum/25W/hackthon/Hackathon-Nov-25-Heilbronn43/heilbron-43/Scripts/python.exe -c "import pybind11; print(pybind11.get_include())"') do set PYBIND11_INCLUDE=%%i
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\nvcc.exe" -c "%ROOT_DIR%/src/cuda_utils/planar_cuda.cu" -o build_direct/planar_cuda.obj -arch=sm_89 --compiler-options "/EHsc /MD" -I"%ROOT_DIR%/src" -I"C:\Users\aloha\AppData\Local\Programs\Python\Python311\Include" -I"%PYBIND11_INCLUDE%"
if %ERRORLEVEL% NEQ 0 exit /b 1

echo Linking module...
for /f "tokens=*" %%i in ('D:/D_backup/2025/tum/25W/hackthon/Hackathon-Nov-25-Heilbronn43/heilbron-43/Scripts/python.exe -c "import sys; print(sys.base_prefix)"') do set PYTHON_BASE=%%i
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\nvcc.exe" --shared build_direct/planar_cuda.obj -o planar_cuda.pyd -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\lib\x64" -L"%PYTHON_BASE%\libs" -lcudart -lpython311 -Xlinker "/NODEFAULTLIB:MSVCRT" -Xlinker "legacy_stdio_definitions.lib" -Xlinker "ucrt.lib" -Xlinker "vcruntime.lib" -Xlinker "msvcrt.lib"
if %ERRORLEVEL% NEQ 0 exit /b 1

echo Build successful!
