@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

echo Compiling Cycle 1 CUDA module...
set PYBIND11_INCLUDE=D:\D_backup\2025\tum\25W\hackthon\Hackathon-Nov-25-Heilbronn43\heilbron-43\Lib\site-packages\pybind11\include
set PYTHON_BASE=C:\Users\aloha\AppData\Local\Programs\Python\Python311
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\nvcc.exe" --shared src/cuda_utils/planar_cuda.cu -o build_artifacts/planar_cuda.pyd -arch=sm_89 --compiler-options "/EHsc /MD" -Isrc -I"%PYTHON_BASE%\Include" -I"%PYBIND11_INCLUDE%" -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\lib\x64" -L"%PYTHON_BASE%\libs" -lcudart -lpython311 -Xlinker "/NODEFAULTLIB:MSVCRT" -Xlinker "legacy_stdio_definitions.lib" -Xlinker "ucrt.lib" -Xlinker "vcruntime.lib" -Xlinker "msvcrt.lib"
if %ERRORLEVEL% NEQ 0 exit /b 1

echo Build successful!
