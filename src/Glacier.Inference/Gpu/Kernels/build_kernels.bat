@echo off
setlocal enabledelayedexpansion

echo ===================================================
echo Glacier.Inference CUDA Kernel Fatbinary Builder
echo ===================================================

:: 1. Check for nvcc in PATH
where nvcc >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] nvcc.exe not found in PATH.
    echo Please install the NVIDIA CUDA Toolkit or add it to PATH.
    exit /b 1
)

:: 2. Find and initialize Visual Studio MSVC environment if cl.exe is not in PATH
where cl.exe >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    set "VS_VCVARS="
    for %%V in (
        "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
        "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
        "C:\Program Files\Microsoft Visual Studio\18\Professional\VC\Auxiliary\Build\vcvars64.bat"
        "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
        "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
        "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
    ) do (
        if exist %%V (
            set "VS_VCVARS=%%~V"
            goto :FoundVcVars
        )
    )

    :: Fallback search via vswhere if available
    set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
    if exist "%VSWHERE%" (
        for /f "usebackq tokens=*" %%I in (`"%VSWHERE%" -latest -property installationPath`) do (
            if exist "%%I\VC\Auxiliary\Build\vcvars64.bat" (
                set "VS_VCVARS=%%I\VC\Auxiliary\Build\vcvars64.bat"
                goto :FoundVcVars
            )
        )
    )

:FoundVcVars
    if defined VS_VCVARS (
        echo [INFO] Initializing MSVC environment from: !VS_VCVARS!
        call "!VS_VCVARS!"
    ) else (
        echo [WARNING] vcvars64.bat not found automatically. nvcc will use default system PATH.
    )
)

:: 3. Compile universal fatbinary CUBIN supporting:
::    - sm_75 (Turing: RTX 2060, GTX 1660, T4)
::    - sm_80 (Ampere Data Center: A100)
::    - sm_86 (Ampere Consumer: RTX 3060, RTX 3070, RTX 3080, RTX 3090)
::    - sm_89 (Ada Lovelace: RTX 4060, RTX 4070, RTX 4080, RTX 4090, L40)
::    - sm_90 (Hopper: H100)
::    - compute_75 (PTX forward compatibility for Blackwell & future architectures)

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

echo [INFO] Compiling kernels.cu to kernels.cubin...
nvcc -fatbin -O3 ^
  -gencode arch=compute_75,code=sm_75 ^
  -gencode arch=compute_80,code=sm_80 ^
  -gencode arch=compute_86,code=sm_86 ^
  -gencode arch=compute_89,code=sm_89 ^
  -gencode arch=compute_90,code=sm_90 ^
  -gencode arch=compute_75,code=compute_75 ^
  -o kernels.cubin kernels.cu

if %ERRORLEVEL% EQU 0 (
    echo [SUCCESS] Successfully built kernels.cubin fatbinary.
    for %%F in (kernels.cubin) do echo [INFO] Size: %%~zF bytes
) else (
    echo [ERROR] nvcc compilation failed with error code %ERRORLEVEL%.
    exit /b %ERRORLEVEL%
)
