@echo off
setlocal enabledelayedexpansion

echo =====================================================================
echo   NanoGEMM - Building Native FASM 32-bit and 64-bit Engines
echo =====================================================================

cd /d "%~dp0"

set FASM=
if exist "C:\proekts\FASM\FASM.EXE" set FASM=C:\proekts\FASM\FASM.EXE
if not defined FASM if exist "C:\asm\hdd\FASM.EXE" set FASM=C:\asm\hdd\FASM.EXE
if not defined FASM (
    echo [ERROR] FASM.EXE not found in C:\proekts\FASM or C:\asm\hdd
    exit /b 1
)

echo Using Flat Assembler: %FASM%
echo.

echo [1/4] Assembling nanogemm64.dll (x86-64 AVX2+FMA DLL) ...
"%FASM%" nanogemm64.asm nanogemm64.dll
if %ERRORLEVEL% neq 0 (
    echo [FAIL] Assembly failed for nanogemm64.asm
    exit /b %ERRORLEVEL%
)

echo.
echo [2/4] Assembling test_nanogemm64.exe (x86-64 Standalone Test Suite) ...
"%FASM%" test_nanogemm64.asm test_nanogemm64.exe
if %ERRORLEVEL% neq 0 (
    echo [FAIL] Assembly failed for test_nanogemm64.asm
    exit /b %ERRORLEVEL%
)

echo.
echo [3/4] Assembling nanogemm32.dll (x86 32-bit SSE2 DLL) ...
"%FASM%" nanogemm32.asm nanogemm32.dll
if %ERRORLEVEL% neq 0 (
    echo [FAIL] Assembly failed for nanogemm32.asm
    exit /b %ERRORLEVEL%
)

echo.
echo [4/4] Assembling test_nanogemm32.exe (x86 32-bit Standalone Test Suite) ...
"%FASM%" test_nanogemm32.asm test_nanogemm32.exe
if %ERRORLEVEL% neq 0 (
    echo [FAIL] Assembly failed for test_nanogemm32.asm
    exit /b %ERRORLEVEL%
)

echo.
echo =====================================================================
echo   All 4 Targets Built Successfully!
echo =====================================================================
echo.
echo --- Executing 64-bit Test Suite ---
test_nanogemm64.exe
if %ERRORLEVEL% neq 0 (
    echo [ERROR] test_nanogemm64.exe failed with code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

echo.
echo --- Executing 32-bit Test Suite ---
test_nanogemm32.exe
if %ERRORLEVEL% neq 0 (
    echo [ERROR] test_nanogemm32.exe failed with code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

echo.
echo [SUCCESS] Both 32-bit and 64-bit FASM Test Suites Passed Cleanly!
