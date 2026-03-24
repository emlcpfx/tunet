@echo off
REM ==============================================================================
REM TuNet PyInstaller Build Script
REM
REM Usage (run from project root or scripts/ directory):
REM   scripts\build_exe.bat          - Build full distribution (with CUDA, ~5.2 GB)
REM   scripts\build_exe.bat package  - Package dist for GitHub Release upload
REM ==============================================================================

REM Always run from project root (one level up from this script)
cd /d "%~dp0.."

echo ============================================
echo  TuNet PyInstaller Build
echo ============================================
echo.

REM Activate conda environment
call conda activate tunet
if errorlevel 1 (
    echo ERROR: Failed to activate tunet conda environment.
    echo Make sure the tunet environment exists: conda env list
    pause
    exit /b 1
)

REM Verify PyInstaller is installed
python -c "import PyInstaller" 2>nul
if errorlevel 1 (
    echo Installing PyInstaller...
    pip install pyinstaller
)

REM Parse argument
set BUILD_TYPE=%1
if "%BUILD_TYPE%"=="" set BUILD_TYPE=build

if "%BUILD_TYPE%"=="package" goto :package
if "%BUILD_TYPE%"=="build" goto :build

echo Unknown option: %BUILD_TYPE%
echo Usage: scripts\build_exe.bat [build^|package]
pause
exit /b 1

:build
echo Building full distribution (with CUDA)...
echo.

REM Clean previous build
if exist "build\tunet" rmdir /s /q "build\tunet"
if exist "dist\TuNet" rmdir /s /q "dist\TuNet"

set OPENCV_IO_ENABLE_OPENEXR=1

pyinstaller scripts\tunet.spec --noconfirm

if errorlevel 1 (
    echo.
    echo ERROR: PyInstaller build failed!
    pause
    exit /b 1
)

echo.
echo ============================================
echo  Build Complete!
echo ============================================
echo  Output: dist\TuNet\TuNet.exe
echo.
echo  Next: run 'scripts\build_exe.bat package' to create
echo  GitHub Release archives.
echo ============================================
goto :done

:package
echo Creating release packages for GitHub upload...
echo.

if not exist "dist\TuNet\TuNet.exe" (
    echo ERROR: No distribution found. Run 'scripts\build_exe.bat' first.
    pause
    exit /b 1
)

set /p VERSION="Enter version (e.g. 0.1.0): "
python scripts\package_release.py --version %VERSION%

if errorlevel 1 (
    echo.
    echo ERROR: Packaging failed!
    pause
    exit /b 1
)

goto :done

:done
echo.
pause
