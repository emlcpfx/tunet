@echo off
REM Always run from project root (one level up from this script)
cd /d "%~dp0.."

echo ============================================
echo TuNet Environment Setup
echo ============================================
echo.

REM Check if conda is available
where conda >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Conda not found!
    echo Please install Miniconda or Anaconda first:
    echo https://docs.conda.io/en/latest/miniconda.html
    echo.
    pause
    exit /b 1
)

echo Creating conda environment from environment.yml...
echo.
conda env create -f environment.yml

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Environment creation failed!
    echo If the environment already exists, you can update it with:
    echo   conda env update -f environment.yml --prune
    echo.
    pause
    exit /b 1
)

echo.
echo ============================================
echo Environment created successfully!
echo ============================================
echo.
echo To activate the environment, run:
echo   conda activate tunet
echo.
echo Then you can:
echo   - Train models: python train.py --config your_config.yaml
echo   - Run inference: python inference.py --checkpoint model.pth --input_dir images/ --output_dir output/
echo   - Launch UI: python ui_app.py
echo   - Multi-GPU inference: python inference_gui_multigpu.py
echo   - Monitor training: python training_monitor.py --output_dir your_output_folder
echo.
pause
