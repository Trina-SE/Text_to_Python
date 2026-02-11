@echo off
REM ==============================================================
REM Docstring-to-Code Generation - Single Script Runner (Windows)
REM ==============================================================
REM Usage:
REM   run.bat                     - Train and evaluate all models
REM   run.bat --model rnn         - Train only Vanilla RNN
REM   run.bat --model lstm        - Train only LSTM
REM   run.bat --model attention   - Train only LSTM with Attention
REM   run.bat --eval-only         - Evaluate existing checkpoints
REM   run.bat --epochs 5          - Train for 5 epochs
REM ==============================================================

set IMAGE_NAME=docstring-to-code
set CONTAINER_NAME=docstring-to-code-run

echo ============================================================
echo Docstring-to-Code Generation Pipeline
echo ============================================================
echo.

echo [Step 1/2] Building Docker image...
docker build -t %IMAGE_NAME% .
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Docker build failed.
    exit /b 1
)
echo Docker image built successfully.
echo.

REM Remove existing container if present
docker rm -f %CONTAINER_NAME% 2>nul

echo [Step 2/2] Running training and evaluation pipeline...
echo.

REM Try with GPU first, fall back to CPU
docker run --gpus all --name %CONTAINER_NAME% -v "%cd%\Checkpoints:/app/Checkpoints" -v "%cd%\Attention_Visualizations:/app/Attention_Visualizations" -v "%cd%\Evaluation:/app/Evaluation" -v "%cd%\Notebook:/app/Notebook" %IMAGE_NAME% python main.py %* 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo GPU not available, running on CPU...
    docker rm -f %CONTAINER_NAME% 2>nul
    docker run --name %CONTAINER_NAME% -v "%cd%\Checkpoints:/app/Checkpoints" -v "%cd%\Attention_Visualizations:/app/Attention_Visualizations" -v "%cd%\Evaluation:/app/Evaluation" -v "%cd%\Notebook:/app/Notebook" %IMAGE_NAME% python main.py %*
)

echo.
echo ============================================================
echo Pipeline complete!
echo.
echo Outputs:
echo   Checkpoints\                  - Model checkpoints
echo   Evaluation\                   - Metrics, plots, results.json
echo   Attention_Visualizations\     - Attention heatmaps
echo ============================================================
