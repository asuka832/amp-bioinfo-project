@echo off
setlocal

echo [1/4] Create or update conda env from environment.yml
call conda env update -f environment.yml --prune
if errorlevel 1 (
  echo Failed: conda env update
  exit /b 1
)

echo [2/4] Verify pandas import
call conda run -n amp_env python -c "import pandas as pd; print('pandas', pd.__version__)"
if errorlevel 1 (
  echo Failed: pandas import
  exit /b 1
)

echo [3/4] Verify torch/cuda import
call conda run -n amp_env python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
if errorlevel 1 (
  echo Failed: torch import
  exit /b 1
)

echo [4/4] Done.
echo Next:
echo   conda activate amp_env
echo   python main.py
endlocal
