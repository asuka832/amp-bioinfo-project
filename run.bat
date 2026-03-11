@echo off
setlocal

REM Bypass broken system proxy for this process only.
set NO_PROXY=*
set no_proxy=*

call conda run -n amp_env python main.py
if errorlevel 1 (
  echo Pipeline failed.
  exit /b 1
)

endlocal
