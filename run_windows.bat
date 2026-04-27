@echo off
cd /d "%~dp0"
py -3 run_thermal_sim.py
if errorlevel 1 (
    echo.
    echo If Python is not found, install Python 3 from https://www.python.org/downloads/windows/
    pause
)
