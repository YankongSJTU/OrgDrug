@echo off
REM One-click demo: synthetic data -> 1-fold LCLO training -> metrics (CPU ok)
cd /d "%~dp0"
python demo\run_demo.py %*
