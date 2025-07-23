@echo off

REM Setup Python environment for Wuwei AI, Change If Needed
call conda activate nn
call "%~dp0..\..\alpha_zero\._env\Scripts\activate.bat"

REM Change directory to the parent of the script location
cd /d "%~dp0.."

REM Start Wuwei
if "%1"=="" (
    echo Start Wuwei AI with Policy Network Only
    python main.py gtp
) else if /I "%1"=="MCTS" (
    echo Start Wuwei AI with MCTS
    python main.py gtp MCTS
) else (
    echo Invalid argument. Use 'MCTS' or No argument.
    exit /b 1
)