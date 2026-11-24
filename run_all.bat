@echo off
rem Runs the whole pipeline (01 ... 11) on Windows: activates the venv, forces UTF-8 mode and calls run_all.py.
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
if not exist "venv\Scripts\activate.bat" (
    echo [run_all] No virtual environment found in .\venv
    echo [run_all] Create it first:  python -m venv venv
    echo [run_all] then:             venv\Scripts\activate  and  pip install -r requirements.txt
    exit /b 1
)
call "venv\Scripts\activate.bat"
python run_all.py %*
exit /b %errorlevel%
