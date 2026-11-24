@echo off
rem Starts the Streamlit dashboard on Windows: activates the venv, forces UTF-8 mode and opens http://localhost:8501
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
if not exist "venv\Scripts\activate.bat" (
    echo [run_app] No virtual environment found in .\venv
    echo [run_app] Create it first:  python -m venv venv
    echo [run_app] then:             venv\Scripts\activate  and  pip install -r requirements.txt
    exit /b 1
)
call "venv\Scripts\activate.bat"
streamlit run app.py %*
exit /b %errorlevel%
