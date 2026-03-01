@echo off
setlocal

set "PROJECT_ROOT=%~dp0"
cd /d "%PROJECT_ROOT%"

echo [run_app] Starting RAG application...

if not exist ".venv\Scripts\python.exe" (
    echo [run_app] Virtual environment not found. Creating .venv ...
    where py >nul 2>nul
    if %errorlevel%==0 (
        py -3 -m venv .venv
    ) else (
        python -m venv .venv
    )
    if errorlevel 1 goto :error
)

echo [run_app] Activating virtual environment...
call ".venv\Scripts\activate.bat"
if errorlevel 1 goto :error

python -m pip show streamlit >nul 2>nul
if errorlevel 1 (
    echo [run_app] Dependencies are not installed in .venv.
    choice /c YN /m "Install dependencies from requirements.txt now?"
    if errorlevel 2 goto :deps_missing
    python -m pip install -r requirements.txt
    if errorlevel 1 goto :error
)

echo [run_app] Starting backend on http://127.0.0.1:8000 ...
start "RAG-Backend-Server" cmd /k "cd /d ""%PROJECT_ROOT%"" && call .venv\Scripts\activate.bat && python -m uvicorn src.backend.server:app --host 127.0.0.1 --port 8000"

echo [run_app] Waiting for backend startup...
timeout /t 8 /nobreak > nul

echo [run_app] Starting frontend on http://localhost:8501 ...
python -m streamlit run src/app.py
if errorlevel 1 goto :error

exit /b 0

:deps_missing
echo.
echo [run_app] Install dependencies first:
echo     .venv\Scripts\python.exe -m pip install -r requirements.txt
echo Press any key to close this window.
pause > nul
exit /b 1

:error
echo.
echo [run_app] Startup failed. Exit code: %errorlevel%
echo Press any key to close this window.
pause > nul
exit /b %errorlevel%
