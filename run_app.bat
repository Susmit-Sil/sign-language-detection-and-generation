@echo off
echo.
echo       GenAI ASL Communicator - App Launcher               
echo.

:: ── Use the signlang_venv (has compatible NumPy + PyTorch + ultralytics) ────
set VENV=%~dp0RealTimeObjectDetection\signlang_venv

echo  [1/3] Activating virtual environment (signlang_venv)...
call "%VENV%\Scripts\activate.bat"

:: ── Install ollama Python client if missing (needed for Text→ASL translator) 
echo  [2/3] Checking dependencies...
python -c "import ollama" 2>nul
if errorlevel 1 (
    echo  Installing ollama client ^(first-time setup^)...
    pip install ollama --quiet
) else (
    echo  All dependencies satisfied.
)

:: ── Launch the unified Streamlit app from the Hackathon root ─────────────────
echo  [3/3] Launching app...
echo.
echo  Open your browser at: http://localhost:8501
echo  Press Ctrl+C to stop.
echo.
streamlit run "%~dp0app.py"

pause
