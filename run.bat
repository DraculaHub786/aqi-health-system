@echo off
REM Run AQI Health Application

echo Starting AQI Health & Activity Planner...
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run setup.bat first
    pause
    exit /b 1
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Run Streamlit application
streamlit run streamlit_app.py

pause
