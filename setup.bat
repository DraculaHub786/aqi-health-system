@echo off
REM Quick Setup Script for Windows

echo ================================================
echo  AQI Health System - Quick Setup
echo ================================================
echo.

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.9+ from https://www.python.org/
    pause
    exit /b 1
)

echo [1/5] Python found!
echo.

REM Create virtual environment
echo [2/5] Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

REM Activate virtual environment
echo [3/5] Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo [4/6] Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo [5/6] Installing dependencies (this may take a few minutes)...
pip install -r requirements.txt

REM Download TextBlob corpora for NLP
echo [6/6] Downloading NLP corpora...
python -m textblob.download_corpora

echo.
echo ================================================
echo  Setup Complete!
echo ================================================
echo.
echo To run the application:
echo   1. Activate virtual environment: venv\Scripts\activate
echo   2. Run: streamlit run streamlit_app.py
echo.
echo Optional: Set up API keys in .env file for real data
echo.
pause
