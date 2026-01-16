#!/bin/bash
# Quick Setup Script for Linux/Mac

echo "================================================"
echo " AQI Health System - Quick Setup"
echo "================================================"
echo ""

# Check Python installation
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.9+ from https://www.python.org/"
    exit 1
fi

echo "[1/5] Python found!"
echo ""

# Create virtual environment
echo "[2/5] Creating virtual environment..."
python3 -m venv venv
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create virtual environment"
    exit 1
fi

# Activate virtual environment
echo "[3/5] Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "[4/6] Upgrading pip..."
python -m pip install --upgrade pip

# Install dependencies
echo "[5/6] Installing dependencies (this may take a few minutes)..."
pip install -r requirements.txt

# Download TextBlob corpora for NLP
echo "[6/6] Downloading NLP corpora..."
python -m textblob.download_corpora

echo ""
echo "================================================"
echo " Setup Complete!"
echo "================================================"
echo ""
echo "To run the application:"
echo "  1. Activate virtual environment: source venv/bin/activate"
echo "  2. Run: streamlit run streamlit_app.py"
echo ""
echo "Optional: Set up API keys in .env file for real data"
echo ""
