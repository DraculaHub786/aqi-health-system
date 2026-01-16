#!/bin/bash
# Run AQI Health Application

echo "Starting AQI Health & Activity Planner..."
echo ""

# Check if virtual environment exists
if [ ! -f "venv/bin/activate" ]; then
    echo "ERROR: Virtual environment not found!"
    echo "Please run setup.sh first"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Run Streamlit application
streamlit run streamlit_app.py
