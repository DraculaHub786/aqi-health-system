"""
Google Colab Optimized Deployment Guide
Copy and paste these cells into Google Colab in order
"""

# ============================================
# CELL 1: Clone Repository
# ============================================
"""
!rm -rf aqi-health-system
!git clone https://github.com/DraculaHub786/aqi-health-system.git
%cd aqi-health-system
"""

# ============================================
# CELL 2: Clean NLTK Data (Critical Fix)
# ============================================
"""
import shutil
shutil.rmtree('/root/nltk_data', ignore_errors=True)
shutil.rmtree('/content/nltk_data', ignore_errors=True)
print('✅ NLTK data cleaned')
"""

# ============================================
# CELL 3: Install Core Dependencies
# ============================================
"""
# Install lightweight packages first
!pip install -q streamlit plotly pandas numpy scikit-learn
!pip install -q vaderSentiment textblob nltk pydantic cachetools
!pip install -q python-dateutil pyyaml requests python-dotenv
print('✅ Core packages installed')
"""

# ============================================
# CELL 4: Install AI/ML Packages (This may take 3-5 minutes)
# ============================================
"""
# Install transformers and torch (large packages)
!pip install -q transformers torch sentencepiece sentence-transformers
!pip install -q accelerate optimum
print('✅ AI/ML packages installed')
"""

# ============================================
# CELL 5: Download NLTK Data
# ============================================
"""
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')
print('✅ NLTK data downloaded')
"""

# ============================================
# CELL 6: Setup Environment & Test NLP Models
# ============================================
"""
import os
os.makedirs('data/kaggle', exist_ok=True)

# Test if transformers are available
print('\\n🔍 Testing NLP Models...')
try:
    from transformers import pipeline
    import torch
    print('✅ Transformers available')
    print(f'   PyTorch version: {torch.__version__}')
    print(f'   Device: {"cuda" if torch.cuda.is_available() else "cpu"}')
    
    # Test loading a small model
    print('\\n📥 Testing model loading (this may take 1-2 minutes)...')
    classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    result = classifier("This is a test")
    print(f'✅ Model loaded successfully! Test result: {result}')
    
except Exception as e:
    print(f'❌ Error: {e}')
    print('⚠️ Advanced NLP may not work. The app will use fallback mode.')
"""

# ============================================
# CELL 7: Install pyngrok for Public URL
# ============================================
"""
!pip install -q pyngrok
print('✅ pyngrok installed')
"""

# ============================================
# CELL 8: Configure ngrok (REPLACE WITH YOUR TOKEN)
# ============================================
"""
# Get your free token from: https://dashboard.ngrok.com/get-started/your-authtoken
NGROK_AUTH_TOKEN = "YOUR_NGROK_TOKEN_HERE"

from pyngrok import ngrok
ngrok.set_auth_token(NGROK_AUTH_TOKEN)
print('✅ ngrok configured')
"""

# ============================================
# CELL 9: Start Streamlit App
# ============================================
"""
import subprocess
import time
from pyngrok import ngrok

print('🚀 Starting Streamlit app...')

# Kill any existing streamlit processes
!pkill -f streamlit

# Start Streamlit in background
proc = subprocess.Popen(
    [
        "streamlit", "run", "streamlit_app.py",
        "--server.headless=true",
        "--server.address=0.0.0.0",
        "--server.port=8501",
        "--logger.level=warning",
    ],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)

# Wait for app to start
time.sleep(10)

# Create public URL
public_url = ngrok.connect(8501, "http")

print("\\n" + "="*60)
print("🎉 YOUR APP IS LIVE!")
print("="*60)
print(f"📱 Public URL: {public_url}")
print("="*60)
print("✨ Click the link above to access your app!")
print("⏱️  App stays live while this Colab session is running")
print("🔄 To stop: Run CELL 10")
print("="*60 + "\\n")

# Keep checking if app is running
import threading
def check_app():
    while True:
        if proc.poll() is not None:
            print("⚠️ App stopped unexpectedly!")
            break
        time.sleep(30)

thread = threading.Thread(target=check_app, daemon=True)
thread.start()
"""

# ============================================
# CELL 10: Stop the App
# ============================================
"""
!pkill -f streamlit
!pkill -f ngrok
print('✅ App stopped')
"""

# ============================================
# TROUBLESHOOTING TIPS
# ============================================
"""
COMMON ISSUES:

1. "Health tips not showing"
   - This means NLP models didn't load
   - Re-run CELL 6 to test model loading
   - Check if you have enough RAM (use Runtime > Change runtime type > High-RAM if available)

2. "Chat responses are generic"
   - Advanced NLP models not loaded (RAM issue or download failed)
   - The app falls back to basic keyword matching
   - Try restarting runtime and running all cells again

3. "Historical data not showing"
   - This is normal on first run
   - Check AQI for the same location 2-3 times
   - Data will appear after multiple checks in the same session

4. "ngrok authentication failed"
   - You need to set your ngrok authtoken in CELL 8
   - Get free token from https://dashboard.ngrok.com

5. "Out of memory"
   - Close other Colab notebooks
   - Use Runtime > Factory reset runtime
   - Re-run all cells from the beginning

TIPS FOR BEST PERFORMANCE:
- Use high-RAM runtime if available
- Don't open multiple browser tabs with the app
- If responses are slow, the models are working (be patient)
- First query may take 5-10 seconds (model loading)
"""
