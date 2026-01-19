#!/usr/bin/env python3
"""
🚀 GOOGLE COLAB DEPLOYMENT - AQI Health System
===============================================

TWO DEPLOYMENT OPTIONS:

1. FULL AI MODE (Recommended - but needs high-RAM runtime)
   - All 5 transformer models (DialoGPT, BERT, BART, etc.)
   - Most advanced AI responses
   - Requires: Colab Pro or high-RAM runtime
   - Download: ~1.3GB
   - See CELL 1A below

2. LIGHTWEIGHT MODE (Works on free Colab!)
   - Smart context-aware NLP (no heavy models)
   - Fast startup, low memory
   - Still very intelligent responses
   - Works on free Colab
   - See CELL 1B below

Choose ONE option and run those cells!
"""

# ============================================================
# OPTION 1: FULL AI MODE (High-RAM runtime required)
# ============================================================

# === CELL 1A: Install Full Dependencies (FULL AI MODE) ===
# Uncomment the line below if you want FULL AI mode:
# !pip install -q streamlit nltk requests plotly beautifulsoup4 pyngrok python-dotenv pandas numpy scikit-learn transformers torch sentence-transformers sentencepiece accelerate vaderSentiment


# ============================================================
# OPTION 2: LIGHTWEIGHT MODE (Works on free Colab!)
# ============================================================

# === CELL 1B: Install Lightweight Dependencies (DEFAULT) ===
# Run this first - takes about 2 minutes
!pip install -q streamlit nltk requests plotly beautifulsoup4 pyngrok python-dotenv pandas numpy scikit-learn vaderSentiment

# === CELL 2: Clone Repository ===
!git clone https://github.com/YOUR_USERNAME/aqi-health-system.git
%cd aqi-health-system

# === CELL 3: Setup NLTK Data ===
import nltk
import os

nltk_data_path = os.path.expanduser('~/nltk_data')
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

print("📥 Downloading NLTK data...")
nltk.download('punkt', download_dir=nltk_data_path, quiet=True)
nltk.download('stopwords', download_dir=nltk_data_path, quiet=True)
nltk.download('wordnet', download_dir=nltk_data_path, quiet=True)
nltk.download('vader_lexicon', download_dir=nltk_data_path, quiet=True)
print("✅ NLTK setup complete!")

# === CELL 4: Environment Setup ===
# Create .env file with settings

# FOR LIGHTWEIGHT MODE (default):
env_content = """
# API Configuration
WAQI_API_KEY=demo

# NLP Configuration - LIGHTWEIGHT MODE
USE_TRANSFORMERS=false
NLP_MODE=lightweight

# Streamlit Configuration
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
"""

# UNCOMMENT BELOW FOR FULL AI MODE (if you ran Cell 1A):
# env_content = """
# # API Configuration
# WAQI_API_KEY=demo
# 
# # NLP Configuration - FULL AI MODE
# USE_TRANSFORMERS=true
# NLP_MODE=advanced
# 
# # Streamlit Configuration
# STREAMLIT_SERVER_PORT=8501
# STREAMLIT_SERVER_ADDRESS=0.0.0.0
# """

with open('.env', 'w') as f:
    f.write(env_content)

print("✅ Environment configured")
print(f"🔧 Mode: {'FULL AI' if 'USE_TRANSFORMERS=true' in env_content else 'LIGHTWEIGHT'}")

# === CELL 5: Start Server ===
from pyngrok import ngrok
import threading
import os
import time

# Set ngrok auth token (GET FREE TOKEN at https://ngrok.com/)
# ngrok.set_auth_token("YOUR_NGROK_TOKEN_HERE")

print("🚀 Starting Streamlit server...")

# Start Streamlit in background
def run_streamlit():
    os.system('streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0')

thread = threading.Thread(target=run_streamlit, daemon=True)
thread.start()

# Wait for server to start
time.sleep(10)

# Create ngrok tunnel
print("🌐 Creating public URL...")
public_url = ngrok.connect(8501)

print("\n" + "="*60)
print("🎉 YOUR APP IS LIVE!")
print("="*60)
print(f"📱 Public URL: {public_url}")
print("="*60)
print("\n⏱️ Server will stay alive for this session")
print("⚠️ URL changes each time you restart")
print("\n💡 Features:")
print("  ✅ Real-time AQI data")
print("  ✅ Personalized health tips")
print("  ✅ Smart NLP chat (context-aware responses)")
print("  ✅ Historical tracking")
print("\n🔍 Check the URL above to access your app!")
print("\n🛑 To stop: Runtime → Interrupt execution")

# Keep alive
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n👋 Server stopped")

"""
USAGE INSTRUCTIONS:
==================

1. Copy this entire file content
2. In Google Colab, create new cells and paste each section
3. Run cells in order (1 → 2 → 3 → 4 → 5)
4. Get ngrok token: https://dashboard.ngrok.com/get-started/your-authtoken
5. Uncomment and add your token in Cell 5
6. Access your app via the printed public URL!

LIGHTWEIGHT MODE BENEFITS:
==========================
✅ Much faster startup (no 1.3GB model download)
✅ Works in free Colab (doesn't need high-RAM)
✅ Still provides smart, context-aware responses
✅ All features work (health tips, chat, historical data)

NLP CAPABILITIES:
================
The lightweight mode uses:
- Semantic similarity matching (fast, accurate)
- Intent classification (rule-based + ML)
- Context-aware response generation
- AQI-specific knowledge base

Responses are still intelligent and contextual, just without
the heavy transformer models (DialoGPT, BERT, BART).

TROUBLESHOOTING:
===============
- "No module named X": Re-run Cell 1
- "NLTK data not found": Re-run Cell 3
- "Tunnel error": Add ngrok token in Cell 5
- Server won't start: Restart runtime and try again
"""
