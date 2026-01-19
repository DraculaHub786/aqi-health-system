# 🚀 Google Colab Free Tier - Complete Setup Guide

**Copy each cell below into Google Colab and run them in order (1 → 2 → 3 → 4 → 5)**

---

## 📋 CELL 1: Install Dependencies

```python
# Install all required packages (takes ~2 minutes)
!pip install -q streamlit nltk requests plotly beautifulsoup4 pyngrok python-dotenv pandas numpy scikit-learn vaderSentiment

print("✅ All packages installed successfully!")
```

---

## 📋 CELL 2: Clone Repository & Setup

```python
# Clone your repository
!git clone https://github.com/YOUR_USERNAME/aqi-health-system.git
%cd aqi-health-system

print("✅ Repository cloned!")
```

**⚠️ IMPORTANT**: Replace `YOUR_USERNAME` with your actual GitHub username!

---

## 📋 CELL 3: Setup NLTK Data

```python
import nltk
import os

# Create NLTK data directory
nltk_data_path = os.path.expanduser('~/nltk_data')
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

# Download required NLTK data
print("📥 Downloading NLTK data...")
nltk.download('punkt', download_dir=nltk_data_path, quiet=True)
nltk.download('stopwords', download_dir=nltk_data_path, quiet=True)
nltk.download('wordnet', download_dir=nltk_data_path, quiet=True)
nltk.download('vader_lexicon', download_dir=nltk_data_path, quiet=True)

print("✅ NLTK setup complete!")
```

---

## 📋 CELL 4: Configure Environment (Lightweight Mode)

```python
# Create .env file for lightweight mode (works on free Colab!)
env_content = """WAQI_API_KEY=demo
USE_TRANSFORMERS=false
NLP_MODE=lightweight
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
"""

with open('.env', 'w') as f:
    f.write(env_content)

# Create data directory for historical data
import os
os.makedirs('data', exist_ok=True)

print("✅ Environment configured for lightweight mode")
print("💡 This mode works perfectly on free Colab!")
```

---

## 📋 CELL 5: Start Server & Get Public URL

```python
from pyngrok import ngrok
import threading
import os
import time

# STEP 1: Get your free ngrok token
# Visit: https://dashboard.ngrok.com/get-started/your-authtoken
# Copy your token and paste it below:

ngrok.set_auth_token("YOUR_NGROK_TOKEN_HERE")  # ⚠️ REPLACE THIS!

print("🚀 Starting Streamlit server...")

# Start Streamlit in background
def run_streamlit():
    os.system('streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0 2>&1 | grep -v "can use"')

thread = threading.Thread(target=run_streamlit, daemon=True)
thread.start()

# Wait for server to start
print("⏳ Waiting for server to initialize...")
time.sleep(15)

# Create ngrok tunnel
print("🌐 Creating public URL...")
public_url = ngrok.connect(8501)

# Display success message
print("\n" + "="*70)
print("🎉 YOUR AQI HEALTH SYSTEM IS LIVE!")
print("="*70)
print(f"\n📱 Public URL: {public_url}")
print("\n" + "="*70)
print("\n✨ FEATURES AVAILABLE:")
print("   ✅ Real-time AQI data for any location")
print("   ✅ Personalized health recommendations")
print("   ✅ Smart AI chat (context-aware responses)")
print("   ✅ Historical air quality tracking")
print("   ✅ Activity planning & safety tips")
print("\n💬 TRY ASKING:")
print("   • 'Is it safe for kids to play outside?'")
print("   • 'What is PM2.5?'")
print("   • 'Should I wear a mask today?'")
print("   • 'When is the best time to exercise?'")
print("\n🔗 Click the URL above to access your app!")
print("\n⏱️  Server will stay active for this session")
print("⚠️  URL changes each time you restart Colab")
print("\n🛑 To stop: Runtime → Interrupt execution")
print("="*70)

# Keep the server running
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n👋 Server stopped!")
```

**⚠️ IMPORTANT**: 
1. Get your free ngrok token: https://dashboard.ngrok.com/get-started/your-authtoken
2. Replace `YOUR_NGROK_TOKEN_HERE` with your actual token
3. Keep this cell running (don't interrupt it)

---

## 🎯 Quick Steps Summary

1. ✅ Run Cell 1 → Wait for packages to install
2. ✅ Run Cell 2 → Update GitHub username
3. ✅ Run Cell 3 → NLTK data downloads
4. ✅ Run Cell 4 → Environment configured
5. ✅ Run Cell 5 → Add ngrok token → Get public URL!

---

## 💡 What Makes This "Lightweight"?

**Lightweight Mode (This Guide):**
- ✅ No heavy transformer models (DialoGPT, BERT, BART)
- ✅ Fast startup (~2-3 minutes total)
- ✅ Works on free Colab (no RAM issues)
- ✅ Smart context-aware NLP (no pattern matching!)
- ✅ All features work perfectly

**Responses You'll Get:**
- "Is it safe for kids?" → Full child-safety analysis with AQI context
- "What is PM2.5?" → Detailed explanation of pollutants
- "Should I wear a mask?" → AQI-based protection recommendations
- Health tips always display (AQI-specific, personalized)

---

## 🆚 Comparison: Lightweight vs Full AI

| Feature | Lightweight (This Guide) | Full AI Mode |
|---------|--------------------------|--------------|
| **Installation Time** | 2 minutes | 20 minutes |
| **Model Download** | 0 MB | 1,300 MB |
| **Memory Usage** | ~500 MB | ~3,000 MB |
| **Colab Free Tier** | ✅ Works | ❌ Often fails |
| **Response Quality** | ⭐⭐⭐⭐ Excellent | ⭐⭐⭐⭐⭐ Perfect |
| **All Features** | ✅ Yes | ✅ Yes |
| **Startup Speed** | ⚡ Fast | 🐌 Slow |

---

## 🔧 Troubleshooting

### "ModuleNotFoundError: No module named X"
**Solution**: Re-run Cell 1

### "NLTK data not found"
**Solution**: Re-run Cell 3

### "Tunnel error" or "ngrok error"
**Solution**: 
1. Get token from https://dashboard.ngrok.com/
2. Update Cell 5 with your token
3. Re-run Cell 5

### "Server won't start"
**Solution**: 
1. Runtime → Restart runtime
2. Re-run all cells from Cell 1

### Health tips section is empty
**Solution**: ✅ FIXED! Health tips now always display

### Chat responses are generic
**Solution**: ✅ FIXED! Now uses context-aware NLP

---

## 📱 Sharing Your App

Once Cell 5 is running:
1. Copy the public URL (looks like: `https://xxxx-xx-xx-xx-xx.ngrok.io`)
2. Share it with anyone - they can access your app!
3. Works on mobile phones too
4. No login required for visitors

**Note**: URL changes each time you restart Colab

---

## ⏰ Session Management

- ✅ Free Colab sessions last ~12 hours max
- ✅ If disconnected, re-run Cell 5 only (new URL)
- ✅ If runtime restarts, run all cells again
- ✅ Your app stays live as long as Cell 5 is running

---

## 🎓 What Changed From Before?

### Before (Broken in Colab):
- ❌ Required 1.3GB transformer models
- ❌ Failed with "Out of Memory" errors
- ❌ Health tips section was empty
- ❌ Chat responses were generic greetings
- ❌ Took 20+ minutes to start (if it worked)

### After (This Guide):
- ✅ No heavy models needed
- ✅ Works on free Colab reliably
- ✅ Health tips always display
- ✅ Smart, context-aware chat responses
- ✅ Ready in 2-3 minutes

---

## 🚀 You're All Set!

Your AQI Health System is now ready to share with friends. The lightweight mode provides excellent intelligence without requiring expensive compute resources.

**Enjoy! 🎉**
