# 🚀 Deploy on Google Colab

## What You Have

✅ App is running locally at `http://localhost:8501`

Perfect starting point for Colab deployment!

---

## Method 1: Deploy with Streamlit Cloud (Easiest - 5 min)

### Step 1: Push Code to GitHub
```bash
# Initialize git (if not already)
git init
git add .
git commit -m "AQI Health System ready for deployment"
git remote add origin https://github.com/YOUR_USERNAME/aqi-health-system.git
git push -u origin main
```

### Step 2: Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Select your GitHub repo
4. Set main file: `streamlit_app.py`
5. Click "Deploy"

That's it! Your app is live! 🎉

---

## Method 2: Deploy on Google Colab (Free - 10 min)

### Step 1: Create Colab Notebook

```python
# Cell 1: Install dependencies
!pip install streamlit streamlit-option-menu plotly pandas numpy scikit-learn joblib
!pip install vaderSentiment textblob nltk transformers torch kaggle python-dotenv pyyaml

# Download TextBlob corpora
!python -m textblob.download_corpora
```

### Step 2: Download Your Project
```python
# Cell 2: Clone or upload project
!git clone https://github.com/YOUR_USERNAME/aqi-health-system.git
%cd aqi-health-system
```

### Step 3: Install Streamlit Tunnel
```python
# Cell 3: Install pyngrok for public URL
!pip install pyngrok
```

### Step 4: Set Up Authentication (Optional but Recommended)
```python
# Cell 4: Set up ngrok (free public tunnel)
from pyngrok import ngrok
import subprocess

# Get free ngrok token from https://ngrok.com (free account)
# Then set your token:
ngrok.set_auth_token("YOUR_NGROK_TOKEN")

# Create tunnel
public_url = ngrok.connect(8501)
print(f"Public URL: {public_url}")
```

### Step 5: Run the App
```python
# Cell 5: Run Streamlit
import subprocess
import threading

def run_streamlit():
    subprocess.run(["streamlit", "run", "streamlit_app.py", "--server.headless=true"])

# Run in background
thread = threading.Thread(target=run_streamlit, daemon=True)
thread.start()

# Wait a moment for startup
import time
time.sleep(3)

# Print access URL
print("App is running!")
print(f"Access at: {public_url}")
```

---

## Method 3: Deploy on Heroku (Free Tier - 10 min)

### Step 1: Prepare for Heroku
```bash
# Create Procfile
echo "web: streamlit run streamlit_app.py --server.port \$PORT --server.address 0.0.0.0" > Procfile

# Create setup.sh
cat > setup.sh << 'EOF'
mkdir -p ~/.streamlit/
echo "[server]
headless = true
port = \$PORT
enableCORS = false
" > ~/.streamlit/config.toml
EOF
```

### Step 2: Deploy
```bash
# Install Heroku CLI from https://devcenter.heroku.com/articles/heroku-cli

# Login
heroku login

# Create app
heroku create your-app-name

# Deploy
git push heroku main

# View live
heroku open
```

---

## Method 4: Deploy on Hugging Face Spaces (Easiest)

### Step 1: Create Space
1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Click "Create new Space"
3. Name: `aqi-health-system`
4. Select Streamlit as space type
5. Click "Create"

### Step 2: Upload Files
1. Clone the space repo
2. Copy all your files
3. Push to repo:
```bash
git add .
git commit -m "Add AQI app"
git push
```

Auto-deploys! 🚀

---

## Quick Comparison

| Platform | Cost | Setup Time | URL Type | Best For |
|----------|------|-----------|----------|----------|
| Streamlit Cloud | Free | 5 min | Custom domain | Easiest |
| Colab | Free | 15 min | Public tunnel | Learning |
| Heroku | Free tier | 10 min | Custom domain | Production |
| Hugging Face | Free | 10 min | Custom | Community |

---

## ⚠️ Important Notes

### Free Tier Limitations:
- Streamlit Cloud: 1 GB memory, auto-sleeps after inactivity
- Colab: 12-hour session limit, but easy to restart
- Heroku: Free dyno goes to sleep after 30 min inactivity
- HF Spaces: Generous free tier, runs 24/7

### For Production:
- Upgrade Streamlit Cloud ($5-50/month)
- Use Heroku paid tier ($7+/month)
- Self-host on AWS/Google Cloud ($5+/month)

---

## My Recommendation

### For Quick Demo: Streamlit Cloud
```bash
# 3 commands, 5 minutes
git push origin main
# Then deploy via share.streamlit.io
```

### For Learning: Google Colab
- Great for experimenting
- Can modify code live
- Free and easy

### For Production: Heroku or HF Spaces
- More control
- Better uptime
- Custom domains

---

## Next Steps

1. Choose a platform above
2. Follow the setup
3. Share your public URL with others!

---

## Access Your Local App Now

While you decide on deployment:

🔗 Local: `http://localhost:8501`

Your app is running and ready to test! 🎉

---

## Troubleshooting

"Port already in use?"
```bash
streamlit run streamlit_app.py --server.port 8502
```

"Module not found?"
```bash
pip install -r requirements.txt
python -m textblob.download_corpora
```

"Still having issues?"
The AI in your app can help! Just ask it any questions. 💬

---

## Share Your App!

Once deployed, share the URL:
- GitHub: Link in README
- Social media: Tweet about it!
- Email: Send to friends

Let them breathe smarter! 🌍

---

Questions about deployment? Let me know which platform interests you most!
