# Streamlit Community Cloud Deployment Guide

## Overview
Streamlit Community Cloud is a free, serverless platform for deploying Streamlit applications. This guide walks you through deploying the AQI Health System to the cloud.

---

## Prerequisites

### 1. **GitHub Repository** (Required)
- Your project must be in a GitHub repository
- The repository must be **public** (free tier requirement)
- Steps:
  - Create a GitHub account if you don't have one: https://github.com/signup
  - Create a new repository on GitHub
  - Push your local code to GitHub:
    ```bash
    git init
    git add .
    git commit -m "Initial commit"
    git remote add origin https://github.com/YOUR_USERNAME/aqi-health-system.git
    git push -u origin main
    ```

### 2. **Streamlit Account**
- Sign up at: https://share.streamlit.io
- Click **"Sign up"** and authenticate with your GitHub account

### 3. **Project Files Ready**
- ✅ `streamlit_app.py` - Your main application file
- ✅ `requirements.txt` - All dependencies listed
- ✅ `.gitignore` - To exclude unnecessary files (recommended)

---

## Step-by-Step Deployment

### Step 1: Prepare Your Repository

**1a. Create/Update .gitignore**
```
__pycache__/
*.pyc
.streamlit/secrets.toml
.env
*.pkl
*.joblib
venv/
.venv/
data/kaggle/*.csv
models/cache/
.DS_Store
```

**1b. Create .streamlit/config.toml** (Optional but recommended)
```ini
[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[client]
showErrorDetails = true

[logger]
level = "info"

[server]
maxUploadSize = 200  # MB
headless = true
enableXsrfProtection = true
```

**1c. Create .streamlit/secrets.toml** (For private credentials)
```toml
# Add any API keys or secrets here
# This file is NOT pushed to GitHub (add to .gitignore)
kaggle_username = "your_kaggle_username"
kaggle_key = "your_kaggle_api_key"
```

### Step 2: Verify requirements.txt

Ensure your `requirements.txt` includes all dependencies:
```
streamlit>=1.31.0
streamlit-option-menu>=0.3.12
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
joblib>=1.3.0
vaderSentiment>=3.3.2
textblob>=0.18.0
nltk>=3.8.0
transformers>=4.35.0
torch>=2.1.0
sentencepiece>=0.1.99
accelerate>=0.25.0
sentence-transformers>=2.2.2
plotly>=5.18.0
requests>=2.31.0
python-dotenv>=1.0.0
pyyaml>=6.0.1
python-dateutil>=2.8.0
kaggle>=1.6.0
pydantic>=2.5.0
cachetools>=5.3.0
```

### Step 3: Deploy to Streamlit Community Cloud

**3a. Go to Streamlit Community Cloud**
- Visit: https://share.streamlit.io
- Click **"New app"** button (top-right)

**3b. Fill in Deployment Details**

| Field | Value |
|-------|-------|
| **Repository** | `your-username/aqi-health-system` |
| **Branch** | `main` |
| **Main file path** | `streamlit_app.py` |

**3c. Click "Deploy!"**
- Streamlit will automatically:
  - Clone your repository
  - Install dependencies from `requirements.txt`
  - Run your `streamlit_app.py`
  - Provide you with a public URL

---

## Configuration & Secrets Management

### Setting Environment Variables in Community Cloud

**For sensitive data (API keys, credentials):**

1. Go to your app's **Settings** (gear icon ⚙️)
2. Click **"Secrets"** tab
3. Add secrets in TOML format:
   ```toml
   kaggle_username = "your_kaggle_username"
   kaggle_key = "your_kaggle_api_key"
   ```
4. Click **"Save"**

**Access secrets in your code:**
```python
import streamlit as st

kaggle_username = st.secrets["kaggle_username"]
kaggle_key = st.secrets["kaggle_key"]
```

---

## Repository Structure for Cloud Deployment

```
your-repo/
├── streamlit_app.py          # Main application (REQUIRED)
├── requirements.txt          # Dependencies (REQUIRED)
├── .streamlit/
│   ├── config.toml          # Optional: UI configuration
│   └── secrets.toml         # Optional: Local secrets only (.gitignored)
├── .gitignore               # Exclude unnecessary files
├── models/
│   ├── __init__.py
│   ├── advanced_ml.py
│   ├── conversational_ai.py
│   ├── nlp_engine.py
│   └── __pycache__/
├── utils/
│   ├── __init__.py
│   ├── config.py
│   ├── data_manager.py
│   ├── kaggle_dataset.py
│   └── __pycache__/
├── static/
│   ├── css/
│   │   └── styles.css
│   └── js/
│       └── app.js
├── templates/
│   └── index.html
├── data/
│   └── kaggle/
│       └── global_air_pollution.csv
└── README.md
```

---

## Common Issues & Solutions

### Issue 1: App Fails to Load (Model Download Timeout)
**Problem:** Transformers/PyTorch models take too long to download

**Solutions:**
```python
# Option A: Use smaller models
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')  # 22MB instead of 400MB

# Option B: Add caching
import streamlit as st
@st.cache_resource
def load_model():
    return SentenceTransformer('model-name')

model = load_model()

# Option C: Lazy load on first use
if 'model' not in st.session_state:
    with st.spinner("Loading model (one-time)..."):
        st.session_state.model = load_model()
```

### Issue 2: Memory Errors
**Problem:** App runs out of memory (Community Cloud has 1GB free tier limit)

**Solutions:**
- Reduce dataset size
- Use data sampling for initial load
- Implement pagination for large datasets
- Use `@st.cache_data(ttl=3600)` for caching

### Issue 3: Missing Dependencies
**Problem:** ModuleNotFoundError after deployment

**Solution:**
```bash
# Regenerate requirements.txt
pip freeze > requirements.txt

# Or for cleaner output (recommended):
pipdeptree
```

### Issue 4: Kaggle Dataset Not Loading
**Problem:** Kaggle API credentials missing

**Solution:**
1. Add to **Streamlit Cloud Secrets**:
   ```toml
   kaggle_username = "your_username"
   kaggle_key = "your_api_key"
   ```

2. Update your code:
   ```python
   import streamlit as st
   from kaggle.api.kaggle_api_extended import KaggleApi
   
   if 'kaggle' not in st.session_state:
       api = KaggleApi()
       api.authenticate(
           username=st.secrets.get("kaggle_username"),
           key=st.secrets.get("kaggle_key")
       )
       st.session_state.kaggle = api
   ```

---

## Performance Optimization for Cloud

### 1. **Caching**
```python
import streamlit as st

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_data():
    return pd.read_csv('data.csv')

@st.cache_resource
def load_model():
    return SentenceTransformer('model-name')
```

### 2. **Lazy Loading**
```python
if 'initialized' not in st.session_state:
    with st.spinner("Loading models..."):
        # Heavy initialization here
    st.session_state.initialized = True
```

### 3. **Optimize requirements.txt**
```
# Remove development-only packages
# Use CPU-only PyTorch if GPU not needed:
torch-cpu>=2.1.0

# Or use ONNX for faster inference:
onnx>=1.14.0
onnxruntime>=1.16.0
```

### 4. **Stream Output**
```python
# Instead of processing everything then displaying
for item in large_dataset:
    result = process(item)
    st.write(result)  # Stream updates
```

---

## Monitoring & Maintenance

### Check App Status
1. Go to https://share.streamlit.io
2. Click on your app
3. View **Logs** tab for errors

### Update Your App
1. Commit and push changes to GitHub:
   ```bash
   git add .
   git commit -m "Update feature"
   git push origin main
   ```
2. Streamlit Cloud automatically redeploys (usually within seconds)

### Reboot App
- Click **"Reboot app"** in app settings if needed

### View Statistics
- Memory usage
- CPU usage
- Request count
- All available in **Settings**

---

## Free Tier Limitations

| Resource | Limit |
|----------|-------|
| **RAM** | 1 GB |
| **Storage** | ~1 GB |
| **CPU** | Shared |
| **Idle Timeout** | App sleeps after 7 days of inactivity |
| **Build Time** | ~5 minutes |
| **Concurrent Users** | Limited (burst supported) |
| **Repository** | Must be public |

### Upgrade to Pro
- **$9/month** - 3 GB RAM per app, priority support
- Visit settings to upgrade

---

## Next Steps

1. ✅ Push code to GitHub
2. ✅ Sign up at https://share.streamlit.io
3. ✅ Click "New app" and fill in details
4. ✅ Wait for deployment (2-5 minutes)
5. ✅ Share your app URL with others

---

## Useful Links

- 📚 [Streamlit Docs](https://docs.streamlit.io)
- 🚀 [Deploy Guide](https://docs.streamlit.io/deploy/streamlit-community-cloud)
- 🔐 [Secrets Management](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management)
- 💬 [Community Forum](https://discuss.streamlit.io)
- 🐛 [Issues & Support](https://github.com/streamlit/streamlit/issues)

---

## Example Deployment URL Format

After successful deployment, your app will be available at:
```
https://your-username-aqi-health-system-xxxxx.streamlit.app
```

Share this URL with anyone to let them access your app! 🎉

---

## Quick Reference: Local Testing Before Cloud Deployment

Test your app locally first:
```bash
# Install dependencies
pip install -r requirements.txt

# Run Streamlit locally
streamlit run streamlit_app.py

# App opens at: http://localhost:8501
```

Once working locally, it will work on Streamlit Cloud!
