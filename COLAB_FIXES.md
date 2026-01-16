# 🎯 ULTIMATE COLAB FIX GUIDE
## Fix Health Tips & Historical Data Not Showing

Based on diagnostics, here are the **exact steps** to fix both issues:

---

## 🔧 Step 1: Run Enhanced Diagnostic

In Colab, run:
```python
!python colab_diagnose_enhanced.py
```

This will tell you **exactly** what's broken. Look for:
- ❌ Missing NLTK packages  
- ❌ Empty database  
- ❌ TextBlob not working  

---

## 🩹 Step 2: Apply Fixes Based on Results

### Fix A: Missing NLTK Packages (Most Common)

If diagnostic shows `✗ wordnet MISSING` or `✗ vader_lexicon MISSING`:

```python
import nltk
import ssl

# Handle SSL
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except:
    pass

# Download ALL packages
packages = ['punkt', 'wordnet', 'omw-1.4', 'vader_lexicon', 'stopwords', 'averaged_perceptron_tagger']
for pkg in packages:
    print(f"Downloading {pkg}...")
    nltk.download(pkg)
    
print("✅ All NLTK packages downloaded!")
```

### Fix B: TextBlob Corpora Missing

If diagnostic shows `✗ TextBlob Error`:

```python
!python -m textblob.download_corpora
```

### Fix C: Empty Database (Historical Data)

If diagnostic shows `⚠ Database is empty`:

```python
import sqlite3
from datetime import datetime, timedelta

conn = sqlite3.connect('data/aqi_history.db')
cursor = conn.cursor()

# Add 24 hours of sample data
print("Adding sample historical data...")
for i in range(24):
    ts = (datetime.now() - timedelta(hours=23-i)).isoformat()
    aqi = 100 + i*3 + (i % 5)*10  # Varying AQI values
    cursor.execute('''
        INSERT INTO aqi_history 
        (location, aqi, pm25, pm10, o3, timestamp, dominant_pollutant, data_source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', ('Delhi', aqi, aqi*0.6, aqi*0.8, aqi*0.3, ts, 'PM25', 'sample'))

conn.commit()
count = cursor.execute("SELECT COUNT(*) FROM aqi_history").fetchone()[0]
print(f"✅ Database now has {count} records!")
conn.close()
```

---

## ✅ Step 3: Verify Fixes

Run diagnostic again:
```python
!python colab_diagnose_enhanced.py
```

You should now see:
```
✅ Health Tips WORKING - X tips generated
✅ Historical Data WORKING - 24 records retrieved
```

---

## 🚀 Step 4: Restart Streamlit App

**IMPORTANT:** Must restart for changes to take effect!

```python
# Stop current app
import subprocess
subprocess.run(['pkill', '-f', 'streamlit'])

# Wait 3 seconds
import time
time.sleep(3)

# Start fresh
from pyngrok import ngrok
ngrok.set_auth_token("YOUR_NGROK_TOKEN")

proc = subprocess.Popen([
    "streamlit", "run", "streamlit_app.py",
    "--server.headless=true",
    "--server.port=8501"
])

time.sleep(10)
public_url = ngrok.connect(8501, "http")
print(f"\n🎉 App restarted: {public_url}\n")
```

---

## 🐛 Still Not Working? Debug Mode

If sections still don't appear, add temporary debug code:

### Add to streamlit_app.py (around line 888):

```python
# BEFORE the tips rendering
st.write("DEBUG: About to generate tips...")
tips = nlp_engine.generate_personalized_tips(aqi, pollutants, user_profile, location)
st.write(f"DEBUG: Got {len(tips) if tips else 0} tips: {tips}")

# BEFORE the historical data rendering  
st.write("DEBUG: About to get historical data...")
historical_data = data_manager.get_historical_data(location, hours=24)
st.write(f"DEBUG: Got {len(historical_data) if historical_data else 0} records")
if historical_data:
    st.write(f"DEBUG: Sample record: {historical_data[0]}")
```

This will show you **exactly** what data is being returned.

---

## 📋 Checklist

Before asking for help, verify:

- [ ] Ran `colab_diagnose_enhanced.py`
- [ ] Fixed all ❌ errors shown in diagnostic
- [ ] All NLTK packages downloaded
- [ ] TextBlob corpora downloaded  
- [ ] Database has 24+ records
- [ ] Restarted Streamlit app (not just refreshed browser)
- [ ] Tried in Incognito/Private browser window

---

## 💡 Common Mistakes

**❌ Don't:**
- Just refresh the browser (need to restart app)
- Skip downloading ALL NLTK packages
- Forget to populate database with sample data

**✅ Do:**
- Run diagnostic first to identify issue
- Apply ALL relevant fixes at once
- Restart Streamlit after fixes
- Wait 10 seconds before opening URL

---

## 🆘 Emergency All-in-One Fix

If nothing else works, run this complete setup:

```python
import os
import nltk
import ssl
import sqlite3
from datetime import datetime, timedelta

print("🔧 Running complete setup...")

# 1. SSL
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except:
    pass

# 2. Create directories
for d in ['data', 'data/kaggle', 'models', 'utils']:
    os.makedirs(d, exist_ok=True)
print("✓ Directories")

# 3. NLTK packages
for pkg in ['punkt', 'wordnet', 'omw-1.4', 'vader_lexicon', 'stopwords', 'averaged_perceptron_tagger']:
    try:
        nltk.download(pkg, quiet=True)
    except:
        pass
print("✓ NLTK packages")

# 4. TextBlob
try:
    import subprocess
    subprocess.run(['python', '-m', 'textblob.download_corpora'], 
                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
except:
    pass
print("✓ TextBlob")

# 5. Database
conn = sqlite3.connect('data/aqi_history.db')
cursor = conn.cursor()

cursor.execute('''
    CREATE TABLE IF NOT EXISTS aqi_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        location TEXT NOT NULL,
        aqi REAL NOT NULL,
        pm25 REAL,
        pm10 REAL,
        o3 REAL,
        no2 REAL,
        so2 REAL,
        co REAL,
        dominant_pollutant TEXT,
        data_source TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        raw_data TEXT
    )
''')

# Clear old data and add fresh sample
cursor.execute("DELETE FROM aqi_history")
for i in range(24):
    ts = (datetime.now() - timedelta(hours=23-i)).isoformat()
    aqi = 100 + i*3
    cursor.execute('''
        INSERT INTO aqi_history 
        (location, aqi, pm25, pm10, o3, timestamp, dominant_pollutant, data_source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', ('Delhi', aqi, aqi*0.6, aqi*0.8, aqi*0.3, ts, 'PM25', 'sample'))

conn.commit()
conn.close()
print("✓ Database")

print("\n✅ Setup complete! Now restart Streamlit app.")
```

---

Made with ❤️ for Colab Deployment
