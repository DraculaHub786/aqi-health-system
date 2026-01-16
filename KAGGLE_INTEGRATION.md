# 📊 Kaggle Data Integration

## What It Does

Brings real-world pollution data into your app for evidence-based recommendations.

You Get:
- Global air quality datasets
- WHO health guidelines
- City-specific patterns
- Pollutant impact data

Works Without Kaggle API: Automatically creates sample data if you don't have an account.

---

## Setup (2 Ways)

### Option 1: Quick Start (No Account)
```bash
python setup_kaggle.py
```
Creates sample data automatically. You're ready!

### Option 2: Real Kaggle Data (Recommended)

Step 1: Get Kaggle API Key
1. Create free account: [kaggle.com](https://www.kaggle.com)
2. Go to Account → API → Create Token
3. Download `kaggle.json`

Step 2: Place Key File
- Windows: `C:\\Users\\<you>\\.kaggle\\kaggle.json`
- Mac/Linux: `~/.kaggle/kaggle.json`

Step 3: Run Setup
```bash
python setup_kaggle.py
```

Downloads real datasets (~50MB). Takes 2-3 minutes.

---

## What You Get

### Datasets Included
1. Global Air Pollution - Worldwide AQI data
2. City AQI History - Major cities tracking
3. US Pollution Trends - Detailed 20+ year data

### Recommendations Database
- AQI Ranges: 6 categories (Good → Hazardous)
- Pollutant Info: PM2.5, PM10, NO2, O3, CO, SO2
- Health Impacts: WHO-backed guidelines
- Activity Thresholds: Evidence-based limits
- Seasonal Patterns: Winter, Spring, Summer, Fall

---

## Using in Code

```python
from utils.kaggle_dataset import KaggleAQIDataset

# Initialize
manager = KaggleAQIDataset()

# Get recommendations
recs = manager.get_recommendations_for_aqi(
    aqi=125,
    pollutants={'PM2.5': 55.5}
)

print(f"Category: {recs['category']}")
print(f"Activities: {recs['activities']}")
print(f"Precautions: {recs['precautions']}")
```

---

## Benefits

Before Kaggle:
- Template responses
- Generic advice
- No real data

After Kaggle:
- ✅ Real pollution patterns
- ✅ Evidence-based recommendations
- ✅ WHO health data
- ✅ City-specific insights
- ✅ Pollutant-specific advice

---

## Troubleshooting

"Kaggle API not found"  
→ Works fine! App uses sample data automatically.

Want real data?  
→ Follow Option 2 setup above

Permission error (Mac/Linux)?  
```bash
chmod 600 ~/.kaggle/kaggle.json
```

---

## Sample vs Real Data

| Feature | Sample | Real Kaggle |
|---------|--------|-------------|
| Works offline | ✅ | ✅ (after download) |
| File size | ~10KB | ~50MB |
| Cities covered | 10 | 1000+ |
| Historical data | Basic | 20+ years |
| Updates | Manual | Re-download |

Both work great! Sample data perfect for testing and demos.

---

## Advanced

### Update Datasets
```bash
python -c "from utils.kaggle_dataset import *; m=KaggleAQIDataset(); m.download_all_datasets(force=True)"
```

### View Database
```python
import pickle
with open('data/kaggle/recommendations_db.pkl', 'rb') as f:
    db = pickle.load(f)
    print(db.keys())
```

---

Questions? Just ask the AI assistant in the app! 💬
