# 🚀 Quick Start Guide

## 60-Second Setup

Windows Users:
```cmd
1. Double-click setup.bat
2. Double-click run.bat
3. Open http://localhost:8501
```

Mac/Linux Users:
```bash
chmod +x setup.sh run.sh
./setup.sh
./run.sh
```

That's it! 🎉

---

## Using the App

### 1️⃣ Enter Your City
Type any city name → Click "Fetch AQI Data"

### 2️⃣ Personalize Your Profile
- Age slider
- Health conditions (if any)
- Activity level

### 3️⃣ Get Insights
- Current AQI & health risk
- 24-hour predictions
- Activity recommendations
- Ask any question!

---

## Ask Me Anything

Our AI responds to every question naturally:

💬 Examples:
- "Is it safe to run?"
- "What can I do today?"
- "How does pollution affect asthma?"
- "Best time to exercise?"
- Even casual chat: "Hi!", "Thanks!"

---

## Understanding Your Dashboard

### AQI Gauge (0-500)
- 🟢 0-50: Good - Enjoy outdoor activities!
- 🟡 51-100: Moderate - Most people are fine
- 🟠 101-150: Unhealthy for sensitive groups
- 🔴 151-200: Unhealthy - Limit outdoor time
- 🟣 201-300: Very unhealthy - Stay indoors
- ⚫ 301-500: Hazardous - Emergency conditions

### Pollutants Explained
- PM2.5: Fine particles (most harmful)
- PM10: Coarse particles (dust)
- O3: Ground-level ozone
- NO2: Traffic pollution
- SO2: Industrial emissions
- CO: Carbon monoxide

---

## Personalization Tips

Set Your Profile for Better Advice:

👶 Children: More protective recommendations  
👴 Elderly: Conservative activity limits  
🫁 Respiratory Conditions: Stricter air quality thresholds  
❤️ Heart Conditions: Enhanced precautions  
🏃 Active Lifestyle: Optimized workout timing

---

## Advanced Features

### 📊 Kaggle Data Integration
Run `python setup_kaggle.py` for:
- Real-world pollution datasets
- Evidence-based recommendations
- City-specific patterns
- WHO health guidelines

### 🔑 Live API Data
Add API keys to `.env`:
```
WAQI_API_KEY=your_token_here
```
Get free token: [aqicn.org/data-platform/token](https://aqicn.org/data-platform/token/)

---

## Troubleshooting

App won't start?
```bash
pip install -r requirements.txt
python -m textblob.download_corpora
```

No data showing?
- Works offline with sample data
- Add API key for live data
- Check your internet connection

Slow performance?
- First run downloads ML models (~150MB)
- Subsequent runs are fast

---

## Tips & Tricks

✨ Best Practices:
- Check AQI before morning exercise
- Set up your profile once for personalized tips
- Use the chat feature to ask specific questions
- Check hourly predictions for planning

🎯 Power User:
- Ask comparative questions: "Indoor or outdoor workout?"
- Request specific times: "Best time to exercise today?"
- Get pollutant details: "Tell me about PM2.5"

---

## Need Help?

Ask the AI assistant anything! It's designed to help with:
- Air quality questions
- Health concerns
- Activity planning
- Protection advice
- General information

Just type naturally - it understands you! 💬

---

Made with ❤️ for healthier living
- Bar chart showing levels of:
  - PM2.5: Fine particles
  - PM10: Coarse particles
  - O₃: Ozone
  - NO₂: Nitrogen dioxide
  - SO₂: Sulfur dioxide
  - CO: Carbon monoxide

#### Personalized Risk Assessment
- Risk Level: Low, Moderate, High, Very High, Severe
- Risk Score: 0-100 probability score
- Detailed Breakdown: Probability for each risk level

#### Smart Activity Recommendations
Top 6 recommended activities with:
- ✅ Safe: Good to go!
- ⚠️ Caution: Be careful
- ❌ Not Recommended: Avoid

Each includes:
- Activity name
- Safety level
- Recommendation text
- Match score

#### 24-Hour Forecast
- ML-predicted AQI for next 24 hours
- Confidence intervals (shaded area)
- Best times highlighted
- Interactive hover details

#### Personalized Health Tips
Up to 8 context-aware tips including:
- Emergency actions for severe pollution
- Protective measures (masks, air purifiers)
- Activity limitations
- Timing recommendations
- Medical advice for health conditions

#### Historical Trends
- AQI trend over last 24 hours
- PM2.5 and PM10 comparison
- Interactive time-series charts
- Builds up as you check regularly

#### Question Answering
Ask questions about air quality:
- "Is it safe to go jogging now?"
- "What is the dominant pollutant?"
- "When is the best time to exercise?"

Get AI-powered answers with confidence scores.

---

## Understanding AQI

### AQI Scale
| AQI | Level | Color | Meaning |
|-----|-------|-------|---------|
| 0-50 | Good | 🟢 Green | Perfect air quality |
| 51-100 | Moderate | 🟡 Yellow | Acceptable for most |
| 101-150 | Unhealthy for Sensitive | 🟠 Orange | Sensitive groups affected |
| 151-200 | Unhealthy | 🔴 Red | Everyone affected |
| 201-300 | Very Unhealthy | 🟣 Purple | Health alert |
| 301-500 | Hazardous | 🟤 Maroon | Health emergency |

### Pollutant Information

PM2.5 (Fine Particles)
- Size: < 2.5 micrometers
- Sources: Vehicle emissions, industrial processes, fires
- Health: Penetrates deep into lungs, enters bloodstream
- Safe level: < 12 µg/m³

PM10 (Coarse Particles)
- Size: < 10 micrometers
- Sources: Road dust, construction, agriculture
- Health: Respiratory irritation, aggravated asthma
- Safe level: < 54 µg/m³

O₃ (Ozone)
- Formation: Sunlight + pollutants
- Peak: Afternoon hours (2-6 PM)
- Health: Lung irritation, reduced function
- Safe level: < 54 ppb

NO₂ (Nitrogen Dioxide)
- Sources: Vehicle exhaust, power plants
- Health: Respiratory problems, reduced immunity
- Safe level: < 53 ppb

SO₂ (Sulfur Dioxide)
- Sources: Coal burning, industrial facilities
- Health: Breathing difficulties, asthma
- Safe level: < 35 ppb

CO (Carbon Monoxide)
- Sources: Incomplete combustion, vehicles
- Health: Reduces oxygen delivery to organs
- Safe level: < 4.4 ppm

---

## Tips for Different AQI Levels

### Good (0-50) ✅
- Perfect for all outdoor activities
- Great time for exercise and sports
- Ideal for children's outdoor play
- No restrictions needed

### Moderate (51-100) 🟡
- Generally safe for most people
- Sensitive individuals: limit prolonged exertion
- Good time for morning walks
- Consider timing for intense exercise

### Unhealthy for Sensitive Groups (101-150) 🟠
- Children, elderly, health conditions: reduce outdoor time
- General public: limit prolonged outdoor exertion
- Wear masks for extended outdoor activities
- Schedule exercise during better air quality periods

### Unhealthy (151-200) 🔴
- Everyone: reduce outdoor exposure
- Sensitive groups: avoid outdoor activities
- Wear N95 masks if going outside
- Keep windows closed
- Use air purifiers indoors

### Very Unhealthy (201-300) 🟣
- Everyone: avoid outdoor activities
- Stay indoors as much as possible
- Use HEPA air purifiers
- Keep emergency medications ready
- Monitor for symptoms

### Hazardous (301-500) 🟤
- HEALTH EMERGENCY
- Stay indoors, seal windows
- Use air purifiers on highest setting
- Avoid all physical exertion
- Contact doctor if experiencing symptoms
- Have emergency plan ready

---

## Advanced Features

### Force Refresh
- Bypasses 5-minute cache
- Fetches fresh data from APIs
- Use sparingly to avoid rate limits
- Located in Advanced Settings

### Debug Mode
- Shows raw API responses
- Displays internal calculations
- Helpful for troubleshooting
- View user profile JSON

### Historical Data
- Automatically tracks AQI over time
- Builds database of trends
- Compare different time periods
- Export for analysis

### Q&A System
- Powered by Hugging Face transformers
- Context-aware responses
- Confidence scoring
- Natural language understanding

---

## Best Practices

### For Daily Use
1. Check AQI every morning
2. Plan outdoor activities during low-AQI hours
3. Set up your profile once for personalized tips
4. Monitor trends to identify patterns
5. Share data with family members

### For Health Conditions
1. Update health status in profile
2. Follow personalized recommendations strictly
3. Have medications ready during poor AQI
4. Monitor symptoms closely
5. Consult doctor if concerned

### For Parents
1. Set profile to "Children"
2. Check before outdoor school activities
3. Limit play during unhealthy AQI
4. Teach kids about air quality
5. Keep indoor activities ready

### For Athletes
1. Time workouts during good AQI
2. Use indoor alternatives when unhealthy
3. Reduce intensity in moderate AQI
4. Stay hydrated
5. Monitor breathing during exercise

### For Outdoor Workers
1. Set outdoor worker profile
2. Take frequent breaks in clean air
3. Use N95 masks during poor AQI
4. Reduce exertion when possible
5. Know employer safety requirements

---

## Troubleshooting

### "Unable to fetch AQI data"
Solutions:
1. Check internet connection
2. Try different city name format
3. Wait a few minutes and retry
4. App will use realistic simulation as fallback

### Models loading slowly
Solutions:
1. First run downloads models (~1GB)
2. Subsequent runs are faster
3. Be patient on first launch
4. Models cached after first load

### High memory usage
Solutions:
1. NLP models require ~2GB RAM
2. Close other applications
3. Restart app if needed
4. Disable NLP in config if constrained

### Data seems old
Solutions:
1. Click "Force refresh" in Advanced Settings
2. Check cache TTL setting
3. Verify API keys are working
4. Check data source displayed

### Charts not displaying
Solutions:
1. Update browser
2. Enable JavaScript
3. Clear browser cache
4. Try different browser

---

## Privacy & Data

### What We Collect
- Location queries (city names)
- AQI historical data (local database)
- User profile settings (local only)
- No personal information sent to servers

### Data Storage
- Local SQLite database
- Stored on your machine only
- Not shared with third parties
- Can be deleted anytime

### API Keys
- Your API keys stay private
- Stored in .env file (never committed to git)
- Direct API calls (no intermediary)
- Rate limits apply per your account

---

## FAQ

Q: Is this free to use?
A: Yes! The app is free. API keys are optional for real data.

Q: How accurate are the predictions?
A: ML models trained on realistic patterns. Accuracy improves with real API data.

Q: Can I use this offline?
A: No, requires internet for API calls. Historical data available offline.

Q: Which cities are supported?
A: Most major cities worldwide. Try variations if city not found.

Q: How often does data update?
A: Every 5 minutes by default. Force refresh for instant updates.

Q: Is my data secure?
A: Yes, all data stored locally. No cloud uploads.

Q: Can I export data?
A: Currently manual export from database. PDF export coming soon.

Q: Mobile version available?
A: Web app works on mobile browsers. Native app planned.

Q: How to get API keys?
A: See README.md for links to WAQI and OpenWeather registration.

Q: What if API is down?
A: App automatically falls back to realistic simulations.

---

## Keyboard Shortcuts

- Ctrl/Cmd + R: Refresh page
- Ctrl/Cmd + K: Focus search
- Esc: Close dialogs

---

## Support & Feedback

### Getting Help
- 📖 Read README.md
- 🚀 Check DEPLOYMENT.md
- 🐛 Report bugs on GitHub
- 💬 Ask questions via email

### Contributing
- Submit feature requests
- Report bugs with details
- Share improvement suggestions
- Contribute code via pull requests

---

## Updates & Changelog

### Version 2.0.0 (Current)
- ✨ Streamlit interface
- 🤖 Hugging Face NLP integration
- 📊 Advanced ML predictions
- 💾 Historical data tracking
- 🎯 Personalized recommendations

### Roadmap
- 📱 Mobile app
- 🌐 Multi-language support
- 📧 Email alerts
- 📄 PDF reports
- 🗺️ Map view
- 📊 Weekly summaries

---

## Credits

Data Sources:
- World Air Quality Index (WAQI)
- OpenWeather Air Pollution API

ML/NLP:
- Hugging Face Transformers
- scikit-learn
- PyTorch

Visualization:
- Streamlit
- Plotly

Made with ❤️ for healthier air

---

Last Updated: January 2026
Version: 2.0.0
