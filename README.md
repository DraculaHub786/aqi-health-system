# 🌍 AQI Health & Activity Planner

> Your personal air quality assistant powered by AI and real-world data

Breathe smart. Live better. Get real-time air quality insights, health recommendations, and personalized activity suggestions—all in one beautiful app.

---

## ✨ What Makes This Special

🤖 AI-Powered Intelligence  
Ask anything about air quality and get smart, conversational answers. Our NLP engine understands you naturally.

📊 Real Data, Real Insights  
Kaggle datasets + live APIs + ML predictions = evidence-based recommendations you can trust.

🎯 Truly Personal  
Tell us about yourself (age, health, activity level) and get advice tailored just for you.

⚡ Works Everywhere  
No API key? No problem. Sample data keeps you informed even offline.

---

## 🚀 Get Started in 2 Minutes

```bash
# 1. Install
pip install -r requirements.txt

# 2. Setup (creates sample data automatically)
python setup_kaggle.py

# 3. Run
streamlit run streamlit_app.py
```

That's it! Open `http://localhost:8501` and start exploring.

---

## 💬 Just Ask Anything

Our universal query handler responds to every question:

- "Is it safe to run?" → Get instant safety advice
- "What can I do today?" → Personalized activity suggestions  
- "How does PM2.5 affect asthma?" → Clear health explanations
- Even "Hi!" or "Thanks!" → Natural conversations

---

## 📦 What's Inside

Core Features:
- 🌡️ Real-time AQI monitoring (works worldwide)
- 🔮 24-hour AI predictions with confidence levels
- 💊 Health risk assessment based on your profile
- 🏃 Smart activity recommendations
- 📈 Historical trends and beautiful visualizations

Enhanced Intelligence:
- 📚 Kaggle datasets (global pollution data)
- 🧠 NLP engine (responds to every query)
- 🎯 Pollutant-specific advice (PM2.5, O3, NO2, etc.)
- 🌍 WHO health guidelines integration

---

## 🔑 API Keys (Optional)

Works without keys using sample data. Want live data?

WAQI (Recommended - Free):
1. Get token: [aqicn.org/data-platform/token](https://aqicn.org/data-platform/token/)
2. Create `.env` file: `WAQI_API_KEY=your_token`

Kaggle (For Real Datasets):
1. Get from [kaggle.com](https://www.kaggle.com/account)
2. Place `kaggle.json` in `~/.kaggle/`
3. Run `python setup_kaggle.py`

### OpenWeather Air Pollution API - Optional
1. Visit [https://openweathermap.org/api](https://openweathermap.org/api)
2. Sign up and get API key
3. Add to `.env`: `OPENWEATHER_API_KEY=your_key`

> Note: The app works with demo keys using intelligent simulations, but real API keys provide accurate data.

## 🏗️ Project Structure

```
aqi-health-system/
│
├── streamlit_app.py              # Main Streamlit application
│
├── models/
│   ├── __init__.py
│   ├── advanced_ml.py            # ML models (Random Forest, Gradient Boosting)
│   └── nlp_engine.py             # Hugging Face NLP models
│
├── utils/
│   ├── __init__.py
│   ├── data_manager.py           # API integration & caching
│   └── config.py                 # Configuration & constants
│
├── data/
│   └── aqi_history.db            # SQLite database (auto-created)
│
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment variables template
├── README.md                     # This file
└── LICENSE                       # MIT License
```

## 🎯 Usage Guide

### 1. Basic Usage
- Enter a city name in the sidebar (e.g., "Delhi", "New York", "Beijing")
- Click "Fetch AQI Data" to get real-time air quality
- View AQI gauge, health analysis, and recommendations

### 2. Personalization
- Set your age and health conditions in the sidebar
- Choose activity intensity preferences
- Get personalized risk assessments and recommendations

### 3. Advanced Features
- 24-Hour Forecast: ML predictions with confidence intervals
- Historical Trends: Track AQI changes over time
- Q&A System: Ask questions about air quality
- Activity Planning: Find best times for outdoor activities

### 4. Production Tips
- Enable caching for faster responses
- Set real API keys for accurate data
- Use force refresh sparingly to avoid rate limits
- Check debug info for troubleshooting

## 🧪 Technology Stack

### Machine Learning
- scikit-learn: Random Forest, Gradient Boosting
- NumPy/Pandas: Data processing and feature engineering
- joblib: Model serialization and caching

### Natural Language Processing
- Transformers (Hugging Face): 
  - `distilgpt2`: Text generation
  - `distilbert`: Sentiment analysis & Q&A
  - `facebook/bart-large-cnn`: Summarization
- PyTorch: Backend for transformers

### Frontend & Visualization
- Streamlit: Web application framework
- Plotly: Interactive charts and visualizations
- Custom CSS: Enhanced UI/UX

### Data & APIs
- SQLite: Local database for historical data
- Requests: HTTP client for API calls
- Python-dotenv: Environment management

## 📊 ML Models Explained

### 1. AQI Time Series Predictor
- Algorithm: Random Forest Regressor
- Features: Hour, day, weather, pollutants, historical AQI
- Output: 24-hour forecast with confidence intervals
- Accuracy: Trained on 5000+ synthetic data points

### 2. Health Risk Classifier
- Algorithm: Gradient Boosting Classifier
- Features: AQI, age, health conditions, exposure time
- Output: 5-level risk classification (Low to Severe)
- Personalization: User-specific risk scoring

### 3. Activity Recommender
- Algorithm: Rule-based ML scoring system
- Features: AQI, weather, time, user preferences
- Output: Ranked activity suggestions with safety levels

### 4. NLP Explanation Generator
- Models: GPT-2 for generation, template enhancement
- Input: AQI data, pollutants, user profile
- Output: Natural language health explanations

## 🔧 Configuration

### Environment Variables
```bash
# API Keys
WAQI_API_KEY=your_waqi_token
OPENWEATHER_API_KEY=your_openweather_key

# Application Settings
ENV=production
DEBUG=false
CACHE_TTL=300
LOG_LEVEL=INFO

# Database
DB_PATH=data/aqi_history.db
```

### Model Configuration
Edit [utils/config.py](utils/config.py):
```python
ML_CONFIG = {
    'predictor': {
        'n_estimators': 100,
        'forecast_hours': 24,
    },
    # ... more settings
}
```

## 🚀 Deployment

### Local Production
```bash
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```

### Docker (Optional)
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py"]
```

### Cloud Platforms

#### Streamlit Cloud
1. Push code to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect repository and deploy

#### Heroku
```bash
# Create Procfile
echo "web: streamlit run streamlit_app.py --server.port $PORT" > Procfile

# Deploy
heroku create your-app-name
git push heroku main
```

#### AWS/Azure/GCP
- Use Docker container
- Deploy to ECS/App Service/Cloud Run
- Configure environment variables
- Set up load balancer for scaling

## 📈 Performance Optimization

### Model Loading
- Models loaded once using singleton pattern
- Cached predictions for repeated queries
- Lazy loading of NLP models

### Database
- Indexed queries for fast lookups
- Automatic cleanup of old data
- Connection pooling

### Caching
- 5-minute TTL for API responses
- LRU cache for frequently accessed data
- Configurable cache settings

## 🛡️ Error Handling

The application includes comprehensive error handling:
- API failures → Graceful fallback to simulations
- Model errors → Template-based responses
- Database issues → In-memory temporary storage
- Network timeouts → Cached data when available

## 🧪 Testing

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/ -v --cov=models --cov=utils

# Generate coverage report
pytest --cov=. --cov-report=html
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🐛 Known Issues & Limitations

1. NLP Models: First run downloads models (~1GB). Subsequent runs are faster.
2. API Rate Limits: Free API tiers have request limits. Use caching.
3. Simulated Data: Without API keys, uses realistic simulations based on location patterns.
4. Memory Usage: NLP models require ~2GB RAM. Disable if constrained.

## 🔮 Future Enhancements

- [ ] Multi-language support
- [ ] Mobile app version
- [ ] Push notifications for alerts
- [ ] PDF report generation
- [ ] Social sharing features
- [ ] Weather integration
- [ ] Air purifier recommendations
- [ ] Comparison between cities
- [ ] Weekly/monthly summaries

## 📞 Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Email: your-email@example.com
- Documentation: [Wiki](https://github.com/your-repo/wiki)

## 🙏 Acknowledgments

- WAQI for air quality data API
- Hugging Face for transformer models
- Streamlit for the amazing framework
- OpenWeather for weather data
- scikit-learn for ML algorithms

## 📊 Stats

---

## 📚 Documentation

- [User Guide](USER_GUIDE.md) - How to use the app
- [Deployment](DEPLOYMENT.md) - Hosting instructions
- [Kaggle Data](KAGGLE_INTEGRATION.md) - Dataset integration
- [Updates](IMPLEMENTATION_SUMMARY.md) - Latest features

---

## 🤝 Support

Questions? The AI assistant in the app can help!

Or check the docs above for detailed guides.

---

Made with ❤️ for healthier living

*Breathe smarter. Live better.* 🌍
