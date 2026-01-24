from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import requests
import numpy as np
from datetime import datetime, timedelta
from ml_model import AQIClassifier, ActivityRecommender, AQIPredictor
from nlp_engine import NLPExplainer
import json
import os

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional

app = Flask(__name__)
CORS(app)

# Initialize AI models
aqi_classifier = AQIClassifier()
activity_recommender = ActivityRecommender()
aqi_predictor = AQIPredictor()
nlp_explainer = NLPExplainer()

# Load API keys from environment
WAQI_API_KEY = os.getenv('WAQI_API_KEY', 'c294968cd8d66811cae8a5cddb3928cf4b7ff695')
OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY', 'demo')

# Multiple API sources for better accuracy
API_KEYS = {
    'waqi': WAQI_API_KEY, 
    'openweather': OPENWEATHER_API_KEY
}

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/api/aqi', methods=['POST'])
def get_aqi_data():
    """API endpoint for AQI data with AI processing"""
    try:
        data = request.json
        location = data.get('location')
        user_profile = data.get('userProfile', 'general')
        
        # Fetch AQI from multiple sources
        aqi_data = fetch_aqi_data_multi_source(location)
        
        if not aqi_data:
            return jsonify({'error': 'Unable to fetch AQI data'}), 400
        
        aqi_value = aqi_data['aqi']
        pollutants = aqi_data['pollutants']
        
        # AI Classification with context
        risk_level = aqi_classifier.classify_with_context(
            aqi_value, 
            pollutants,
            datetime.now().hour,
            user_profile
        )
        
        # AI-powered NLP Explanation
        explanation = nlp_explainer.generate_contextual_explanation(
            aqi_value,
            pollutants,
            risk_level,
            user_profile,
            location,
            datetime.now()
        )
        
        # ML-based Activity Recommendations
        recommendations = activity_recommender.get_smart_recommendations(
            aqi_value,
            pollutants,
            user_profile,
            datetime.now(),
            aqi_data.get('weather', {})
        )
        
        # AI Time slot predictions using ML
        time_slots = aqi_predictor.predict_hourly_aqi(
            aqi_value,
            pollutants,
            location,
            datetime.now()
        )
        
        # Context-aware health tips
        health_tips = generate_smart_health_tips(
            aqi_value,
            pollutants,
            user_profile,
            risk_level
        )
        
        response = {
            'aqi': aqi_value,
            'category': risk_level['category'],
            'color': risk_level['color'],
            'pollutants': pollutants,
            'dominantPollutant': aqi_data.get('dominant_pollutant', 'PM2.5'),
            'timestamp': datetime.now().isoformat(),
            'explanation': explanation,
            'recommendations': recommendations,
            'timeSlots': time_slots,
            'healthTips': health_tips,
            'confidence': risk_level.get('confidence', 85),
            'dataSource': aqi_data.get('source', 'Multiple Sources')
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        print(f"Error in get_aqi_data: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

def fetch_aqi_data_multi_source(location):
    """Fetch AQI from multiple sources and aggregate"""
    
    # Try WAQI API first (most reliable)
    waqi_data = fetch_from_waqi(location)
    if waqi_data:
        return waqi_data
    
    # Fallback to OpenWeather
    openweather_data = fetch_from_openweather(location)
    if openweather_data:
        return openweather_data
    
    # Ultimate fallback: Realistic simulation based on location
    return fetch_realistic_simulation(location)

def fetch_from_waqi(location):
    """Fetch from World Air Quality Index API"""
    try:
        token = API_KEYS.get('waqi', 'demo')
        
        # Try with token
        url = f"https://api.waqi.info/feed/{location}/?token={token}"
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if data.get('status') == 'ok':
            aqi_data = data['data']
            iaqi = aqi_data.get('iaqi', {})
            
            pollutants = {
                'pm25': iaqi.get('pm25', {}).get('v', 0),
                'pm10': iaqi.get('pm10', {}).get('v', 0),
                'o3': iaqi.get('o3', {}).get('v', 0),
                'no2': iaqi.get('no2', {}).get('v', 0),
                'so2': iaqi.get('so2', {}).get('v', 0),
                'co': iaqi.get('co', {}).get('v', 0)
            }
            
            # Determine dominant pollutant
            dominant = max(pollutants.items(), key=lambda x: x[1])[0]
            
            return {
                'aqi': aqi_data['aqi'],
                'pollutants': pollutants,
                'dominant_pollutant': dominant.upper(),
                'source': 'WAQI',
                'weather': aqi_data.get('weather', {})
            }
    except Exception as e:
        print(f"WAQI API Error: {e}")
    
    return None

def fetch_from_openweather(location):
    """Fetch from OpenWeather Air Pollution API"""
    try:
        api_key = API_KEYS.get('openweather')
        if not api_key or api_key == 'your_openweather_key_here':
            return None
        
        # First get coordinates
        geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={location}&limit=1&appid={api_key}"
        geo_response = requests.get(geo_url, timeout=5)
        geo_data = geo_response.json()
        
        if not geo_data:
            return None
        
        lat = geo_data[0]['lat']
        lon = geo_data[0]['lon']
        
        # Get air pollution data
        pollution_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}"
        pollution_response = requests.get(pollution_url, timeout=5)
        pollution_data = pollution_response.json()
        
        if pollution_data.get('list'):
            components = pollution_data['list'][0]['components']
            
            # Convert to US AQI
            pm25_aqi = calculate_aqi_from_concentration(components.get('pm2_5', 0), 'pm25')
            
            pollutants = {
                'pm25': int(components.get('pm2_5', 0)),
                'pm10': int(components.get('pm10', 0)),
                'o3': int(components.get('o3', 0)),
                'no2': int(components.get('no2', 0)),
                'so2': int(components.get('so2', 0)),
                'co': int(components.get('co', 0) / 100)
            }
            
            dominant = max(pollutants.items(), key=lambda x: x[1])[0]
            
            return {
                'aqi': pm25_aqi,
                'pollutants': pollutants,
                'dominant_pollutant': dominant.upper(),
                'source': 'OpenWeather'
            }
    except Exception as e:
        print(f"OpenWeather API Error: {e}")
    
    return None

def calculate_aqi_from_concentration(concentration, pollutant):
    """Convert pollutant concentration to AQI"""
    if pollutant == 'pm25':
        # PM2.5 breakpoints
        breakpoints = [
            (0, 12.0, 0, 50),
            (12.1, 35.4, 51, 100),
            (35.5, 55.4, 101, 150),
            (55.5, 150.4, 151, 200),
            (150.5, 250.4, 201, 300),
            (250.5, 500.4, 301, 500)
        ]
        
        for bp_lo, bp_hi, aqi_lo, aqi_hi in breakpoints:
            if bp_lo <= concentration <= bp_hi:
                return int(((aqi_hi - aqi_lo) / (bp_hi - bp_lo)) * (concentration - bp_lo) + aqi_lo)
    
    return int(concentration * 2)  # Simple conversion

def fetch_realistic_simulation(location):
    """Generate realistic AQI based on location patterns and time"""
    import random
    
    # Base AQI patterns for Indian cities (more realistic)
    city_patterns = {
        'delhi': {'base': 180, 'variance': 50, 'pollution_profile': 'severe'},
        'mumbai': {'base': 110, 'variance': 30, 'pollution_profile': 'moderate'},
        'bangalore': {'base': 75, 'variance': 25, 'pollution_profile': 'good'},
        'pune': {'base': 85, 'variance': 25, 'pollution_profile': 'moderate'},
        'kolkata': {'base': 145, 'variance': 35, 'pollution_profile': 'high'},
        'chennai': {'base': 95, 'variance': 25, 'pollution_profile': 'moderate'},
        'hyderabad': {'base': 90, 'variance': 25, 'pollution_profile': 'moderate'},
        'ahmedabad': {'base': 125, 'variance': 30, 'pollution_profile': 'high'},
        'jaipur': {'base': 135, 'variance': 35, 'pollution_profile': 'high'},
        'lucknow': {'base': 155, 'variance': 40, 'pollution_profile': 'high'},
        'nagpur': {'base': 105, 'variance': 30, 'pollution_profile': 'moderate'},
        'indore': {'base': 120, 'variance': 30, 'pollution_profile': 'moderate'},
        'patna': {'base': 170, 'variance': 45, 'pollution_profile': 'severe'},
        'kanpur': {'base': 175, 'variance': 45, 'pollution_profile': 'severe'},
    }
    
    # Time-based variations
    hour = datetime.now().hour
    time_multiplier = 1.0
    
    if 6 <= hour <= 9:  # Morning - traffic peak
        time_multiplier = 1.15
    elif 10 <= hour <= 16:  # Daytime - better dispersion
        time_multiplier = 0.90
    elif 18 <= hour <= 21:  # Evening - traffic peak
        time_multiplier = 1.20
    else:  # Night - lower activity
        time_multiplier = 0.95
    
    # Get city pattern
    location_lower = location.lower()
    pattern = None
    
    for city, data in city_patterns.items():
        if city in location_lower:
            pattern = data
            break
    
    if not pattern:
        pattern = {'base': 100, 'variance': 30, 'pollution_profile': 'moderate'}
    
    # Calculate AQI with time variation
    base_aqi = pattern['base']
    variance = pattern['variance']
    
    aqi = int(base_aqi * time_multiplier + random.randint(-variance, variance))
    aqi = max(20, min(400, aqi))
    
    # Generate realistic pollutant distribution
    pollutants = generate_pollutant_profile(aqi, pattern['pollution_profile'])
    
    # Determine dominant pollutant
    dominant = max(pollutants.items(), key=lambda x: x[1])[0]
    
    return {
        'aqi': aqi,
        'pollutants': pollutants,
        'dominant_pollutant': dominant.upper(),
        'source': 'Simulated (Pattern-Based)'
    }

def generate_pollutant_profile(aqi, profile_type):
    """Generate realistic pollutant distribution"""
    import random
    
    if profile_type == 'severe':
        # Delhi-like: High PM2.5 and PM10
        return {
            'pm25': int(aqi * 0.75 + random.randint(-15, 15)),
            'pm10': int(aqi * 0.85 + random.randint(-20, 20)),
            'o3': int(aqi * 0.25 + random.randint(-5, 10)),
            'no2': int(aqi * 0.35 + random.randint(-8, 12)),
            'so2': int(aqi * 0.15 + random.randint(-5, 8)),
            'co': int(aqi * 3.5 + random.randint(-30, 50))
        }
    elif profile_type == 'high':
        # Moderate-high pollution
        return {
            'pm25': int(aqi * 0.65 + random.randint(-10, 10)),
            'pm10': int(aqi * 0.75 + random.randint(-15, 15)),
            'o3': int(aqi * 0.30 + random.randint(-5, 8)),
            'no2': int(aqi * 0.30 + random.randint(-5, 10)),
            'so2': int(aqi * 0.18 + random.randint(-3, 5)),
            'co': int(aqi * 3.0 + random.randint(-25, 40))
        }
    elif profile_type == 'moderate':
        # Balanced pollution
        return {
            'pm25': int(aqi * 0.55 + random.randint(-8, 8)),
            'pm10': int(aqi * 0.65 + random.randint(-12, 12)),
            'o3': int(aqi * 0.35 + random.randint(-5, 5)),
            'no2': int(aqi * 0.28 + random.randint(-5, 8)),
            'so2': int(aqi * 0.15 + random.randint(-3, 5)),
            'co': int(aqi * 2.5 + random.randint(-20, 30))
        }
    else:  # good
        # Coastal/clean air cities
        return {
            'pm25': int(aqi * 0.45 + random.randint(-5, 5)),
            'pm10': int(aqi * 0.55 + random.randint(-8, 8)),
            'o3': int(aqi * 0.40 + random.randint(-3, 5)),
            'no2': int(aqi * 0.25 + random.randint(-3, 5)),
            'so2': int(aqi * 0.12 + random.randint(-2, 3)),
            'co': int(aqi * 2.0 + random.randint(-15, 25))
        }

def generate_smart_health_tips(aqi, pollutants, profile, risk_level):
    """Generate AI-powered, context-aware health tips"""
    tips = []
    
    # Analyze dominant pollutant
    dominant = max(pollutants.items(), key=lambda x: x[1])
    dominant_name, dominant_value = dominant
    
    # Base tips on AQI level
    if aqi > 200:
        tips.append(f"⚠️ CRITICAL: AQI is {aqi}. Stay indoors and avoid all physical exertion.")
        tips.append("🏠 Seal windows and doors. Use air purifiers on highest setting.")
        
        if profile == 'children':
            tips.append("👶 Children: Keep indoors. Monitor for coughing, wheezing, or breathing difficulty.")
        elif profile == 'elderly':
            tips.append("👴 Elderly: This is a medical emergency. Contact healthcare provider if experiencing symptoms.")
        elif profile == 'workers':
            tips.append("👷 Workers: Employers must halt non-essential outdoor work. N95 masks mandatory.")
            
    elif aqi > 150:
        tips.append(f"🔴 UNHEALTHY: AQI is {aqi}. Everyone should reduce outdoor exposure.")
        
        if dominant_name == 'pm25':
            tips.append("😷 High PM2.5 detected. Wear N95/KN95 masks outdoors.")
        elif dominant_name == 'o3':
            tips.append("🌤️ High Ozone levels. Avoid outdoor activities during afternoon hours.")
            
        if profile == 'children':
            tips.append("🏫 Schools: Cancel outdoor activities. Keep children in air-filtered rooms.")
        elif profile == 'elderly':
            tips.append("💊 Seniors: Have medications ready. Watch for chest discomfort or irregular heartbeat.")
            
    elif aqi > 100:
        tips.append(f"🟡 MODERATE: AQI is {aqi}. Sensitive groups should limit prolonged outdoor exposure.")
        
        if dominant_name in ['pm25', 'pm10']:
            tips.append("😷 Consider wearing masks during outdoor activities.")
        
        tips.append("💧 Stay hydrated and take frequent breaks if exercising outdoors.")
        
        if profile == 'children':
            tips.append("⚽ Children: Limit intense outdoor sports to 30-45 minutes.")
        elif profile == 'workers':
            tips.append("☕ Outdoor workers: Take 10-minute breaks every 2 hours in clean air.")
            
    else:
        tips.append(f"✅ GOOD: AQI is {aqi}. Perfect conditions for outdoor activities!")
        tips.append("🏃 Great day for exercise, sports, and outdoor recreation.")
        
        if profile == 'children':
            tips.append("🎈 Excellent conditions for children to play outside.")
        elif profile == 'elderly':
            tips.append("🚶 Ideal for morning walks and outdoor relaxation.")
    
    # Add time-specific tips
    hour = datetime.now().hour
    if 18 <= hour <= 21 and aqi > 100:
        tips.append("🌆 Evening traffic increases pollution. Stay indoors during peak hours.")
    elif 6 <= hour <= 9 and aqi < 100:
        tips.append("🌅 Morning air quality is typically best. Perfect time for exercise.")
    
    return tips[:6]  # Return top 6 tips

if __name__ == '__main__':
    print("=" * 70)
    print("🌍 AI AQI Health & Activity Planner Server")
    print("=" * 70)
    print("\n✅ Server starting with AI/ML capabilities...")
    print("📍 Access the application at: http://localhost:5000")
    print("🤖 AI Features: NLP Explanations | ML Predictions | Smart Recommendations")
    print("🔄 Press CTRL+C to stop the server\n")
    print("=" * 70)
    
    # Windows-compatible run configuration
    app.run(debug=True, port=5000, host='0.0.0.0', use_reloader=False)