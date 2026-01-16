import os
from typing import Dict, List

# Application Configuration
APP_CONFIG = {
    'app_name': 'AQI Health & Activity Planner',
    'version': '2.0.0',
    'environment': os.getenv('ENV', 'production'),
    'debug': os.getenv('DEBUG', 'False').lower() == 'true',
    'cache_ttl': int(os.getenv('CACHE_TTL', '300')),  # 5 minutes
    'db_path': os.getenv('DB_PATH', 'data/aqi_history.db'),
    'log_level': os.getenv('LOG_LEVEL', 'INFO'),
}

# API Configuration
API_CONFIG = {
    'waqi': {
        'base_url': 'https://api.waqi.info',
        'api_key': os.getenv('WAQI_API_KEY'),
        'timeout': 10,
    },
    'openweather': {
        'base_url': 'http://api.openweathermap.org',
        'api_key': os.getenv('OPENWEATHER_API_KEY', 'demo'),
        'timeout': 5,
    }
}

# AQI Categories with detailed information
AQI_CATEGORIES = {
    'good': {
        'range': (0, 50),
        'color': '#00e400',
        'label': 'Good',
        'description': 'Air quality is satisfactory, and air pollution poses little or no risk.',
        'health_implications': 'None',
        'cautionary_statement': 'None',
    },
    'moderate': {
        'range': (51, 100),
        'color': '#ffff00',
        'label': 'Moderate',
        'description': 'Air quality is acceptable. However, there may be a risk for some people.',
        'health_implications': 'Unusually sensitive people should consider limiting prolonged outdoor exertion.',
        'cautionary_statement': 'Active children and adults, and people with respiratory disease should limit prolonged outdoor exertion.',
    },
    'unhealthy_sensitive': {
        'range': (101, 150),
        'color': '#ff7e00',
        'label': 'Unhealthy for Sensitive Groups',
        'description': 'Members of sensitive groups may experience health effects.',
        'health_implications': 'People with respiratory or heart disease should limit prolonged outdoor exertion.',
        'cautionary_statement': 'Active children and adults should limit prolonged outdoor exertion.',
    },
    'unhealthy': {
        'range': (151, 200),
        'color': '#ff0000',
        'label': 'Unhealthy',
        'description': 'Some members of the general public may experience health effects.',
        'health_implications': 'Everyone should limit prolonged outdoor exertion.',
        'cautionary_statement': 'People with respiratory or heart disease should avoid prolonged outdoor exertion.',
    },
    'very_unhealthy': {
        'range': (201, 300),
        'color': '#8f3f97',
        'label': 'Very Unhealthy',
        'description': 'Health alert: The risk of health effects is increased for everyone.',
        'health_implications': 'Everyone should avoid all outdoor exertion.',
        'cautionary_statement': 'People with respiratory or heart disease should remain indoors and keep activity levels low.',
    },
    'hazardous': {
        'range': (301, 500),
        'color': '#7e0023',
        'label': 'Hazardous',
        'description': 'Health warning of emergency conditions: everyone is more likely to be affected.',
        'health_implications': 'Everyone should avoid all outdoor exertion.',
        'cautionary_statement': 'Everyone should remain indoors and keep windows closed. Use air purifiers.',
    }
}

# Pollutant Information
POLLUTANT_INFO = {
    'pm25': {
        'name': 'PM2.5',
        'full_name': 'Fine Particulate Matter',
        'unit': 'µg/m³',
        'description': 'Particles less than 2.5 micrometers in diameter. Can penetrate deep into lungs.',
        'health_effects': [
            'Respiratory irritation',
            'Decreased lung function',
            'Aggravated asthma',
            'Chronic respiratory disease',
            'Cardiovascular effects'
        ],
        'sources': [
            'Vehicle emissions',
            'Industrial processes',
            'Residential wood burning',
            'Construction dust'
        ],
        'safe_level': 12.0,
        'unhealthy_level': 35.5,
    },
    'pm10': {
        'name': 'PM10',
        'full_name': 'Coarse Particulate Matter',
        'unit': 'µg/m³',
        'description': 'Particles less than 10 micrometers in diameter.',
        'health_effects': [
            'Eye, nose, and throat irritation',
            'Coughing and sneezing',
            'Shortness of breath',
            'Aggravated asthma'
        ],
        'sources': [
            'Road dust',
            'Construction sites',
            'Agricultural activities',
            'Wind-blown dust'
        ],
        'safe_level': 54.0,
        'unhealthy_level': 154.0,
    },
    'o3': {
        'name': 'O₃',
        'full_name': 'Ozone',
        'unit': 'ppb',
        'description': 'Ground-level ozone formed by chemical reactions in sunlight.',
        'health_effects': [
            'Respiratory irritation',
            'Reduced lung function',
            'Aggravated asthma',
            'Increased respiratory infections'
        ],
        'sources': [
            'Vehicle emissions (precursors)',
            'Industrial emissions (precursors)',
            'Chemical solvents',
            'Sunlight-driven reactions'
        ],
        'safe_level': 54.0,
        'unhealthy_level': 70.0,
    },
    'no2': {
        'name': 'NO₂',
        'full_name': 'Nitrogen Dioxide',
        'unit': 'ppb',
        'description': 'Gas produced from burning fuel.',
        'health_effects': [
            'Respiratory irritation',
            'Increased asthma symptoms',
            'Reduced immunity to respiratory infections'
        ],
        'sources': [
            'Vehicle emissions',
            'Power plants',
            'Industrial boilers',
            'Off-road equipment'
        ],
        'safe_level': 53.0,
        'unhealthy_level': 100.0,
    },
    'so2': {
        'name': 'SO₂',
        'full_name': 'Sulfur Dioxide',
        'unit': 'ppb',
        'description': 'Gas produced from burning sulfur-containing fuels.',
        'health_effects': [
            'Respiratory irritation',
            'Breathing difficulties',
            'Aggravated asthma',
            'Cardiovascular effects'
        ],
        'sources': [
            'Coal-fired power plants',
            'Industrial facilities',
            'Diesel engines',
            'Metal processing'
        ],
        'safe_level': 35.0,
        'unhealthy_level': 75.0,
    },
    'co': {
        'name': 'CO',
        'full_name': 'Carbon Monoxide',
        'unit': 'ppm',
        'description': 'Odorless, colorless gas from incomplete combustion.',
        'health_effects': [
            'Reduced oxygen delivery to organs',
            'Headaches',
            'Dizziness',
            'Vision problems',
            'Cardiovascular effects'
        ],
        'sources': [
            'Vehicle emissions',
            'Industrial processes',
            'Residential heating',
            'Wildfires'
        ],
        'safe_level': 4.4,
        'unhealthy_level': 9.4,
    }
}

# User Profile Templates
USER_PROFILES = {
    'general': {
        'age': 30,
        'respiratory_condition': False,
        'heart_condition': False,
        'outdoor_worker': False,
        'preferred_intensity': 'medium',
        'daily_outdoor_hours': 2,
    },
    'children': {
        'age': 8,
        'respiratory_condition': False,
        'heart_condition': False,
        'outdoor_worker': False,
        'preferred_intensity': 'high',
        'daily_outdoor_hours': 3,
    },
    'elderly': {
        'age': 70,
        'respiratory_condition': False,
        'heart_condition': False,
        'outdoor_worker': False,
        'preferred_intensity': 'low',
        'daily_outdoor_hours': 1,
    },
    'outdoor_worker': {
        'age': 35,
        'respiratory_condition': False,
        'heart_condition': False,
        'outdoor_worker': True,
        'preferred_intensity': 'medium',
        'daily_outdoor_hours': 8,
    },
    'athlete': {
        'age': 25,
        'respiratory_condition': False,
        'heart_condition': False,
        'outdoor_worker': False,
        'preferred_intensity': 'high',
        'daily_outdoor_hours': 4,
    }
}

# ML Model Configuration
ML_CONFIG = {
    'predictor': {
        'n_estimators': 100,
        'max_depth': 15,
        'random_state': 42,
        'forecast_hours': 24,
    },
    'classifier': {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 5,
        'random_state': 42,
    },
    'cache_models': True,
    'model_path': 'models/cache',
}

# NLP Configuration
NLP_CONFIG = {
    'generator_model': 'distilgpt2',  # Lightweight for speed
    'sentiment_model': 'distilbert-base-uncased-finetuned-sst-2-english',
    'summarizer_model': 'facebook/bart-large-cnn',
    'qa_model': 'distilbert-base-cased-distilled-squad',
    'device': -1,  # CPU (-1), GPU (0)
    'max_length': 200,
    'temperature': 0.7,
}

# Feature Flags
FEATURES = {
    'enable_nlp': True,
    'enable_ml_predictions': True,
    'enable_historical_data': True,
    'enable_caching': True,
    'enable_qa': True,
    'enable_pdf_export': True,
}

# Locations database (for autocomplete and validation)
POPULAR_LOCATIONS = [
    # India
    'Delhi', 'Mumbai', 'Bangalore', 'Pune', 'Kolkata', 'Chennai', 'Hyderabad',
    'Ahmedabad', 'Jaipur', 'Lucknow', 'Kanpur', 'Nagpur', 'Indore', 'Patna',
    'Bhopal', 'Ludhiana', 'Agra', 'Nashik', 'Vadodara', 'Surat',
    
    # International
    'Beijing', 'Shanghai', 'Tokyo', 'Seoul', 'Bangkok', 'Singapore',
    'London', 'Paris', 'Berlin', 'New York', 'Los Angeles', 'Chicago',
    'San Francisco', 'Seattle', 'Boston', 'Sydney', 'Melbourne',
]

# Visualization Configuration
VIZ_CONFIG = {
    'theme': 'plotly',
    'color_scheme': {
        'primary': '#1f77b4',
        'secondary': '#ff7f0e',
        'success': '#2ca02c',
        'warning': '#ff7f00',
        'danger': '#d62728',
    },
    'chart_height': 400,
    'gauge_height': 350,
}


def get_category_for_aqi(aqi: float) -> Dict:
    """Get AQI category information for a given AQI value"""
    for category, info in AQI_CATEGORIES.items():
        min_aqi, max_aqi = info['range']
        if min_aqi <= aqi <= max_aqi:
            return info
    
    # Default to hazardous if above all ranges
    return AQI_CATEGORIES['hazardous']


def validate_config():
    """Validate configuration on startup"""
    errors = []
    
    # Check API keys
    if API_CONFIG['waqi']['api_key'] == 'demo':
        errors.append("Warning: Using demo WAQI API key. Get your key from https://aqicn.org/data-platform/token/")
    
    if API_CONFIG['openweather']['api_key'] == 'demo':
        errors.append("Info: OpenWeather API key not set. Using fallback data sources.")
    
    # Check database path
    db_dir = os.path.dirname(APP_CONFIG['db_path'])
    if db_dir and not os.path.exists(db_dir):
        try:
            os.makedirs(db_dir, exist_ok=True)
        except Exception as e:
            errors.append(f"Error: Cannot create database directory: {e}")
    
    return errors


# Initialize and validate on import
config_errors = validate_config()
if config_errors:
    import logging
    logger = logging.getLogger(__name__)
    for error in config_errors:
        logger.warning(error)
