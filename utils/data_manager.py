import requests
import sqlite3
import json
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from functools import lru_cache
import hashlib
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AQIDataManager:
    """
    Centralized data management for AQI data
    Handles multiple API sources with fallbacks and caching
    """
    
    def __init__(self, db_path: str = "data/aqi_history.db", cache_ttl: int = 300):
        """
        Initialize data manager
        
        Args:
            db_path: Path to SQLite database
            cache_ttl: Cache time-to-live in seconds
        """
        self.db_path = db_path
        self.cache_ttl = cache_ttl
        self._ensure_database()
        
        # API Keys - Users should set these in environment or config
        self.api_keys = {
            'waqi': os.getenv('WAQI_API_KEY', 'c294968cd8d66811cae8a5cddb3928cf4b7ff695'),
            'openweather': os.getenv('OPENWEATHER_API_KEY', 'demo')
        }
        
    def _ensure_database(self):
        """Create database and tables if not exist"""
        try:
            os.makedirs(os.path.dirname(self.db_path) if os.path.dirname(self.db_path) else 'data', exist_ok=True)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # AQI history table
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
            
            # User profiles table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_profiles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT UNIQUE NOT NULL,
                    age INTEGER,
                    has_respiratory_condition BOOLEAN,
                    has_heart_condition BOOLEAN,
                    preferred_intensity TEXT,
                    outdoor_worker BOOLEAN,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Cache table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS api_cache (
                    cache_key TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    expiry DATETIME NOT NULL
                )
            ''')
            
            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_location_timestamp ON aqi_history(location, timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_cache_expiry ON api_cache(expiry)')
            
            conn.commit()
            conn.close()
            
            logger.info(f"Database initialized at {self.db_path}")
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            
    def get_aqi_data(self, location: str, force_refresh: bool = False) -> Optional[Dict]:
        """
        Get AQI data from cache or API
        
        Args:
            location: Location name or coordinates
            force_refresh: Skip cache and fetch fresh data
            
        Returns:
            AQI data dictionary or None
        """
        # Check cache first
        if not force_refresh:
            cached_data = self._get_from_cache(location)
            if cached_data:
                logger.info(f"Cache hit for {location}")
                return cached_data
                
        # Fetch from APIs
        logger.info(f"Fetching fresh data for {location}")
        
        # Try WAQI first
        data = self._fetch_from_waqi(location)
        if data:
            self._save_to_cache(location, data)
            self._save_to_history(location, data)
            return data
            
        # Fallback to OpenWeather
        data = self._fetch_from_openweather(location)
        if data:
            self._save_to_cache(location, data)
            self._save_to_history(location, data)
            return data
            
        # Ultimate fallback: realistic simulation
        logger.warning(f"Using simulation for {location}")
        data = self._generate_simulation(location)
        self._save_to_cache(location, data)
        return data
        
    def _get_from_cache(self, location: str) -> Optional[Dict]:
        """Retrieve data from cache if not expired"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cache_key = self._generate_cache_key(location)
            cursor.execute(
                'SELECT data, expiry FROM api_cache WHERE cache_key = ?',
                (cache_key,)
            )
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                data_json, expiry = result
                expiry_time = datetime.fromisoformat(expiry)
                
                if datetime.now() < expiry_time:
                    return json.loads(data_json)
                else:
                    # Expired, delete it
                    self._delete_from_cache(cache_key)
                    
        except Exception as e:
            logger.error(f"Cache retrieval error: {e}")
            
        return None
        
    def _save_to_cache(self, location: str, data: Dict):
        """Save data to cache"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cache_key = self._generate_cache_key(location)
            expiry = datetime.now() + timedelta(seconds=self.cache_ttl)
            
            cursor.execute('''
                INSERT OR REPLACE INTO api_cache (cache_key, data, timestamp, expiry)
                VALUES (?, ?, CURRENT_TIMESTAMP, ?)
            ''', (cache_key, json.dumps(data), expiry.isoformat()))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Cache save error: {e}")
            
    def _delete_from_cache(self, cache_key: str):
        """Delete expired cache entry"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('DELETE FROM api_cache WHERE cache_key = ?', (cache_key,))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Cache deletion error: {e}")
            
    def _generate_cache_key(self, location: str) -> str:
        """Generate cache key from location"""
        return hashlib.md5(location.lower().encode()).hexdigest()
        
    def _fetch_from_waqi(self, location: str) -> Optional[Dict]:
        """Fetch from World Air Quality Index API"""
        try:
            token = self.api_keys.get('waqi', 'demo')
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
                
                # Remove zero values
                pollutants = {k: v for k, v in pollutants.items() if v > 0}
                
                if not pollutants:
                    pollutants = {'pm25': aqi_data['aqi'] * 0.6}
                
                dominant = max(pollutants.items(), key=lambda x: x[1])[0]
                
                return {
                    'aqi': aqi_data['aqi'],
                    'pollutants': pollutants,
                    'dominant_pollutant': dominant.upper(),
                    'source': 'WAQI',
                    'location': aqi_data.get('city', {}).get('name', location),
                    'timestamp': datetime.now().isoformat(),
                    'weather': aqi_data.get('weather', {})
                }
                
        except Exception as e:
            logger.error(f"WAQI API error: {e}")
            
        return None
        
    def _fetch_from_openweather(self, location: str) -> Optional[Dict]:
        """Fetch from OpenWeather Air Pollution API"""
        try:
            api_key = self.api_keys.get('openweather')
            if not api_key or api_key == 'demo':
                return None
                
            # Get coordinates
            geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={location}&limit=1&appid={api_key}"
            geo_response = requests.get(geo_url, timeout=5)
            geo_data = geo_response.json()
            
            if not geo_data:
                return None
                
            lat = geo_data[0]['lat']
            lon = geo_data[0]['lon']
            
            # Get pollution data
            pollution_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}"
            pollution_response = requests.get(pollution_url, timeout=5)
            pollution_data = pollution_response.json()
            
            if pollution_data.get('list'):
                components = pollution_data['list'][0]['components']
                
                # Convert to AQI
                pm25_aqi = self._calculate_aqi(components.get('pm2_5', 0), 'pm25')
                
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
                    'source': 'OpenWeather',
                    'location': geo_data[0].get('name', location),
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"OpenWeather API error: {e}")
            
        return None
        
    def _calculate_aqi(self, concentration: float, pollutant: str) -> int:
        """Convert concentration to AQI"""
        if pollutant == 'pm25':
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
                    
        return int(concentration * 2)
        
    def _generate_simulation(self, location: str) -> Dict:
        """Generate realistic simulation based on location patterns"""
        import random
        
        city_patterns = {
            'delhi': {'base': 180, 'variance': 50, 'profile': 'severe'},
            'mumbai': {'base': 110, 'variance': 30, 'profile': 'moderate'},
            'bangalore': {'base': 75, 'variance': 25, 'profile': 'good'},
            'pune': {'base': 85, 'variance': 25, 'profile': 'moderate'},
            'kolkata': {'base': 145, 'variance': 35, 'profile': 'high'},
            'chennai': {'base': 95, 'variance': 25, 'profile': 'moderate'},
            'hyderabad': {'base': 90, 'variance': 25, 'profile': 'moderate'},
            'beijing': {'base': 160, 'variance': 40, 'profile': 'severe'},
            'london': {'base': 65, 'variance': 20, 'profile': 'good'},
            'new york': {'base': 70, 'variance': 20, 'profile': 'good'},
            'los angeles': {'base': 85, 'variance': 25, 'profile': 'moderate'},
        }
        
        # Time-based variations
        hour = datetime.now().hour
        time_multiplier = 1.0
        
        if 6 <= hour <= 9:
            time_multiplier = 1.15
        elif 10 <= hour <= 16:
            time_multiplier = 0.90
        elif 18 <= hour <= 21:
            time_multiplier = 1.20
        else:
            time_multiplier = 0.95
            
        # Get pattern
        location_lower = location.lower()
        pattern = None
        
        for city, data in city_patterns.items():
            if city in location_lower:
                pattern = data
                break
                
        if not pattern:
            pattern = {'base': 100, 'variance': 30, 'profile': 'moderate'}
            
        # Calculate AQI
        base_aqi = pattern['base']
        variance = pattern['variance']
        aqi = int(base_aqi * time_multiplier + random.randint(-variance, variance))
        aqi = max(20, min(400, aqi))
        
        # Generate pollutants
        pollutants = self._generate_pollutants(aqi, pattern['profile'])
        dominant = max(pollutants.items(), key=lambda x: x[1])[0]
        
        return {
            'aqi': aqi,
            'pollutants': pollutants,
            'dominant_pollutant': dominant.upper(),
            'source': 'Simulated',
            'location': location,
            'timestamp': datetime.now().isoformat()
        }
        
    def _generate_pollutants(self, aqi: float, profile: str) -> Dict[str, int]:
        """Generate realistic pollutant distribution"""
        import random
        
        profiles = {
            'severe': {'pm25': 0.75, 'pm10': 0.85, 'o3': 0.25, 'no2': 0.35, 'so2': 0.15, 'co': 3.5},
            'high': {'pm25': 0.65, 'pm10': 0.75, 'o3': 0.30, 'no2': 0.30, 'so2': 0.18, 'co': 3.0},
            'moderate': {'pm25': 0.55, 'pm10': 0.65, 'o3': 0.35, 'no2': 0.28, 'so2': 0.15, 'co': 2.5},
            'good': {'pm25': 0.45, 'pm10': 0.55, 'o3': 0.40, 'no2': 0.25, 'so2': 0.12, 'co': 2.0}
        }
        
        multipliers = profiles.get(profile, profiles['moderate'])
        
        return {
            'pm25': int(aqi * multipliers['pm25'] + random.randint(-10, 10)),
            'pm10': int(aqi * multipliers['pm10'] + random.randint(-15, 15)),
            'o3': int(aqi * multipliers['o3'] + random.randint(-5, 5)),
            'no2': int(aqi * multipliers['no2'] + random.randint(-5, 8)),
            'so2': int(aqi * multipliers['so2'] + random.randint(-3, 5)),
            'co': int(aqi * multipliers['co'] + random.randint(-20, 30))
        }
        
    def _save_to_history(self, location: str, data: Dict):
        """Save AQI data to history"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            pollutants = data.get('pollutants', {})
            
            cursor.execute('''
                INSERT INTO aqi_history 
                (location, aqi, pm25, pm10, o3, no2, so2, co, dominant_pollutant, data_source, raw_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                location,
                data['aqi'],
                pollutants.get('pm25', 0),
                pollutants.get('pm10', 0),
                pollutants.get('o3', 0),
                pollutants.get('no2', 0),
                pollutants.get('so2', 0),
                pollutants.get('co', 0),
                data.get('dominant_pollutant', 'PM2.5'),
                data.get('source', 'Unknown'),
                json.dumps(data)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving to history: {e}")
            
    def get_historical_data(
        self,
        location: str,
        hours: int = 24
    ) -> List[Dict]:
        """
        Get historical AQI data
        
        Args:
            location: Location name
            hours: Number of hours of history
            
        Returns:
            List of historical data points
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff = datetime.now() - timedelta(hours=hours)
            
            cursor.execute('''
                SELECT aqi, pm25, pm10, o3, timestamp
                FROM aqi_history
                WHERE location = ? AND timestamp >= ?
                ORDER BY timestamp DESC
                LIMIT 100
            ''', (location, cutoff.isoformat()))
            
            results = cursor.fetchall()
            conn.close()
            
            return [
                {
                    'aqi': row[0],
                    'pm25': row[1],
                    'pm10': row[2],
                    'o3': row[3],
                    'timestamp': row[4]
                }
                for row in results
            ]
            
        except Exception as e:
            logger.error(f"Error fetching history: {e}")
            return []
            
    def cleanup_old_data(self, days: int = 30):
        """Clean up old cache and history data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff = datetime.now() - timedelta(days=days)
            
            cursor.execute('DELETE FROM api_cache WHERE timestamp < ?', (cutoff.isoformat(),))
            cursor.execute('DELETE FROM aqi_history WHERE timestamp < ?', (cutoff.isoformat(),))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Cleaned up data older than {days} days")
            
        except Exception as e:
            logger.error(f"Error cleaning up data: {e}")


# Singleton instance
_data_manager_instance = None


def get_data_manager() -> AQIDataManager:
    """Get or create data manager instance"""
    global _data_manager_instance
    if _data_manager_instance is None:
        _data_manager_instance = AQIDataManager()
    return _data_manager_instance
