import os
import logging
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Optional
import pickle

logger = logging.getLogger(__name__)

class KaggleAQIDataset:
    """Manager for Kaggle AQI datasets"""
    
    def __init__(self, data_dir: str = "data/kaggle"):
        """
        Initialize Kaggle dataset manager
        
        Args:
            data_dir: Directory to store downloaded datasets
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.datasets_info = {
            'global_aqi': {
                'name': 'global-air-pollution-dataset',
                'owner': 'hasibalmuzdadid',
                'file': 'global air pollution dataset.csv',
                'local_path': self.data_dir / 'global_air_pollution.csv'
            },
            'city_aqi': {
                'name': 'air-quality-data-in-india',
                'owner': 'rohanrao',
                'file': 'city_day.csv',
                'local_path': self.data_dir / 'city_day.csv'
            },
            'us_pollution': {
                'name': 'us-pollution-data-200-to-2022',
                'owner': 'guslovesmath',
                'file': 'pollution_us_2000_2022.csv',
                'local_path': self.data_dir / 'us_pollution.csv'
            }
        }
        
        self.dataset_cache = {}
        self.recommendations_db = None
        
    def check_kaggle_api(self) -> bool:
        """Check if Kaggle API is configured"""
        try:
            import kaggle
            # Try to authenticate
            kaggle.api.authenticate()
            logger.info("✓ Kaggle API authenticated successfully")
            return True
        except Exception as e:
            logger.warning(f"Kaggle API not configured: {e}")
            logger.info("To use Kaggle datasets:")
            logger.info("1. Create account at kaggle.com")
            logger.info("2. Go to Account > API > Create New API Token")
            logger.info("3. Place kaggle.json in ~/.kaggle/ (Linux/Mac) or C:\\Users\\<user>\\.kaggle\\ (Windows)")
            return False
    
    def download_dataset(self, dataset_key: str, force: bool = False) -> bool:
        """
        Download dataset from Kaggle
        
        Args:
            dataset_key: Key from datasets_info
            force: Force re-download even if exists
            
        Returns:
            Success status
        """
        if dataset_key not in self.datasets_info:
            logger.error(f"Unknown dataset: {dataset_key}")
            return False
        
        dataset = self.datasets_info[dataset_key]
        
        # Check if already downloaded
        if dataset['local_path'].exists() and not force:
            logger.info(f"Dataset already exists: {dataset['local_path']}")
            return True
        
        try:
            import kaggle
            
            # Download dataset
            dataset_id = f"{dataset['owner']}/{dataset['name']}"
            logger.info(f"Downloading {dataset_id}...")
            
            kaggle.api.dataset_download_files(
                dataset_id,
                path=self.data_dir,
                unzip=True,
                quiet=False
            )
            
            # Rename if needed
            downloaded_file = self.data_dir / dataset['file']
            if downloaded_file.exists() and downloaded_file != dataset['local_path']:
                downloaded_file.rename(dataset['local_path'])
            
            logger.info(f"✓ Downloaded: {dataset['local_path']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {dataset_key}: {e}")
            return False
    
    def download_all_datasets(self, force: bool = False) -> Dict[str, bool]:
        """Download all configured datasets"""
        results = {}
        
        if not self.check_kaggle_api():
            logger.warning("Kaggle API not available. Using sample data.")
            self._create_sample_datasets()
            return {'sample_data': True}
        
        for key in self.datasets_info:
            results[key] = self.download_dataset(key, force)
        
        return results
    
    def load_dataset(self, dataset_key: str) -> Optional[pd.DataFrame]:
        """
        Load dataset into pandas DataFrame
        
        Args:
            dataset_key: Dataset identifier
            
        Returns:
            DataFrame or None
        """
        if dataset_key in self.dataset_cache:
            return self.dataset_cache[dataset_key]
        
        if dataset_key not in self.datasets_info:
            logger.error(f"Unknown dataset: {dataset_key}")
            return None
        
        local_path = self.datasets_info[dataset_key]['local_path']
        
        if not local_path.exists():
            logger.warning(f"Dataset not found: {local_path}")
            # Return None instead of trying to download (will use sample data)
            return None
        
        try:
            df = pd.read_csv(local_path)
            logger.info(f"✓ Loaded {len(df)} rows from {dataset_key}")
            self.dataset_cache[dataset_key] = df
            return df
        except Exception as e:
            logger.error(f"Failed to load {dataset_key}: {e}")
            return None
    
    def build_recommendations_database(self) -> Dict:
        """
        Build comprehensive recommendations database from Kaggle datasets
        
        Returns:
            Recommendations database
        """
        logger.info("Building recommendations database from Kaggle datasets...")
        
        recommendations = {
            'aqi_ranges': self._extract_aqi_patterns(),
            'pollutant_health_impacts': self._extract_pollutant_impacts(),
            'seasonal_patterns': self._extract_seasonal_patterns(),
            'city_specific_advice': self._extract_city_patterns(),
            'activity_safety_thresholds': self._generate_activity_thresholds(),
            'health_statistics': self._calculate_health_statistics()
        }
        
        # Save to cache
        cache_file = self.data_dir / 'recommendations_db.pkl'
        with open(cache_file, 'wb') as f:
            pickle.dump(recommendations, f)
        
        logger.info(f"✓ Recommendations database saved to {cache_file}")
        self.recommendations_db = recommendations
        return recommendations
    
    def _extract_aqi_patterns(self) -> Dict:
        """Extract AQI patterns from datasets"""
        patterns = {
            'good': {'range': (0, 50), 'activities': ['all outdoor activities', 'exercise', 'running', 'cycling']},
            'moderate': {'range': (51, 100), 'activities': ['most outdoor activities', 'light exercise'], 
                        'precautions': ['sensitive groups should limit prolonged outdoor exertion']},
            'unhealthy_sensitive': {'range': (101, 150), 'activities': ['reduced outdoor activities'],
                                   'precautions': ['sensitive groups avoid prolonged outdoor exertion', 'others limit prolonged outdoor activities']},
            'unhealthy': {'range': (151, 200), 'activities': ['indoor activities only'],
                         'precautions': ['everyone avoid prolonged outdoor exertion', 'wear N95 mask if outside']},
            'very_unhealthy': {'range': (201, 300), 'activities': ['stay indoors'],
                              'precautions': ['everyone avoid outdoor activities', 'use air purifier', 'keep windows closed']},
            'hazardous': {'range': (301, 500), 'activities': ['emergency conditions - stay indoors'],
                         'precautions': ['everyone remain indoors', 'air purifier essential', 'seal windows']}
        }
        
        # Try to enhance with real data
        try:
            df = self.load_dataset('global_aqi')
            if df is not None and 'AQI Value' in df.columns:
                # Calculate actual statistics from data
                for category, info in patterns.items():
                    aqi_min, aqi_max = info['range']
                    subset = df[(df['AQI Value'] >= aqi_min) & (df['AQI Value'] <= aqi_max)]
                    if len(subset) > 0:
                        info['frequency'] = len(subset) / len(df)
                        info['avg_aqi'] = subset['AQI Value'].mean()
        except Exception as e:
            logger.warning(f"Could not enhance patterns with data: {e}")
        
        return patterns
    
    def _extract_pollutant_impacts(self) -> Dict:
        """Extract pollutant health impact information"""
        impacts = {
            'PM2.5': {
                'name': 'Fine Particulate Matter',
                'sources': ['vehicle emissions', 'industrial processes', 'wildfires', 'dust'],
                'health_effects': [
                    'respiratory irritation',
                    'cardiovascular issues',
                    'aggravated asthma',
                    'reduced lung function'
                ],
                'safe_level': 12.0,  # µg/m³ annual average
                'high_risk_groups': ['children', 'elderly', 'respiratory conditions', 'heart disease']
            },
            'PM10': {
                'name': 'Coarse Particulate Matter',
                'sources': ['dust', 'construction', 'agriculture', 'road dust'],
                'health_effects': [
                    'respiratory tract irritation',
                    'coughing',
                    'difficulty breathing',
                    'aggravated asthma'
                ],
                'safe_level': 50.0,
                'high_risk_groups': ['children', 'elderly', 'asthma sufferers']
            },
            'NO2': {
                'name': 'Nitrogen Dioxide',
                'sources': ['vehicle emissions', 'power plants', 'industrial facilities'],
                'health_effects': [
                    'airway inflammation',
                    'reduced immunity to lung infections',
                    'aggravated asthma',
                    'increased susceptibility to allergens'
                ],
                'safe_level': 40.0,
                'high_risk_groups': ['asthma sufferers', 'children']
            },
            'O3': {
                'name': 'Ozone',
                'sources': ['chemical reaction of pollutants in sunlight'],
                'health_effects': [
                    'chest pain',
                    'coughing',
                    'throat irritation',
                    'congestion',
                    'reduced lung function'
                ],
                'safe_level': 100.0,
                'high_risk_groups': ['children', 'outdoor workers', 'asthma sufferers']
            },
            'CO': {
                'name': 'Carbon Monoxide',
                'sources': ['vehicle emissions', 'incomplete combustion'],
                'health_effects': [
                    'reduced oxygen delivery to organs',
                    'headaches',
                    'dizziness',
                    'fatigue',
                    'chest pain in heart patients'
                ],
                'safe_level': 9.0,
                'high_risk_groups': ['heart disease', 'pregnant women', 'elderly']
            },
            'SO2': {
                'name': 'Sulfur Dioxide',
                'sources': ['fossil fuel combustion', 'industrial processes'],
                'health_effects': [
                    'respiratory problems',
                    'breathing difficulties',
                    'aggravated asthma',
                    'eye irritation'
                ],
                'safe_level': 20.0,
                'high_risk_groups': ['asthma sufferers', 'children', 'elderly']
            }
        }
        
        return impacts
    
    def _extract_seasonal_patterns(self) -> Dict:
        """Extract seasonal AQI patterns"""
        patterns = {
            'winter': {
                'months': [12, 1, 2],
                'common_issues': ['increased PM2.5 from heating', 'temperature inversions trap pollutants'],
                'advice': 'Monitor AQI closely during cold days, use indoor air purifier'
            },
            'spring': {
                'months': [3, 4, 5],
                'common_issues': ['pollen mixed with pollutants', 'dust storms in some regions'],
                'advice': 'Watch for combined allergen and pollution alerts'
            },
            'summer': {
                'months': [6, 7, 8],
                'common_issues': ['increased ozone from sunlight', 'wildfires in dry regions'],
                'advice': 'Avoid outdoor exercise during peak afternoon hours'
            },
            'fall': {
                'months': [9, 10, 11],
                'common_issues': ['crop burning in agricultural areas', 'temperature inversions begin'],
                'advice': 'Watch for smoke from agricultural burning'
            }
        }
        
        return patterns
    
    def _extract_city_patterns(self) -> Dict:
        """Extract city-specific patterns if data available"""
        city_data = {}
        
        try:
            df = self.load_dataset('city_aqi')
            if df is not None and 'City' in df.columns:
                # Aggregate by city
                cities = df.groupby('City').agg({
                    'PM2.5': 'mean',
                    'PM10': 'mean',
                    'AQI': 'mean' if 'AQI' in df.columns else lambda x: None
                }).round(2)
                
                city_data = cities.to_dict(orient='index')
        except Exception as e:
            logger.warning(f"Could not extract city patterns: {e}")
        
        return city_data
    
    def _generate_activity_thresholds(self) -> Dict:
        """Generate activity safety thresholds"""
        return {
            'intense_exercise': {'max_aqi': 50, 'examples': ['running', 'HIIT', 'competitive sports']},
            'moderate_exercise': {'max_aqi': 100, 'examples': ['jogging', 'cycling', 'tennis']},
            'light_exercise': {'max_aqi': 150, 'examples': ['walking', 'yoga', 'stretching']},
            'outdoor_leisure': {'max_aqi': 100, 'examples': ['picnics', 'outdoor dining', 'gardening']},
            'children_play': {'max_aqi': 75, 'examples': ['playground', 'outdoor games']},
            'elderly_outdoor': {'max_aqi': 75, 'examples': ['walking', 'sitting outside']},
            'indoor_activities': {'aqi_above': 150, 'examples': ['gym workout', 'indoor sports', 'home exercise']}
        }
    
    def _calculate_health_statistics(self) -> Dict:
        """Calculate health-related statistics from data"""
        stats = {
            'global_summary': 'Based on WHO data, 99% of global population breathes air exceeding quality guidelines',
            'pm25_deaths_annual': '4.2 million premature deaths attributed to outdoor air pollution (WHO)',
            'economic_impact': 'Air pollution costs global economy $5 trillion annually',
            'vulnerable_population': '2.4 billion people exposed to dangerous pollution levels'
        }
        
        return stats
    
    def _create_sample_datasets(self):
        """Create sample datasets if Kaggle API not available"""
        logger.info("Creating sample AQI dataset for demonstration...")
        
        # Create sample global AQI data
        sample_data = {
            'City': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 
                    'Delhi', 'Mumbai', 'Beijing', 'Tokyo', 'London'],
            'Country': ['USA', 'USA', 'USA', 'USA', 'USA', 
                       'India', 'India', 'China', 'Japan', 'UK'],
            'AQI Value': [45, 89, 52, 67, 73, 178, 156, 134, 38, 42],
            'PM2.5': [10.5, 22.3, 12.1, 16.8, 18.2, 89.5, 78.3, 67.4, 8.2, 9.5],
            'PM10': [25.3, 45.6, 28.7, 35.2, 38.9, 145.2, 132.6, 115.3, 18.5, 22.1],
            'NO2': [18.5, 32.1, 21.3, 25.6, 28.3, 52.3, 48.7, 45.2, 15.2, 19.8],
            'O3': [35.2, 68.5, 42.1, 51.3, 58.6, 28.5, 32.1, 38.9, 45.2, 38.7]
        }
        
        df = pd.DataFrame(sample_data)
        sample_file = self.data_dir / 'global_air_pollution.csv'
        df.to_csv(sample_file, index=False)
        logger.info(f"✓ Created sample dataset: {sample_file}")
        
        # Update paths
        self.datasets_info['global_aqi']['local_path'] = sample_file
    
    def get_recommendations_for_aqi(self, aqi: float, pollutants: Dict = None) -> Dict:
        """
        Get personalized recommendations based on AQI and pollutants
        
        Args:
            aqi: Current AQI value
            pollutants: Dictionary of pollutant levels
            
        Returns:
            Comprehensive recommendations
        """
        if self.recommendations_db is None:
            self.build_recommendations_database()
        
        # Determine AQI category
        category = None
        for cat, info in self.recommendations_db['aqi_ranges'].items():
            if info['range'][0] <= aqi <= info['range'][1]:
                category = cat
                break
        
        if category is None:
            category = 'hazardous' if aqi > 300 else 'good'
        
        recommendations = {
            'aqi': aqi,
            'category': category,
            'category_info': self.recommendations_db['aqi_ranges'][category],
            'activities': self.recommendations_db['aqi_ranges'][category].get('activities', []),
            'precautions': self.recommendations_db['aqi_ranges'][category].get('precautions', []),
            'activity_thresholds': self.recommendations_db['activity_safety_thresholds']
        }
        
        # Add pollutant-specific advice
        if pollutants:
            pollutant_advice = []
            for pollutant, level in pollutants.items():
                if pollutant in self.recommendations_db['pollutant_health_impacts']:
                    impact_info = self.recommendations_db['pollutant_health_impacts'][pollutant]
                    if level > impact_info['safe_level']:
                        pollutant_advice.append({
                            'pollutant': pollutant,
                            'level': level,
                            'safe_level': impact_info['safe_level'],
                            'health_effects': impact_info['health_effects'],
                            'high_risk_groups': impact_info['high_risk_groups']
                        })
            
            recommendations['pollutant_alerts'] = pollutant_advice
        
        return recommendations


def initialize_kaggle_datasets():
    """Initialize and download Kaggle datasets"""
    manager = KaggleAQIDataset()
    
    # Try to download datasets
    results = manager.download_all_datasets()
    
    # Build recommendations database
    db = manager.build_recommendations_database()
    
    return manager, db


if __name__ == "__main__":
    # Test the dataset manager
    manager, db = initialize_kaggle_datasets()
    
    # Test recommendations
    recs = manager.get_recommendations_for_aqi(
        aqi=125,
        pollutants={'PM2.5': 55.5, 'NO2': 45.2}
    )
    
    print("Sample Recommendations:")
    print(json.dumps(recs, indent=2))
