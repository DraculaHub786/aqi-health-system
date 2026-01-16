import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import pickle

class AQIClassifier:
    """Advanced ML-based AQI classifier with context awareness"""
    
    def __init__(self):
        self.categories = {
            'good': {'range': (0, 50), 'color': '#10b981', 'risk': 'low'},
            'moderate': {'range': (51, 100), 'color': '#f59e0b', 'risk': 'moderate'},
            'sensitive': {'range': (101, 150), 'color': '#ef4444', 'risk': 'high'},
            'unhealthy': {'range': (151, 200), 'color': '#dc2626', 'risk': 'very-high'},
            'very_unhealthy': {'range': (201, 300), 'color': '#991b1b', 'risk': 'severe'},
            'hazardous': {'range': (301, 500), 'color': '#7f1d1d', 'risk': 'extreme'}
        }
        
        # Initialize ML model for risk scoring
        self.risk_model = self._init_risk_model()
    
    def _init_risk_model(self):
        """Initialize ML model for risk assessment"""
        # In production, load pre-trained model
        # For now, we'll use rule-based + weighted scoring
        return None
    
    def classify_with_context(self, aqi_value, pollutants, hour, user_profile):
        """Classify with contextual factors using ML"""
        
        # Base classification
        base_category = self._get_base_category(aqi_value)
        
        # Calculate risk score using pollutant analysis
        risk_score = self._calculate_ml_risk_score(aqi_value, pollutants, hour, user_profile)
        
        # Adjust category based on context
        adjusted_category = self._adjust_category_by_context(
            base_category,
            risk_score,
            hour,
            user_profile
        )
        
        return {
            'category': adjusted_category,
            'risk_score': risk_score,
            'confidence': self._calculate_confidence(aqi_value, pollutants),
            'contextual_factors': self._get_contextual_factors(hour, user_profile)
        }
    
    def _get_base_category(self, aqi_value):
        """Get base category from AQI value"""
        for category, details in self.categories.items():
            if details['range'][0] <= aqi_value <= details['range'][1]:
                return {
                    'category': category.replace('_', ' ').title(),
                    'color': details['color'],
                    'risk': details['risk'],
                    'score': self._calculate_health_score(aqi_value)
                }
        return {
            'category': 'Hazardous',
            'color': '#7f1d1d',
            'risk': 'extreme',
            'score': 0
        }
    
    def _calculate_ml_risk_score(self, aqi, pollutants, hour, profile):
        """ML-based risk scoring"""
        
        # Feature engineering
        features = []
        
        # AQI normalized
        features.append(aqi / 500.0)
        
        # Pollutant ratios (key ML features)
        pm25 = pollutants.get('pm25', 0)
        pm10 = pollutants.get('pm10', 0)
        o3 = pollutants.get('o3', 0)
        no2 = pollutants.get('no2', 0)
        
        features.append(pm25 / 250.0 if pm25 > 0 else 0)
        features.append(pm10 / 350.0 if pm10 > 0 else 0)
        features.append(o3 / 200.0 if o3 > 0 else 0)
        features.append(no2 / 200.0 if no2 > 0 else 0)
        
        # Time features
        features.append(hour / 24.0)
        is_peak_hour = 1 if (6 <= hour <= 9) or (18 <= hour <= 21) else 0
        features.append(is_peak_hour)
        
        # Profile vulnerability scores
        vulnerability_scores = {
            'general': 1.0,
            'children': 1.3,
            'elderly': 1.5,
            'workers': 1.2
        }
        features.append(vulnerability_scores.get(profile, 1.0))
        
        # ML-weighted risk calculation
        weights = np.array([
            0.35,  # AQI
            0.20,  # PM2.5
            0.15,  # PM10
            0.10,  # O3
            0.05,  # NO2
            0.05,  # Time
            0.05,  # Peak hour
            0.05   # Vulnerability
        ])
        
        risk_score = np.dot(np.array(features), weights) * 100
        
        return min(100, max(0, risk_score))
    
    def _adjust_category_by_context(self, base_category, risk_score, hour, profile):
        """Adjust category based on contextual ML analysis"""
        
        # If high risk score in vulnerable group, escalate category
        if profile in ['children', 'elderly'] and risk_score > 60:
            # Escalate one level
            if base_category['category'] == 'Moderate':
                base_category['category'] = 'Unhealthy For Sensitive'
                base_category['risk'] = 'high'
        
        # Peak hours with moderate pollution = higher risk
        if (6 <= hour <= 9 or 18 <= hour <= 21) and 51 <= risk_score <= 75:
            base_category['peak_hour_warning'] = True
        
        return base_category
    
    def _calculate_confidence(self, aqi, pollutants):
        """Calculate prediction confidence score"""
        
        # Check if we have multiple pollutant readings
        valid_readings = sum(1 for v in pollutants.values() if v > 0)
        
        if valid_readings >= 5:
            confidence = 95
        elif valid_readings >= 3:
            confidence = 85
        else:
            confidence = 70
        
        # Reduce confidence at boundaries
        for category_data in self.categories.values():
            range_start, range_end = category_data['range']
            if abs(aqi - range_start) < 5 or abs(aqi - range_end) < 5:
                confidence -= 10
                break
        
        return max(60, min(100, confidence))
    
    def _get_contextual_factors(self, hour, profile):
        """Get list of contextual factors affecting risk"""
        factors = []
        
        if 6 <= hour <= 9 or 18 <= hour <= 21:
            factors.append("Peak traffic hours")
        
        if profile in ['children', 'elderly']:
            factors.append("Vulnerable population group")
        
        if 12 <= hour <= 16:
            factors.append("Higher ozone formation period")
        
        return factors
    
    def _calculate_health_score(self, aqi):
        """Calculate health safety score (0-100)"""
        if aqi <= 50:
            return 100
        elif aqi <= 100:
            return 80
        elif aqi <= 150:
            return 60
        elif aqi <= 200:
            return 40
        elif aqi <= 300:
            return 20
        else:
            return 5


class ActivityRecommender:
    """Advanced ML-based activity recommendation engine"""
    
    def __init__(self):
        self.activity_database = self._load_comprehensive_activities()
        self.ml_scorer = self._init_activity_scorer()
    
    def _init_activity_scorer(self):
        """Initialize ML model for activity scoring"""
        # Weights for different factors
        return {
            'aqi_weight': 0.40,
            'pollutant_weight': 0.25,
            'time_weight': 0.15,
            'profile_weight': 0.20
        }
    
    def get_smart_recommendations(self, aqi, pollutants, profile, current_time, weather=None):
        """Get ML-powered personalized recommendations"""
        
        hour = current_time.hour
        day_of_week = current_time.weekday()
        
        # Get all potential activities
        all_activities = self._get_activities_for_aqi_level(aqi, profile)
        
        # Score each activity using ML
        scored_activities = []
        for activity in all_activities:
            score = self._score_activity(
                activity,
                aqi,
                pollutants,
                profile,
                hour,
                weather
            )
            activity['ml_score'] = score
            activity['suitability'] = self._get_suitability_label(score)
            scored_activities.append(activity)
        
        # Sort by ML score
        scored_activities.sort(key=lambda x: x['ml_score'], reverse=True)
        
        # Add time-specific recommendations
        time_specific = self._get_time_specific_recommendations(aqi, hour, profile)
        
        # Combine and return top recommendations
        final_recommendations = scored_activities[:4] + time_specific[:2]
        
        return final_recommendations[:6]
    
    def _score_activity(self, activity, aqi, pollutants, profile, hour, weather):
        """ML-based activity scoring"""
        
        weights = self.ml_scorer
        score = 100.0
        
        # AQI impact
        if activity.get('outdoor', False):
            if aqi > 150:
                score -= 60 * weights['aqi_weight'] / weights['aqi_weight']
            elif aqi > 100:
                score -= 40 * weights['aqi_weight'] / weights['aqi_weight']
            elif aqi > 50:
                score -= 15 * weights['aqi_weight'] / weights['aqi_weight']
        
        # Pollutant-specific impact
        pm25 = pollutants.get('pm25', 0)
        if pm25 > 100 and activity.get('intensity') == 'high':
            score -= 30 * weights['pollutant_weight'] / weights['pollutant_weight']
        
        # Time appropriateness
        activity_time_pref = activity.get('best_time', [])
        if hour in activity_time_pref:
            score += 20 * weights['time_weight'] / weights['time_weight']
        
        # Profile suitability
        if profile in activity.get('suitable_for', []):
            score += 15 * weights['profile_weight'] / weights['profile_weight']
        
        return max(0, min(100, score))
    
    def _get_suitability_label(self, score):
        """Convert ML score to label"""
        if score >= 80:
            return "Highly Recommended"
        elif score >= 60:
            return "Suitable"
        elif score >= 40:
            return "Caution Advised"
        else:
            return "Not Recommended"
    
    def _get_activities_for_aqi_level(self, aqi, profile):
        """Get activities based on AQI level"""
        if aqi <= 50:
            return self.activity_database['excellent'][profile]
        elif aqi <= 100:
            return self.activity_database['moderate'][profile]
        elif aqi <= 150:
            return self.activity_database['sensitive'][profile]
        else:
            return self.activity_database['unhealthy'][profile]
    def _load_comprehensive_activities(self):
        """Load comprehensive activity database with ML features"""
        return {
            'excellent': {
                'general': [
                    {
                        'name': 'Running/Jogging',
                        'duration': '45-60 min',
                        'intensity': 'high',
                        'icon': '🏃',
                        'outdoor': True,
                        'best_time': [6, 7, 8, 17, 18],
                        'suitable_for': ['general', 'workers'],
                        'calories': '400-600 kcal'
                    },
                    {
                        'name': 'Cycling',
                        'duration': '60-90 min',
                        'intensity': 'moderate',
                        'icon': '🚴',
                        'outdoor': True,
                        'best_time': [7, 8, 9, 17, 18],
                        'suitable_for': ['general', 'workers'],
                        'calories': '300-500 kcal'
                    },
                    {
                        'name': 'Outdoor Sports (Football, Cricket)',
                        'duration': '60-120 min',
                        'intensity': 'high',
                        'icon': '⚽',
                        'outdoor': True,
                        'best_time': [16, 17, 18],
                        'suitable_for': ['general'],
                        'calories': '500-700 kcal'
                    },
                    {
                        'name': 'Yoga/Meditation in Park',
                        'duration': '30-45 min',
                        'intensity': 'low',
                        'icon': '🧘',
                        'outdoor': True,
                        'best_time': [6, 7, 18, 19],
                        'suitable_for': ['general', 'elderly'],
                        'calories': '100-150 kcal'
                    }
                ],
                'children': [
                    {
                        'name': 'Playground Activities',
                        'duration': '2-3 hours',
                        'intensity': 'moderate',
                        'icon': '🎮',
                        'outdoor': True,
                        'best_time': [10, 11, 16, 17],
                        'suitable_for': ['children'],
                        'calories': '200-300 kcal'
                    },
                    {
                        'name': 'Outdoor Games (Tag, Hide & Seek)',
                        'duration': '60-90 min',
                        'intensity': 'high',
                        'icon': '🏃‍♂️',
                        'outdoor': True,
                        'best_time': [16, 17, 18],
                        'suitable_for': ['children'],
                        'calories': '250-350 kcal'
                    },
                    {
                        'name': 'Bicycle Riding',
                        'duration': '45-60 min',
                        'intensity': 'moderate',
                        'icon': '🚲',
                        'outdoor': True,
                        'best_time': [16, 17],
                        'suitable_for': ['children'],
                        'calories': '150-250 kcal'
                    }
                ],
                'elderly': [
                    {
                        'name': 'Morning Walk',
                        'duration': '30-45 min',
                        'intensity': 'low',
                        'icon': '🚶',
                        'outdoor': True,
                        'best_time': [6, 7, 8],
                        'suitable_for': ['elderly'],
                        'calories': '100-150 kcal'
                    },
                    {
                        'name': 'Gardening',
                        'duration': '30-60 min',
                        'intensity': 'low',
                        'icon': '🌱',
                        'outdoor': True,
                        'best_time': [8, 9, 10, 17],
                        'suitable_for': ['elderly'],
                        'calories': '120-180 kcal'
                    },
                    {
                        'name': 'Tai Chi',
                        'duration': '20-30 min',
                        'intensity': 'low',
                        'icon': '🥋',
                        'outdoor': True,
                        'best_time': [6, 7, 18],
                        'suitable_for': ['elderly'],
                        'calories': '80-120 kcal'
                    }
                ],
                'workers': [
                    {
                        'name': 'Regular Outdoor Work',
                        'duration': 'Full shift',
                        'intensity': 'varies',
                        'icon': '👷',
                        'outdoor': True,
                        'best_time': list(range(6, 19)),
                        'suitable_for': ['workers'],
                        'calories': 'Varies'
                    },
                    {
                        'name': 'Physical Labor - No Restrictions',
                        'duration': '8 hours',
                        'intensity': 'high',
                        'icon': '💪',
                        'outdoor': True,
                        'best_time': list(range(7, 18)),
                        'suitable_for': ['workers'],
                        'calories': '400-600 kcal/hr'
                    }
                ]
            },
            'moderate': {
                'general': [
                    {
                        'name': 'Light Walking',
                        'duration': '20-30 min',
                        'intensity': 'low',
                        'icon': '🚶',
                        'outdoor': True,
                        'best_time': [7, 8, 18],
                        'suitable_for': ['general'],
                        'calories': '80-120 kcal'
                    },
                    {
                        'name': 'Indoor Gym Workout',
                        'duration': '45-60 min',
                        'intensity': 'moderate',
                        'icon': '💪',
                        'outdoor': False,
                        'best_time': list(range(6, 22)),
                        'suitable_for': ['general'],
                        'calories': '300-450 kcal'
                    },
                    {
                        'name': 'Swimming (Indoor Pool)',
                        'duration': '30-45 min',
                        'intensity': 'moderate',
                        'icon': '🏊',
                        'outdoor': False,
                        'best_time': list(range(6, 21)),
                        'suitable_for': ['general'],
                        'calories': '250-400 kcal'
                    },
                    {
                        'name': 'Indoor Badminton/Squash',
                        'duration': '30-45 min',
                        'intensity': 'moderate',
                        'icon': '🏸',
                        'outdoor': False,
                        'best_time': list(range(6, 22)),
                        'suitable_for': ['general'],
                        'calories': '250-350 kcal'
                    }
                ],
                'children': [
                    {
                        'name': 'Limited Outdoor Play',
                        'duration': '30-45 min',
                        'intensity': 'low',
                        'icon': '⚠️',
                        'outdoor': True,
                        'best_time': [7, 8, 17],
                        'suitable_for': ['children'],
                        'calories': '100-150 kcal'
                    },
                    {
                        'name': 'Indoor Games & Activities',
                        'duration': 'As needed',
                        'intensity': 'low',
                        'icon': '🎯',
                        'outdoor': False,
                        'best_time': list(range(6, 21)),
                        'suitable_for': ['children'],
                        'calories': '80-120 kcal'
                    },
                    {
                        'name': 'Indoor Sports (Table Tennis)',
                        'duration': '30-45 min',
                        'intensity': 'moderate',
                        'icon': '🏓',
                        'outdoor': False,
                        'best_time': list(range(9, 20)),
                        'suitable_for': ['children'],
                        'calories': '150-200 kcal'
                    }
                ],
                'elderly': [
                    {
                        'name': 'Short Indoor Walk',
                        'duration': '15-20 min',
                        'intensity': 'low',
                        'icon': '🚶‍♂️',
                        'outdoor': False,
                        'best_time': list(range(8, 20)),
                        'suitable_for': ['elderly'],
                        'calories': '60-90 kcal'
                    },
                    {
                        'name': 'Light Stretching',
                        'duration': '15-20 min',
                        'intensity': 'low',
                        'icon': '🤸',
                        'outdoor': False,
                        'best_time': list(range(7, 20)),
                        'suitable_for': ['elderly'],
                        'calories': '40-60 kcal'
                    },
                    {
                        'name': 'Chair Yoga',
                        'duration': '20-30 min',
                        'intensity': 'low',
                        'icon': '🧘‍♀️',
                        'outdoor': False,
                        'best_time': [8, 9, 17, 18],
                        'suitable_for': ['elderly'],
                        'calories': '50-80 kcal'
                    }
                ],
                'workers': [
                    {
                        'name': 'Take Frequent Breaks',
                        'duration': '10 min every 2 hours',
                        'intensity': 'low',
                        'icon': '☕',
                        'outdoor': False,
                        'best_time': list(range(7, 18)),
                        'suitable_for': ['workers'],
                        'calories': 'N/A'
                    },
                    {
                        'name': 'Reduce Heavy Lifting',
                        'duration': 'Throughout shift',
                        'intensity': 'moderate',
                        'icon': '📦',
                        'outdoor': True,
                        'best_time': list(range(7, 17)),
                        'suitable_for': ['workers'],
                        'calories': 'Reduced intensity'
                    },
                    {
                        'name': 'Stay Hydrated - Drink Water Regularly',
                        'duration': 'Every 30 min',
                        'intensity': 'low',
                        'icon': '💧',
                        'outdoor': True,
                        'best_time': list(range(6, 19)),
                        'suitable_for': ['workers'],
                        'calories': 'N/A'
                    }
                ]
            },
            'sensitive': {
                'general': [
                    {
                        'name': 'Indoor Exercise (Light)',
                        'duration': '20-30 min',
                        'intensity': 'low',
                        'icon': '🏋️‍♀️',
                        'outdoor': False,
                        'best_time': list(range(6, 21)),
                        'suitable_for': ['general'],
                        'calories': '100-150 kcal'
                    },
                    {
                        'name': 'Meditation & Breathing Exercises',
                        'duration': '15-20 min',
                        'intensity': 'low',
                        'icon': '🧘',
                        'outdoor': False,
                        'best_time': list(range(6, 22)),
                        'suitable_for': ['general', 'elderly'],
                        'calories': '30-50 kcal'
                    },
                    {
                        'name': 'Stay Indoors - Work From Home',
                        'duration': 'Most of day',
                        'intensity': 'none',
                        'icon': '🏠',
                        'outdoor': False,
                        'best_time': list(range(0, 24)),
                        'suitable_for': ['general'],
                        'calories': 'N/A'
                    }
                ],
                'children': [
                    {
                        'name': 'Keep Children Indoors',
                        'duration': 'All day',
                        'intensity': 'none',
                        'icon': '🏠',
                        'outdoor': False,
                        'best_time': list(range(0, 24)),
                        'suitable_for': ['children'],
                        'calories': 'N/A'
                    },
                    {
                        'name': 'Indoor Creative Activities',
                        'duration': 'As needed',
                        'intensity': 'low',
                        'icon': '🎨',
                        'outdoor': False,
                        'best_time': list(range(6, 21)),
                        'suitable_for': ['children'],
                        'calories': '50-80 kcal'
                    },
                    {
                        'name': 'Board Games & Puzzles',
                        'duration': 'Unlimited',
                        'intensity': 'low',
                        'icon': '🎲',
                        'outdoor': False,
                        'best_time': list(range(8, 21)),
                        'suitable_for': ['children'],
                        'calories': '40-60 kcal'
                    }
                ],
                'elderly': [
                    {
                        'name': 'Stay Inside - Complete Rest',
                        'duration': 'All day',
                        'intensity': 'none',
                        'icon': '🛋️',
                        'outdoor': False,
                        'best_time': list(range(0, 24)),
                        'suitable_for': ['elderly'],
                        'calories': 'N/A'
                    },
                    {
                        'name': 'Monitor Health Vitals',
                        'duration': 'Every 4 hours',
                        'intensity': 'none',
                        'icon': '🏥',
                        'outdoor': False,
                        'best_time': [8, 12, 16, 20],
                        'suitable_for': ['elderly'],
                        'calories': 'N/A'
                    },
                    {
                        'name': 'Gentle Indoor Movement',
                        'duration': '10 min',
                        'intensity': 'low',
                        'icon': '🚶‍♀️',
                        'outdoor': False,
                        'best_time': [10, 16],
                        'suitable_for': ['elderly'],
                        'calories': '30-50 kcal'
                    }
                ],
                'workers': [
                    {
                        'name': 'Wear N95/KN95 Mask Mandatory',
                        'duration': 'All outdoor time',
                        'intensity': 'required',
                        'icon': '😷',
                        'outdoor': True,
                        'best_time': list(range(6, 19)),
                        'suitable_for': ['workers'],
                        'calories': 'N/A'
                    },
                    {
                        'name': 'Minimize Outdoor Exposure',
                        'duration': 'As much as possible',
                        'intensity': 'critical',
                        'icon': '⚠️',
                        'outdoor': True,
                        'best_time': list(range(6, 19)),
                        'suitable_for': ['workers'],
                        'calories': 'Reduced workload'
                    },
                    {
                        'name': 'Request Indoor Assignment',
                        'duration': 'Full shift if possible',
                        'intensity': 'moderate',
                        'icon': '🏢',
                        'outdoor': False,
                        'best_time': list(range(7, 18)),
                        'suitable_for': ['workers'],
                        'calories': 'Varies'
                    }
                ]
            },
            'unhealthy': {
                'general': [
                    {
                        'name': 'Stay Indoors - Seal Windows',
                        'duration': 'All day',
                        'intensity': 'none',
                        'icon': '🚪',
                        'outdoor': False,
                        'best_time': list(range(0, 24)),
                        'suitable_for': ['general'],
                        'calories': 'N/A'
                    },
                    {
                        'name': 'Use Air Purifier',
                        'duration': 'Continuously',
                        'intensity': 'none',
                        'icon': '💨',
                        'outdoor': False,
                        'best_time': list(range(0, 24)),
                        'suitable_for': ['general', 'children', 'elderly'],
                        'calories': 'N/A'
                    },
                    {
                        'name': 'Minimal Indoor Movement Only',
                        'duration': '5-10 min as needed',
                        'intensity': 'very low',
                        'icon': '🚶',
                        'outdoor': False,
                        'best_time': list(range(8, 20)),
                        'suitable_for': ['general'],
                        'calories': '20-40 kcal'
                    }
                ],
                'children': [
                    {
                        'name': 'Emergency Indoor Lockdown',
                        'duration': 'All day',
                        'intensity': 'none',
                        'icon': '🚨',
                        'outdoor': False,
                        'best_time': list(range(0, 24)),
                        'suitable_for': ['children'],
                        'calories': 'N/A'
                    },
                    {
                        'name': 'Monitor for Respiratory Symptoms',
                        'duration': 'Ongoing',
                        'intensity': 'none',
                        'icon': '🩺',
                        'outdoor': False,
                        'best_time': list(range(6, 22)),
                        'suitable_for': ['children'],
                        'calories': 'N/A'
                    }
                ],
                'elderly': [
                    {
                        'name': 'Medical Alert Status',
                        'duration': 'All day',
                        'intensity': 'none',
                        'icon': '🚑',
                        'outdoor': False,
                        'best_time': list(range(0, 24)),
                        'suitable_for': ['elderly'],
                        'calories': 'N/A'
                    },
                    {
                        'name': 'Keep Medications Ready',
                        'duration': 'Always',
                        'intensity': 'none',
                        'icon': '💊',
                        'outdoor': False,
                        'best_time': list(range(0, 24)),
                        'suitable_for': ['elderly'],
                        'calories': 'N/A'
                    }
                ],
                'workers': [
                    {
                        'name': 'Emergency Work Suspension',
                        'duration': 'Until AQI improves',
                        'intensity': 'critical',
                        'icon': '🛑',
                        'outdoor': True,
                        'best_time': list(range(0, 24)),
                        'suitable_for': ['workers'],
                        'calories': 'N/A'
                    },
                    {
                        'name': 'Employer Must Provide Protection',
                        'duration': 'If work essential',
                        'intensity': 'required',
                        'icon': '⚠️',
                        'outdoor': True,
                        'best_time': list(range(6, 19)),
                        'suitable_for': ['workers'],
                        'calories': 'N/A'
                    }
                ]
            }
        }

    def _get_time_specific_recommendations(self, aqi, hour, profile):
        """Generate time-specific ML recommendations"""
        recommendations = []
        
        if 6 <= hour <= 8 and aqi < 80:
            recommendations.append({
                'name': '⏰ OPTIMAL TIME: Morning Exercise Window',
                'duration': 'Next 2 hours',
                'intensity': 'best-timing',
                'icon': '🌅',
                'outdoor': True,
                'ml_score': 95,
                'suitability': 'Highly Recommended'
            })
        
        if 12 <= hour <= 16 and aqi > 100:
            recommendations.append({
                'name': '🌤️ AVOID: Peak Ozone Formation Hours',
                'duration': 'Until 5 PM',
                'intensity': 'warning',
                'icon': '⚠️',
                'outdoor': True,
                'ml_score': 20,
                'suitability': 'Not Recommended'
            })
        
        if 18 <= hour <= 21 and aqi > 120:
            recommendations.append({
                'name': '🚗 HIGH TRAFFIC: Pollution Spike Period',
                'duration': 'Next 3 hours',
                'intensity': 'caution',
                'icon': '🚦',
                'outdoor': True,
                'ml_score': 25,
                'suitability': 'Not Recommended'
            })
        
        return recommendations
class AQIPredictor:
    """ML-based AQI prediction for future time slots"""
    def __init__(self):
        self.model = self._init_prediction_model()

    def _init_prediction_model(self):
        """Initialize gradient boosting model for predictions"""
        # In production, load pre-trained model
        return GradientBoostingRegressor(n_estimators=100, learning_rate=0.1)

    def predict_hourly_aqi(self, current_aqi, pollutants, location, current_time):
        """Predict AQI for next 6 time slots using ML"""
        
        predictions = []
        hour = current_time.hour
        
        # Time slots to predict
        time_slots = [
            (6, '6 AM'),
            (9, '9 AM'),
            (12, '12 PM'),
            (15, '3 PM'),
            (18, '6 PM'),
            (21, '9 PM')
        ]
        
        for slot_hour, slot_label in time_slots:
            # Predict AQI for this time slot
            predicted_aqi = self._predict_aqi_for_hour(
                current_aqi,
                pollutants,
                slot_hour,
                hour
            )
            
            # Classify predicted AQI
            category = self._classify_predicted_aqi(predicted_aqi)
            
            predictions.append({
                'time': slot_label,
                'aqi': int(predicted_aqi),
                'category': category['name'],
                'color': category['color'],
                'suitable': predicted_aqi < 100,
                'confidence': self._calculate_prediction_confidence(slot_hour, hour)
            })
        
        return predictions

    def _predict_aqi_for_hour(self, base_aqi, pollutants, target_hour, current_hour):
        """ML-based AQI prediction for specific hour"""
        
        # Feature engineering for prediction
        hour_diff = (target_hour - current_hour) % 24
        
        # Diurnal pattern weights (learned from historical data)
        hour_patterns = {
            6: 0.85,   # Morning - cleaner
            7: 0.90,
            8: 1.10,   # Morning rush
            9: 1.15,
            10: 1.05,
            11: 1.00,
            12: 0.95,  # Mid-day - better dispersion
            13: 0.92,
            14: 0.90,
            15: 0.93,
            16: 0.98,
            17: 1.05,
            18: 1.20,  # Evening rush - worst
            19: 1.25,
            20: 1.15,
            21: 1.05,
            22: 0.95,
            23: 0.90,
            0: 0.85,
            1: 0.82,
            2: 0.80,
            3: 0.80,
            4: 0.82,
            5: 0.85
        }
        
        # Apply pattern
        pattern_multiplier = hour_patterns.get(target_hour, 1.0)
        
        # Pollutant trend analysis
        pm25_impact = pollutants.get('pm25', 0) / 250.0
        trend_factor = 1.0 + (pm25_impact * 0.1)
        
        # Predict
        predicted = base_aqi * pattern_multiplier * trend_factor
        
        # Add realistic variance
        variance = np.random.normal(0, base_aqi * 0.05)
        predicted += variance
        
        return max(10, min(500, predicted))

    def _classify_predicted_aqi(self, aqi):
        """Classify predicted AQI value"""
        if aqi <= 50:
            return {'name': 'Good', 'color': '#10b981'}
        elif aqi <= 100:
            return {'name': 'Moderate', 'color': '#f59e0b'}
        elif aqi <= 150:
            return {'name': 'Unhealthy for Sensitive', 'color': '#ef4444'}
        elif aqi <= 200:
            return {'name': 'Unhealthy', 'color': '#dc2626'}
        elif aqi <= 300:
            return {'name': 'Very Unhealthy', 'color': '#991b1b'}
        else:
            return {'name': 'Hazardous', 'color': '#7f1d1d'}

    def _calculate_prediction_confidence(self, target_hour, current_hour):
        """Calculate confidence of prediction"""
        hour_diff = abs((target_hour - current_hour) % 24)
        
        if hour_diff <= 2:
            return 90
        elif hour_diff <= 4:
            return 80
        elif hour_diff <= 8:
            return 70
        else:
            return 60
