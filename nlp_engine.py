import random
from datetime import datetime

class NLPExplainer:
    """Advanced NLP engine with contextual understanding"""
    
    def __init__(self):
        self.templates = self._load_advanced_templates()
        self.context_analyzers = self._init_context_analyzers()
    
    def generate_contextual_explanation(self, aqi, pollutants, risk_level, profile, location, current_time):
        """Generate advanced contextual explanation using NLP"""
        
        # Analyze context
        context = self._analyze_full_context(aqi, pollutants, current_time, profile)
        
        # Generate explanation parts
        intro = self._generate_intro(aqi, risk_level, context)
        health_impact = self._generate_health_impact(aqi, pollutants, profile, context)
        pollutant_analysis = self._generate_pollutant_analysis(pollutants, aqi)
        time_context = self._generate_time_context(current_time, aqi, context)
        profile_specific = self._generate_profile_guidance(profile, aqi, risk_level)
        action_items = self._generate_action_items(aqi, profile, context)
        
        # Combine into natural narrative
        full_explanation = f"{intro} {health_impact} {pollutant_analysis} {time_context} {profile_specific} {action_items}"
        
        return full_explanation
    
    def _analyze_full_context(self, aqi, pollutants, current_time, profile):
        """Comprehensive context analysis"""
        context = {}
        
        # Time context
        hour = current_time.hour
        context['is_morning'] = 6 <= hour <= 10
        context['is_afternoon'] = 11 <= hour <= 16
        context['is_evening'] = 17 <= hour <= 21
        context['is_night'] = hour >= 22 or hour <= 5
        context['is_peak_hour'] = (7 <= hour <= 9) or (18 <= hour <= 20)
        
        # Pollution context
        context['severity'] = self._determine_severity(aqi)
        context['dominant_pollutant'] = max(pollutants.items(), key=lambda x: x[1])[0]
        context['pm25_high'] = pollutants.get('pm25', 0) > 75
        context['ozone_high'] = pollutants.get('o3', 0) > 100
        
        # Profile vulnerability
        context['high_vulnerability'] = profile in ['children', 'elderly']
        context['moderate_vulnerability'] = profile == 'workers'
        
        # Risk assessment
        context['emergency'] = aqi > 200
        context['high_alert'] = 150 < aqi <= 200
        context['caution'] = 100 < aqi <= 150
        context['safe'] = aqi <= 100
        
        return context
    
    def _generate_intro(self, aqi, risk_level, context):
        """Generate contextual introduction"""
        if context['emergency']:
            intros = [
                f"🚨 CRITICAL AIR QUALITY ALERT: The AQI has reached a hazardous level of {aqi}.",
                f"⚠️ HEALTH EMERGENCY: Air quality is in the hazardous range at AQI {aqi}.",
                f"🔴 SEVERE POLLUTION WARNING: Current AQI of {aqi} poses serious health risks to everyone."
            ]
        elif context['high_alert']:
            intros = [
                f"⚠️ The air quality is unhealthy with an AQI of {aqi}.",
                f"🟠 Air pollution levels are concerning today at AQI {aqi}.",
                f"⚡ Current AQI reading of {aqi} indicates unhealthy air conditions."
            ]
        elif context['caution']:
            intros = [
                f"🟡 Air quality is moderate to unhealthy at AQI {aqi}.",
                f"⚠️ The AQI stands at {aqi}, which may affect sensitive groups.",
                f"📊 Today's air quality index of {aqi} calls for caution."
            ]
        else:
            intros = [
                f"✅ Good news! Air quality is {risk_level['category'].lower()} with an AQI of {aqi}.",
                f"🌟 The air quality is excellent today at AQI {aqi}.",
                f"🌿 You're breathing clean air today with an AQI reading of {aqi}."
            ]
        
        return random.choice(intros)
    
    def _generate_health_impact(self, aqi, pollutants, profile, context):
        """Generate health impact analysis"""
        if context['emergency']:
            if profile == 'children':
                return "This is extremely dangerous for children. Respiratory systems are at severe risk, and symptoms like coughing, wheezing, and breathing difficulty are likely. All outdoor activities must cease immediately."
            elif profile == 'elderly':
                return "Seniors are in a medical emergency situation. Those with heart or lung conditions face life-threatening risks. Immediate indoor isolation with clean air is critical."
            elif profile == 'workers':
                return "Outdoor work poses extreme health hazards. Employers are legally required to halt non-essential operations. Essential workers must use full respiratory protection."
            else:
                return "Everyone will experience serious health effects. Healthy individuals will likely develop symptoms. Those with existing conditions are at severe risk."
        elif context['high_alert']:
            if profile == 'children':
                return "Children are particularly vulnerable at this pollution level. School activities should be moved indoors, and parents should monitor for any breathing difficulties or unusual fatigue."
            elif profile == 'elderly':
                return "Seniors with cardiovascular or respiratory conditions should remain indoors. Even healthy elderly individuals should minimize exertion and stay alert for symptoms like chest tightness or irregular breathing."
            elif profile == 'workers':
                return "Outdoor workers are experiencing prolonged exposure to harmful pollutants. Frequent breaks in filtered air environments and N95 mask usage are essential to prevent health complications."
            else:
                return "The general population will likely experience effects such as breathing discomfort, throat irritation, and increased heart rate with physical activity."
        
        elif context['caution']:
            if profile == 'children':
                return "Children with asthma or allergies should limit outdoor time. Healthy children can play outside but should avoid intense, prolonged activities."
            elif profile == 'elderly':
                return "Older adults should consider shorter outdoor sessions and opt for gentler activities. Those with pre-existing conditions should prioritize indoor environments."
            else:
                return "Sensitive individuals may experience minor respiratory symptoms. Most healthy people can maintain normal activities with some precautions."
        
        else:
            if context['is_morning']:
                return "This is the ideal time for outdoor activities. Your lungs can breathe easy, and you'll benefit from clean, fresh air during exercise."
            else:
                return "Everyone can safely enjoy outdoor activities without health concerns. This is excellent air quality for physical activity and recreation."

    def _generate_pollutant_analysis(self, pollutants, aqi):
        """Analyze specific pollutants"""
        pm25 = pollutants.get('pm25', 0)
        pm10 = pollutants.get('pm10', 0)
        o3 = pollutants.get('o3', 0)
        no2 = pollutants.get('no2', 0)
        
        analyses = []
        
        if pm25 > 100:
            analyses.append(f"Fine particulate matter (PM2.5) is critically high at {pm25} µg/m³. These microscopic particles can penetrate deep into lungs and bloodstream, causing serious cardiovascular and respiratory issues.")
        elif pm25 > 50:
            analyses.append(f"PM2.5 levels are elevated at {pm25} µg/m³. These fine particles from vehicle emissions and combustion can irritate airways and reduce lung function.")
        
        if pm10 > 150:
            analyses.append(f"Coarse particulate matter (PM10) is very high at {pm10} µg/m³, primarily from dust and construction activities.")
        
        if o3 > 100:
            analyses.append(f"Ground-level ozone is elevated at {o3} ppb. Ozone irritates airways and can trigger asthma attacks, especially during afternoon hours when sunlight is strongest.")
        
        if no2 > 100:
            analyses.append(f"Nitrogen dioxide from vehicle exhaust is concerning at {no2} ppb, indicating heavy traffic pollution.")
        
        if not analyses:
            analyses.append("All major pollutants are within safe limits. The air composition is healthy for breathing.")
        
        return " ".join(analyses)

    def _generate_time_context(self, current_time, aqi, context):
        """Generate time-specific context"""
        hour = current_time.hour
        
        if context['is_morning'] and aqi < 100:
            return "Morning hours typically offer the best air quality of the day as pollutants from the previous day have dispersed overnight. This is an optimal window for outdoor exercise."
        
        elif context['is_peak_hour'] and aqi > 100:
            return "You're experiencing peak traffic hours when vehicle emissions dramatically increase pollution levels. Air quality typically worsens by 15-25% during these periods."
        
        elif context['is_afternoon'] and context['ozone_high']:
            return "Afternoon sunlight triggers chemical reactions that form ground-level ozone. This is the worst time for outdoor activities when ozone levels are high."
        
        elif context['is_evening']:
            return "Evening hours often see a second pollution spike from rush-hour traffic combined with cooler temperatures that trap pollutants closer to ground level."
        
        return "Air quality can vary significantly throughout the day. Check for updates if you're planning extended outdoor activities."

    def _generate_profile_guidance(self, profile, aqi, risk_level):
        """Generate profile-specific guidance"""
        if profile == 'children':
            if aqi > 150:
                return "For parents and schools: All outdoor recess and sports must be cancelled. Children should remain in air-conditioned or filtered indoor spaces. Watch for symptoms like persistent coughing or rapid breathing."
            elif aqi > 100:
                return "Parents should limit children's outdoor play to 30-45 minutes maximum. Avoid intense activities like running or sports that increase breathing rate."
            else:
                return "Great conditions for kids to play outside! This is a perfect opportunity for outdoor activities, sports, and exploration."
        
        elif profile == 'elderly':
            if aqi > 150:
                return "Seniors must stay indoors. Keep all necessary medications readily accessible. If you experience chest pain, dizziness, or breathing difficulty, seek immediate medical attention."
            elif aqi > 100:
                return "Older adults should opt for gentle indoor activities. If you must go outside, keep it brief and avoid any physical exertion."
            else:
                return "Excellent conditions for seniors to enjoy outdoor walks, gardening, or other leisure activities. The clean air supports healthy aging and cardiovascular function."
        
        elif profile == 'workers':
            if aqi > 150:
                return "Employers have a legal duty of care: Provide N95 masks, ensure adequate breaks in clean environments, and consider temporary work modifications. Workers should document any health symptoms."
            elif aqi > 100:
                return "Outdoor workers should increase break frequency (10 minutes every 2 hours), stay well-hydrated, and wear appropriate respiratory protection for heavy labor."
            else:
                return "Normal outdoor work conditions. No special precautions needed, but workers should still follow standard safety protocols."
        
        else:
            if aqi > 150:
                return "Everyone should modify daily routines: work from home if possible, postpone outdoor errands, and create a clean air sanctuary indoors."
            elif aqi > 100:
                return "Consider rescheduling non-essential outdoor activities. If you must be outside, limit duration and intensity of physical activity."
            else:
                return "Perfect conditions for your normal routine. Enjoy outdoor activities, exercise, and fresh air without concern."

    def _generate_action_items(self, aqi, profile, context):
        """Generate specific action items"""
        actions = []
        
        if context['emergency']:
            actions.append("IMMEDIATE ACTIONS: Close all windows and doors. Run air purifiers on maximum setting. Cancel all outdoor plans.")
            actions.append("Create a 'clean room' with an air purifier for family members to retreat to.")
        elif context['high_alert']:
            actions.append("Keep windows closed during peak pollution hours. Use air purifiers in bedrooms and living areas.")
            actions.append("Monitor air quality apps for real-time updates before going outside.")
        elif context['caution']:
            actions.append("Consider wearing KN95/N95 masks for extended outdoor activities.")
            actions.append("Stay hydrated - drink extra water to help your body process pollutants.")
        else:
            actions.append("Take advantage of this clean air! Open windows for natural ventilation.")
            actions.append("This is an ideal time for outdoor exercise and activities you've been postponing.")
        
        return " ".join(actions)

    def _determine_severity(self, aqi):
        """Determine severity level"""
        if aqi > 200:
            return 'critical'
        elif aqi > 150:
            return 'severe'
        elif aqi > 100:
            return 'moderate'
        elif aqi > 50:
            return 'slight'
        else:
            return 'minimal'

    def _load_advanced_templates(self):
        """Load advanced NLP templates"""
        # Templates with contextual placeholders
        return {}

    def _init_context_analyzers(self):
        """Initialize context analysis functions"""
        return {}