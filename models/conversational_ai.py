"""
Advanced Conversational AI Engine for AQI Health System
Uses state-of-the-art transformers for natural language understanding
Provides human-like interactions with context awareness and memory
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import random
import json
from collections import deque

# Core NLP Libraries
try:
    from transformers import (
        pipeline,
        AutoTokenizer,
        AutoModelForSequenceClassification,
        AutoModelForQuestionAnswering,
        AutoModelForCausalLM,
        ConversationPipeline,
        Conversation
    )
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers not available. Install with: pip install transformers torch")

try:
    from sentence_transformers import SentenceTransformer, util
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("Sentence-transformers not available")

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConversationalAIEngine:
    """
    Advanced Conversational AI using Hugging Face Transformers
    Provides human-like chat interactions with deep NLP understanding
    """
    
    def __init__(self, use_gpu: bool = False):
        """
        Initialize conversational AI engine
        
        Args:
            use_gpu: Whether to use GPU acceleration (default: False for compatibility)
        """
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing Conversational AI on {self.device}")
        
        # Conversation memory (last 10 exchanges)
        self.conversation_history = deque(maxlen=10)
        self.user_context = {}
        
        # Initialize models
        self._init_models()
        
        # Knowledge base for AQI-specific information
        self.aqi_knowledge = self._build_aqi_knowledge_base()
        
        # Sentiment analyzer
        self.sentiment_analyzer = SentimentIntensityAnalyzer() if VADER_AVAILABLE else None
        
    def _init_models(self):
        """Initialize all transformer models"""
        if not TRANSFORMERS_AVAILABLE:
            logger.error("Transformers not available! Install with: pip install transformers torch")
            self.qa_pipeline = None
            self.intent_classifier = None
            self.conversational_model = None
            self.sentence_encoder = None
            return
        
        try:
            # 1. Question Answering Model (for specific AQI questions)
            logger.info("Loading Q&A model (distilbert-qa)...")
            self.qa_pipeline = pipeline(
                "question-answering",
                model="distilbert-base-cased-distilled-squad",
                device=-1  # CPU
            )
            logger.info("✓ Q&A model loaded")
            
            # 2. Intent Classification (zero-shot for flexibility)
            logger.info("Loading intent classifier (facebook/bart-large-mnli)...")
            self.intent_classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=-1
            )
            logger.info("✓ Intent classifier loaded")
            
            # 3. Conversational Model (for natural dialogue)
            logger.info("Loading conversational model (microsoft/DialoGPT-medium)...")
            self.conversational_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
            self.conversational_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
            self.conversational_model.to(self.device)
            logger.info("✓ Conversational model loaded")
            
            # 4. Sentence Embeddings (for semantic similarity)
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                logger.info("Loading sentence encoder (all-MiniLM-L6-v2)...")
                self.sentence_encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
                logger.info("✓ Sentence encoder loaded")
            else:
                self.sentence_encoder = None
            
            logger.info("🚀 All conversational AI models ready!")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            self.qa_pipeline = None
            self.intent_classifier = None
            self.conversational_model = None
            self.sentence_encoder = None
    
    def _build_aqi_knowledge_base(self) -> Dict:
        """Build comprehensive AQI knowledge base for contextual responses"""
        return {
            "pm25": {
                "full_name": "Particulate Matter 2.5",
                "description": "PM2.5 refers to tiny particles or droplets in the air that are 2.5 micrometers or less in width. These are so small they can get deep into your lungs and even enter your bloodstream.",
                "health_effects": "Can cause respiratory problems, heart disease, decreased lung function, and asthma attacks. Long-term exposure is linked to premature death in people with heart or lung disease.",
                "sources": "Vehicle emissions, power plants, wood burning, industrial processes, and wildfires",
                "safe_level": "Under 12 µg/m³ is considered safe for 24-hour exposure"
            },
            "pm10": {
                "full_name": "Particulate Matter 10",
                "description": "PM10 are inhalable particles with diameters of 10 micrometers or less. They're about 1/7th the width of a human hair.",
                "health_effects": "Can irritate eyes, nose, and throat. May cause coughing and difficulty breathing. Affects people with heart and lung diseases.",
                "sources": "Dust from roads, construction sites, agriculture, and crushing/grinding operations",
                "safe_level": "Under 54 µg/m³ for 24-hour exposure"
            },
            "o3": {
                "full_name": "Ozone",
                "description": "Ground-level ozone is a harmful air pollutant created by chemical reactions between oxides of nitrogen (NOx) and volatile organic compounds (VOCs) in sunlight.",
                "health_effects": "Irritates the respiratory system, reduces lung function, inflames airways, worsens asthma and other lung diseases.",
                "sources": "Not directly emitted - formed from reactions of pollutants from cars, power plants, refineries, and chemical plants in presence of sunlight",
                "safe_level": "Under 70 ppb (parts per billion) for 8-hour exposure"
            },
            "no2": {
                "full_name": "Nitrogen Dioxide",
                "description": "NO2 is a reddish-brown gas with a sharp, harsh odor. It's one of a group of highly reactive gases known as nitrogen oxides.",
                "health_effects": "Irritates airways, can trigger asthma attacks, increases susceptibility to respiratory infections.",
                "sources": "Emissions from cars, trucks, buses, power plants, and off-road equipment",
                "safe_level": "Under 100 µg/m³ for 1-hour exposure"
            },
            "so2": {
                "full_name": "Sulfur Dioxide",
                "description": "SO2 is a colorless gas with a pungent, irritating smell. It's released from volcanic eruptions and industrial processes.",
                "health_effects": "Can cause breathing problems, especially for people with asthma. Can irritate eyes and respiratory tract.",
                "sources": "Fossil fuel combustion at power plants, metal smelting, and other industrial processes",
                "safe_level": "Under 75 ppb for 1-hour exposure"
            },
            "co": {
                "full_name": "Carbon Monoxide",
                "description": "CO is a colorless, odorless gas formed when carbon in fuel doesn't completely burn. It's extremely dangerous in enclosed spaces.",
                "health_effects": "Reduces oxygen delivery to organs and tissues. Can cause dizziness, confusion, unconsciousness, and death at high levels.",
                "sources": "Motor vehicle exhaust, indoor fuel-burning appliances, and industrial processes",
                "safe_level": "Under 9 ppm for 8-hour exposure"
            },
            "safety_guidelines": {
                "children": "Children are more vulnerable because they breathe faster, are more active outdoors, and their lungs are still developing. Keep them indoors when AQI > 100.",
                "elderly": "Older adults often have underlying heart or lung conditions making them more susceptible. Should limit outdoor exposure when AQI > 100.",
                "exercise": "Exercise increases breathing rate, pulling more pollution deeper into lungs. Avoid outdoor exercise when AQI > 150.",
                "masks": "N95 or N99 masks filter out 95-99% of airborne particles. Surgical masks are less effective. Must fit properly to work.",
                "indoor_safety": "Keep windows closed, use HEPA air purifiers, avoid smoking indoors, reduce indoor cooking emissions.",
            },
            "aqi_categories": {
                "0-50": "Good - Air quality is satisfactory, and air pollution poses little or no risk.",
                "51-100": "Moderate - Air quality is acceptable. However, there may be a risk for some people who are unusually sensitive to air pollution.",
                "101-150": "Unhealthy for Sensitive Groups - Members of sensitive groups may experience health effects. The general public is less likely to be affected.",
                "151-200": "Unhealthy - Some members of the general public may experience health effects; members of sensitive groups may experience more serious health effects.",
                "201-300": "Very Unhealthy - Health alert: The risk of health effects is increased for everyone.",
                "301+": "Hazardous - Health warning of emergency conditions: everyone is more likely to be affected."
            }
        }
    
    def chat(self, user_message: str, aqi_context: Dict = None) -> Dict:
        """
        Main chat interface - processes user input and generates intelligent response
        
        Args:
            user_message: The user's input message
            aqi_context: Current AQI data {aqi, pollutants, location, etc.}
            
        Returns:
            Response dictionary with answer, confidence, intent, and suggestions
        """
        if not user_message or not user_message.strip():
            return self._empty_response()
        
        # Add to conversation history
        self.conversation_history.append({
            'user': user_message,
            'timestamp': datetime.now().isoformat()
        })
        
        # Initialize context if not provided
        if aqi_context is None:
            aqi_context = {'aqi': 100, 'pollutants': {}, 'location': 'your area'}
        
        # Update user context
        self._update_user_context(user_message, aqi_context)
        
        # Analyze sentiment
        sentiment = self._analyze_sentiment(user_message)
        
        # Classify intent using AI
        intent_result = self._classify_intent_ai(user_message)
        intent = intent_result['intent']
        
        logger.info(f"User intent: {intent} (confidence: {intent_result['confidence']:.2f})")
        
        # Route to appropriate handler based on intent
        response = None
        
        if intent in ['greeting', 'acknowledgment']:
            response = self._handle_social_interaction(user_message, intent, aqi_context)
        elif intent in ['safety', 'health_concern']:
            response = self._handle_safety_query(user_message, aqi_context, sentiment)
        elif intent == 'pollutant_info':
            response = self._handle_pollutant_question(user_message, aqi_context)
        elif intent == 'activity_recommendation':
            response = self._handle_activity_query(user_message, aqi_context)
        elif intent == 'timing':
            response = self._handle_timing_query(user_message, aqi_context)
        elif intent == 'protection':
            response = self._handle_protection_query(user_message, aqi_context)
        elif intent == 'help':
            response = self._handle_help_request(aqi_context)
        else:
            # Use general conversational AI
            response = self._handle_general_conversation(user_message, aqi_context)
        
        # Add to history
        if response:
            self.conversation_history[-1]['assistant'] = response.get('answer', '')
            self.conversation_history[-1]['intent'] = intent
        
        return response
    
    def _classify_intent_ai(self, message: str) -> Dict:
        """Classify user intent using transformer model"""
        if not self.intent_classifier:
            return self._classify_intent_fallback(message)
        
        try:
            # Define all possible intents
            candidate_labels = [
                "greeting",
                "acknowledgment", 
                "safety",
                "health concern",
                "pollutant info",
                "activity recommendation",
                "timing",
                "protection",
                "help",
                "general conversation"
            ]
            
            result = self.intent_classifier(message, candidate_labels, multi_label=False)
            
            # Map to our intent categories
            intent_mapping = {
                "greeting": "greeting",
                "acknowledgment": "acknowledgment",
                "safety": "safety",
                "health concern": "health_concern",
                "pollutant info": "pollutant_info",
                "activity recommendation": "activity_recommendation",
                "timing": "timing",
                "protection": "protection",
                "help": "help",
                "general conversation": "general"
            }
            
            top_label = result['labels'][0]
            return {
                'intent': intent_mapping.get(top_label, 'general'),
                'confidence': result['scores'][0],
                'all_intents': dict(zip(result['labels'], result['scores']))
            }
            
        except Exception as e:
            logger.warning(f"Intent classification error: {e}")
            return self._classify_intent_fallback(message)
    
    def _classify_intent_fallback(self, message: str) -> Dict:
        """Fallback intent classification using keywords"""
        message_lower = message.lower()
        
        patterns = {
            'greeting': ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening'],
            'acknowledgment': ['thanks', 'thank you', 'appreciate', 'helpful', 'great'],
            'safety': ['safe', 'dangerous', 'hazardous', 'risk', 'can i', 'should i', 'okay to'],
            'health_concern': ['health', 'sick', 'symptom', 'breathe', 'cough', 'affect', 'impact'],
            'pollutant_info': ['what is', 'pm2.5', 'pm10', 'ozone', 'no2', 'so2', 'co', 'explain', 'mean'],
            'activity_recommendation': ['activity', 'do', 'recommend', 'suggest', 'exercise', 'run', 'jog', 'play'],
            'timing': ['when', 'what time', 'best time', 'morning', 'evening', 'afternoon'],
            'protection': ['mask', 'protect', 'prevent', 'filter', 'purifier', 'defense'],
            'help': ['help', 'how', 'what can', 'assist', 'support']
        }
        
        best_intent = 'general'
        best_score = 0
        
        for intent, keywords in patterns.items():
            score = sum(1 for kw in keywords if kw in message_lower)
            if score > best_score:
                best_score = score
                best_intent = intent
        
        confidence = min(0.5 + best_score * 0.15, 0.95) if best_score > 0 else 0.3
        
        return {
            'intent': best_intent,
            'confidence': confidence,
            'method': 'keyword'
        }
    
    def _analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of user message"""
        if self.sentiment_analyzer:
            scores = self.sentiment_analyzer.polarity_scores(text)
            return {
                'compound': scores['compound'],
                'label': 'positive' if scores['compound'] > 0.05 else 'negative' if scores['compound'] < -0.05 else 'neutral'
            }
        return {'compound': 0, 'label': 'neutral'}
    
    def _update_user_context(self, message: str, aqi_context: Dict):
        """Track user context across conversation"""
        message_lower = message.lower()
        
        # Detect user profile
        if any(word in message_lower for word in ['child', 'kid', 'baby', 'toddler']):
            self.user_context['has_children'] = True
        if any(word in message_lower for word in ['elderly', 'senior', 'old', 'parent']):
            self.user_context['has_elderly'] = True
        if any(word in message_lower for word in ['asthma', 'copd', 'respiratory', 'lung']):
            self.user_context['respiratory_condition'] = True
        if any(word in message_lower for word in ['exercise', 'workout', 'gym', 'run', 'jog']):
            self.user_context['active_lifestyle'] = True
        
        # Store current AQI
        self.user_context['last_aqi'] = aqi_context.get('aqi')
        self.user_context['location'] = aqi_context.get('location')
    
    def _handle_social_interaction(self, message: str, intent: str, aqi_context: Dict) -> Dict:
        """Handle greetings and acknowledgments"""
        aqi = aqi_context.get('aqi', 100)
        category = self._get_aqi_category(aqi)
        
        if intent == 'greeting':
            greetings = [
                f"👋 Hello! I'm your AI air quality assistant. The current AQI is {int(aqi)} ({category}). How can I help you stay safe today?",
                f"Hi there! 🌍 Air quality is {category.lower()} right now (AQI: {int(aqi)}). What would you like to know?",
                f"Hello! 😊 Today's air quality is {category} with an AQI of {int(aqi)}. I'm here to answer any questions you have!"
            ]
            answer = random.choice(greetings)
        else:  # acknowledgment
            acknowledgments = [
                "You're very welcome! 😊 Feel free to ask me anything else about air quality or health safety.",
                "Happy to help! 🌟 Let me know if you have any other questions.",
                "Glad I could assist! 💚 Stay safe and breathe easy!"
            ]
            answer = random.choice(acknowledgments)
        
        return {
            'answer': answer,
            'confidence': 1.0,
            'intent': intent,
            'suggestions': [
                "Is it safe to exercise outside?",
                "What are the current pollutant levels?",
                "When is the best time to go out?"
            ]
        }
    
    def _handle_safety_query(self, message: str, aqi_context: Dict, sentiment: Dict) -> Dict:
        """Handle safety-related questions with empathy"""
        aqi = aqi_context.get('aqi', 100)
        message_lower = message.lower()
        
        # Detect specific safety concerns
        if any(word in message_lower for word in ['child', 'kid', 'baby']):
            return self._safety_response_children(aqi, sentiment)
        elif any(word in message_lower for word in ['exercise', 'run', 'jog', 'workout']):
            return self._safety_response_exercise(aqi, sentiment)
        elif any(word in message_lower for word in ['outdoor', 'outside']):
            return self._safety_response_outdoor(aqi, sentiment)
        else:
            return self._safety_response_general(aqi, sentiment)
    
    def _safety_response_children(self, aqi: float, sentiment: Dict) -> Dict:
        """Safety response focused on children"""
        # Add empathy if user seems worried
        empathy = ""
        if sentiment['label'] == 'negative':
            empathy = "I understand your concern for your children's safety. "
        
        if aqi <= 50:
            answer = f"{empathy}✅ Great news! The air quality is excellent (AQI: {int(aqi)}). It's completely safe for children to play outside. They can enjoy all outdoor activities without any restrictions!"
        elif aqi <= 100:
            answer = f"{empathy}🟡 The air quality is moderate (AQI: {int(aqi)}). It's generally safe for children to play outside, but if your child has asthma or respiratory issues, watch for any symptoms and consider limiting very strenuous activities."
        elif aqi <= 150:
            answer = f"{empathy}🟠 Air quality is unhealthy for sensitive groups (AQI: {int(aqi)}). I'd recommend limiting prolonged outdoor play for children, especially those with asthma or respiratory conditions. Short outdoor activities are okay, but keep them light and take frequent breaks indoors."
        elif aqi <= 200:
            answer = f"{empathy}🔴 The air is unhealthy right now (AQI: {int(aqi)}). It's best to keep children indoors today. If they must go outside, limit it to short periods and avoid any strenuous activities. Consider indoor activities like arts, crafts, or indoor games instead."
        else:
            answer = f"{empathy}⚫ Air quality is hazardous (AQI: {int(aqi)}). Children should stay indoors with windows closed. Use air purifiers if available. This is not safe for any outdoor activities. Plan fun indoor activities to keep them engaged and safe."
        
        return {
            'answer': answer,
            'confidence': 0.95,
            'intent': 'safety',
            'context': 'children',
            'recommendations': self._get_child_activity_suggestions(aqi)
        }
    
    def _safety_response_exercise(self, aqi: float, sentiment: Dict) -> Dict:
        """Safety response for exercise activities"""
        empathy = ""
        if sentiment['label'] == 'negative':
            empathy = "I know staying active is important to you. "
        
        if aqi <= 50:
            answer = f"{empathy}✅ Perfect conditions for outdoor exercise! AQI is {int(aqi)}, which is excellent. Feel free to run, jog, or do any high-intensity workout outside. Your lungs will thank you for the fresh air! 🏃"
        elif aqi <= 100:
            answer = f"{empathy}🟡 AQI is {int(aqi)} - acceptable for most outdoor exercise. If you're generally healthy, you can proceed with your workout. Just pay attention to how you feel. If you have asthma or respiratory issues, consider reducing intensity or working out indoors."
        elif aqi <= 150:
            answer = f"{empathy}🟠 AQI is {int(aqi)} - I'd suggest moving your workout indoors today. If you must exercise outside, keep it light (like walking) and short (under 30 minutes). Heavy breathing during intense exercise pulls more pollutants into your lungs."
        elif aqi <= 200:
            answer = f"{empathy}🔴 Air quality is unhealthy (AQI: {int(aqi)}). Outdoor exercise is not recommended today. Consider indoor alternatives: home workouts, gym, yoga, or strength training. Your body needs exercise, but your lungs need protection!"
        else:
            answer = f"{empathy}⚫ Hazardous air quality (AQI: {int(aqi)}). Definitely skip outdoor exercise today! Stick to indoor workouts. Try online fitness classes, indoor cycling, bodyweight exercises, or yoga. Stay active, stay safe, stay indoors! 🏠💪"
        
        return {
            'answer': answer,
            'confidence': 0.95,
            'intent': 'safety',
            'context': 'exercise',
            'alternative_activities': self._get_indoor_exercise_alternatives(aqi)
        }
    
    def _safety_response_outdoor(self, aqi: float, sentiment: Dict) -> Dict:
        """General outdoor safety response"""
        category = self._get_aqi_category(aqi)
        
        if aqi <= 50:
            answer = f"✅ Yes, it's safe to be outside! Air quality is {category} (AQI: {int(aqi)}). Enjoy outdoor activities freely - this is perfect weather for fresh air!"
        elif aqi <= 100:
            answer = f"🟡 Generally safe for most people (AQI: {int(aqi)} - {category}). You can go outside, but if you're in a sensitive group (children, elderly, respiratory conditions), just be mindful of how long you stay out."
        elif aqi <= 150:
            answer = f"🟠 It's {category} (AQI: {int(aqi)}). You can go outside for short periods, but I'd recommend limiting your time outdoors, especially if you're sensitive to air pollution. Keep activities light and take breaks."
        elif aqi <= 200:
            answer = f"🔴 Air quality is {category} (AQI: {int(aqi)}). It's better to stay indoors if possible. If you must go out, keep it brief, avoid physical exertion, and consider wearing an N95 mask."
        else:
            answer = f"⚫ {category} conditions (AQI: {int(aqi)})! Everyone should avoid going outside. Stay indoors with windows closed. Use air purifiers if available. Only go out if absolutely necessary, and wear a proper N95 or N99 mask."
        
        return {
            'answer': answer,
            'confidence': 0.92,
            'intent': 'safety',
            'context': 'outdoor',
            'related_topics': ['protection', 'timing', 'health']
        }
    
    def _safety_response_general(self, aqi: float, sentiment: Dict) -> Dict:
        """General safety response"""
        category = self._get_aqi_category(aqi)
        
        answer = f"Current air quality is {category} with an AQI of {int(aqi)}. "
        
        if aqi <= 50:
            answer += "This is excellent! Air pollution poses little to no risk. All activities are safe for everyone."
        elif aqi <= 100:
            answer += "This is acceptable for most people. Unusually sensitive individuals should consider limiting prolonged outdoor exertion."
        elif aqi <= 150:
            answer += "Sensitive groups (children, elderly, people with respiratory conditions) should limit prolonged outdoor activities. General public is less likely to be affected."
        elif aqi <= 200:
            answer += "Everyone may begin to experience health effects. Sensitive groups should avoid outdoor activities. Others should limit outdoor exertion."
        else:
            answer += "This is a health alert! Everyone should avoid all outdoor activities. Health effects will be more serious for sensitive groups."
        
        return {
            'answer': answer,
            'confidence': 0.90,
            'intent': 'safety',
            'suggestions': [
                "Ask about specific activities",
                "Learn about protective measures",
                "Check when air will be better"
            ]
        }
    
    def _handle_pollutant_question(self, message: str, aqi_context: Dict) -> Dict:
        """Handle questions about specific pollutants with detailed information"""
        message_lower = message.lower()
        pollutants = aqi_context.get('pollutants', {})
        
        # Identify which pollutant user is asking about
        pollutant_mentioned = None
        for pollutant_key in ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']:
            if pollutant_key in message_lower or pollutant_key.replace('o3', 'ozone') in message_lower:
                pollutant_mentioned = pollutant_key
                break
        
        # Handle "PM2.5" specifically as mentioned by user
        if 'pm2.5' in message_lower or 'pm 2.5' in message_lower:
            pollutant_mentioned = 'pm25'
        
        if pollutant_mentioned and pollutant_mentioned in self.aqi_knowledge:
            info = self.aqi_knowledge[pollutant_mentioned]
            current_level = pollutants.get(pollutant_mentioned, 'unknown')
            
            answer = f"📖 **{info['full_name']} ({pollutant_mentioned.upper()})**\n\n"
            answer += f"**What it is:** {info['description']}\n\n"
            answer += f"**Health Effects:** {info['health_effects']}\n\n"
            answer += f"**Sources:** {info['sources']}\n\n"
            answer += f"**Safe Level:** {info['safe_level']}\n\n"
            
            if current_level != 'unknown':
                answer += f"**Current Level:** {current_level:.1f}\n\n"
                # Add assessment
                if pollutant_mentioned == 'pm25':
                    if current_level < 12:
                        answer += "✅ Current level is in the safe range."
                    elif current_level < 35:
                        answer += "🟡 Current level is moderate - acceptable but not ideal."
                    else:
                        answer += "🔴 Current level is elevated - take precautions!"
            
            return {
                'answer': answer,
                'confidence': 0.98,
                'intent': 'pollutant_info',
                'pollutant': pollutant_mentioned,
                'related_topics': ['health', 'protection', 'sources']
            }
        else:
            # General explanation about pollutants
            answer = "🔬 **Air Quality Pollutants**\n\n"
            answer += "The main pollutants we monitor are:\n\n"
            answer += "• **PM2.5** - Tiny particles that penetrate deep into lungs\n"
            answer += "• **PM10** - Larger inhalable particles\n"
            answer += "• **Ozone (O3)** - Irritates respiratory system\n"
            answer += "• **NO2** - Nitrogen dioxide from vehicle emissions\n"
            answer += "• **SO2** - Sulfur dioxide from industrial processes\n"
            answer += "• **CO** - Carbon monoxide, reduces oxygen delivery\n\n"
            answer += "Which pollutant would you like to learn more about?"
            
            return {
                'answer': answer,
                'confidence': 0.85,
                'intent': 'pollutant_info',
                'suggestions': [
                    "Tell me about PM2.5",
                    "What is ozone?",
                    "Explain nitrogen dioxide"
                ]
            }
    
    def _handle_activity_query(self, message: str, aqi_context: Dict) -> Dict:
        """Handle activity recommendation queries"""
        aqi = aqi_context.get('aqi', 100)
        
        answer = f"🎯 **Activity Recommendations for AQI {int(aqi)}**\n\n"
        
        if aqi <= 50:
            answer += "✅ **Perfect conditions!** You can enjoy:\n"
            answer += "• Outdoor sports (running, cycling, soccer)\n"
            answer += "• High-intensity workouts\n"
            answer += "• Children's outdoor play\n"
            answer += "• Gardening or yard work\n"
            answer += "• Picnics and outdoor gatherings\n\n"
            answer += "Take advantage of this great air quality! 🌟"
        elif aqi <= 100:
            answer += "🟡 **Good for most activities:**\n"
            answer += "• Light to moderate exercise\n"
            answer += "• Walking or casual biking\n"
            answer += "• Children can play outside\n"
            answer += "• Gardening is fine\n\n"
            answer += "⚠️ If you're unusually sensitive, watch for symptoms."
        elif aqi <= 150:
            answer += "🟠 **Recommended activities:**\n"
            answer += "• Short outdoor walks (under 30 min)\n"
            answer += "• Indoor exercise (gym, home workouts)\n"
            answer += "• Indoor sports facilities\n"
            answer += "• Arts and crafts\n"
            answer += "• Reading, board games\n\n"
            answer += "❌ **Avoid:** Prolonged outdoor activities, intense exercise"
        elif aqi <= 200:
            answer += "🔴 **Best to stay indoors:**\n"
            answer += "• Home workouts or yoga\n"
            answer += "• Indoor gym (if well-ventilated)\n"
            answer += "• Swimming (indoor pool)\n"
            answer += "• Indoor entertainment\n"
            answer += "• Online classes or hobbies\n\n"
            answer += "❌ **Avoid:** All outdoor activities"
        else:
            answer += "⚫ **Stay indoors only:**\n"
            answer += "• Indoor exercise (bodyweight, yoga)\n"
            answer += "• Home entertainment\n"
            answer += "• Reading, cooking, crafts\n"
            answer += "• Virtual socializing\n"
            answer += "• Online workouts\n\n"
            answer += "🚨 **Critical:** Do not go outside unless absolutely necessary!"
        
        return {
            'answer': answer,
            'confidence': 0.93,
            'intent': 'activity_recommendation',
            'aqi_level': aqi
        }
    
    def _handle_timing_query(self, message: str, aqi_context: Dict) -> Dict:
        """Handle questions about timing"""
        aqi = aqi_context.get('aqi', 100)
        
        answer = "⏰ **Best Times for Outdoor Activities**\n\n"
        answer += "Based on typical air quality patterns:\n\n"
        answer += "**🌅 Early Morning (5-7 AM):**\n"
        answer += "Often the cleanest air of the day. Best for exercise before traffic builds up.\n\n"
        answer += "**🌇 Evening (7-9 PM):**\n"
        answer += "Air quality usually improves after sunset as traffic decreases.\n\n"
        answer += "**❌ Avoid:**\n"
        answer += "• Rush hours (7-9 AM, 5-7 PM) - peak traffic pollution\n"
        answer += "• Hot afternoons - higher ozone levels\n\n"
        
        if aqi > 150:
            answer += f"⚠️ **Today's Note:** With current AQI of {int(aqi)}, even the best times may not be safe. Consider indoor activities."
        else:
            answer += f"✅ Current AQI: {int(aqi)} - Choose early morning or evening for outdoor time!"
        
        return {
            'answer': answer,
            'confidence': 0.88,
            'intent': 'timing',
            'related_topics': ['activity', 'safety']
        }
    
    def _handle_protection_query(self, message: str, aqi_context: Dict) -> Dict:
        """Handle questions about protection and masks"""
        aqi = aqi_context.get('aqi', 100)
        
        answer = "😷 **Air Pollution Protection Guide**\n\n"
        
        # Mask recommendations
        answer += "**Mask Recommendations:**\n"
        if aqi <= 100:
            answer += "• Mask not necessary at current levels (AQI: {int(aqi)})\n"
            answer += "• Use only if you're particularly sensitive\n"
        elif aqi <= 150:
            answer += f"• Consider wearing a mask for prolonged outdoor exposure (AQI: {int(aqi)})\n"
            answer += "• KN95 or N95 masks recommended\n"
        else:
            answer += f"• **Wear a mask if you must go outside** (AQI: {int(aqi)})\n"
            answer += "• **N95 or N99 masks** - filter 95-99% of particles\n"
            answer += "• Ensure proper fit - should seal against your face\n"
            answer += "• ❌ Surgical masks are less effective\n"
        
        answer += "\n**Other Protection Measures:**\n"
        answer += "• Keep windows and doors closed\n"
        answer += "• Use HEPA air purifiers indoors\n"
        answer += "• Avoid outdoor exercise\n"
        answer += "• Stay hydrated\n"
        answer += "• Monitor symptoms (coughing, shortness of breath)\n"
        
        return {
            'answer': answer,
            'confidence': 0.95,
            'intent': 'protection',
            'aqi_level': aqi
        }
    
    def _handle_help_request(self, aqi_context: Dict) -> Dict:
        """Handle help requests"""
        answer = "🤝 **How I Can Help You**\n\n"
        answer += "I'm an AI-powered air quality assistant. You can ask me:\n\n"
        answer += "1. **Safety Questions** - \"Is it safe for kids to play outside?\"\n"
        answer += "2. **Activity Advice** - \"What activities can I do today?\"\n"
        answer += "3. **Pollutant Information** - \"What is PM2.5?\"\n"
        answer += "4. **Health Impact** - \"How does air pollution affect health?\"\n"
        answer += "5. **Protection Tips** - \"What mask should I wear?\"\n"
        answer += "6. **Timing** - \"When is the best time to exercise?\"\n"
        answer += "7. **General Questions** - Ask me anything naturally!\n\n"
        answer += "💬 **I understand natural language** - just ask in your own words!"
        
        return {
            'answer': answer,
            'confidence': 1.0,
            'intent': 'help'
        }
    
    def _handle_general_conversation(self, message: str, aqi_context: Dict) -> Dict:
        """Handle general conversation using DialoGPT"""
        if not self.conversational_model or not self.conversational_tokenizer:
            return self._fallback_response(message, aqi_context)
        
        try:
            # Prepare input with AQI context
            aqi = aqi_context.get('aqi', 100)
            category = self._get_aqi_category(aqi)
            
            # Encode the conversation
            context = f"Current air quality is {category} (AQI: {int(aqi)}). User asks: {message}"
            inputs = self.conversational_tokenizer.encode(context + self.conversational_tokenizer.eos_token, return_tensors='pt')
            inputs = inputs.to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.conversational_model.generate(
                    inputs,
                    max_length=200,
                    num_return_sequences=1,
                    pad_token_id=self.conversational_tokenizer.eos_token_id,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    temperature=0.7
                )
            
            # Decode response
            generated_text = self.conversational_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the response part
            response_text = generated_text.replace(context, '').strip()
            
            # If response is too short or doesn't make sense, use fallback
            if len(response_text) < 10:
                return self._fallback_response(message, aqi_context)
            
            # Add AQI context to response
            final_answer = f"{response_text}\n\n📊 Current AQI: {int(aqi)} ({category})"
            
            return {
                'answer': final_answer,
                'confidence': 0.75,
                'intent': 'general',
                'method': 'DialoGPT'
            }
            
        except Exception as e:
            logger.warning(f"Conversational AI error: {e}")
            return self._fallback_response(message, aqi_context)
    
    def _fallback_response(self, message: str, aqi_context: Dict) -> Dict:
        """Fallback response when AI models fail"""
        aqi = aqi_context.get('aqi', 100)
        category = self._get_aqi_category(aqi)
        
        answer = f"I understand you're asking about air quality. Currently, the AQI is {int(aqi)} ({category}). "
        
        if aqi <= 50:
            answer += "This is excellent air quality - safe for all activities!"
        elif aqi <= 100:
            answer += "This is moderate - generally safe for most people."
        elif aqi <= 150:
            answer += "This is unhealthy for sensitive groups - be cautious."
        else:
            answer += "This requires caution - limit outdoor activities."
        
        answer += "\n\nCould you rephrase your question? Or try asking about:\n"
        answer += "• Safety for specific activities\n"
        answer += "• Health impacts\n"
        answer += "• Protective measures"
        
        return {
            'answer': answer,
            'confidence': 0.60,
            'intent': 'general',
            'method': 'fallback'
        }
    
    def _empty_response(self) -> Dict:
        """Response for empty input"""
        return {
            'answer': "I didn't catch that. Could you ask me about air quality, health impacts, safety, or activities?",
            'confidence': 1.0,
            'intent': 'empty'
        }
    
    def _get_aqi_category(self, aqi: float) -> str:
        """Get AQI category name"""
        if aqi <= 50:
            return "Good"
        elif aqi <= 100:
            return "Moderate"
        elif aqi <= 150:
            return "Unhealthy for Sensitive Groups"
        elif aqi <= 200:
            return "Unhealthy"
        elif aqi <= 300:
            return "Very Unhealthy"
        else:
            return "Hazardous"
    
    def _get_child_activity_suggestions(self, aqi: float) -> List[str]:
        """Get activity suggestions for children based on AQI"""
        if aqi <= 50:
            return ["Outdoor play", "Sports", "Playground", "Biking", "Running"]
        elif aqi <= 100:
            return ["Light outdoor play", "Walking", "Casual biking", "Indoor backup ready"]
        elif aqi <= 150:
            return ["Short outdoor play only", "Indoor games", "Arts & crafts", "Reading"]
        else:
            return ["Indoor games", "Arts & crafts", "Movies", "Board games", "Indoor play areas"]
    
    def _get_indoor_exercise_alternatives(self, aqi: float) -> List[str]:
        """Get indoor exercise alternatives"""
        return [
            "Home workout videos",
            "Yoga or Pilates",
            "Indoor gym",
            "Bodyweight exercises",
            "Dance fitness",
            "Indoor cycling",
            "Strength training"
        ]
    
    def get_conversation_summary(self) -> Dict:
        """Get summary of current conversation"""
        return {
            'message_count': len(self.conversation_history),
            'user_context': self.user_context,
            'recent_intents': [msg.get('intent') for msg in list(self.conversation_history)[-5:] if 'intent' in msg]
        }
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history.clear()
        self.user_context.clear()
        logger.info("Conversation history cleared")


# Singleton instance
_conversational_ai_instance = None

def get_conversational_ai() -> ConversationalAIEngine:
    """Get or create conversational AI instance"""
    global _conversational_ai_instance
    if _conversational_ai_instance is None:
        _conversational_ai_instance = ConversationalAIEngine(use_gpu=False)
    return _conversational_ai_instance
