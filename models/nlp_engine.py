"""
Lightweight NLP Engine for AQI Health System
Uses VADER + TextBlob as base, with optional Hugging Face enhancement
Lightweight HF models: ~150MB total (vs 2.5GB for full models)
Enhanced with Kaggle datasets for better recommendations
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime
import random
import sys
import os

# Add utils path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import Kaggle dataset manager
try:
    from utils.kaggle_dataset import KaggleAQIDataset
    KAGGLE_AVAILABLE = True
except ImportError:
    KAGGLE_AVAILABLE = False
    logging.warning("Kaggle dataset manager not available")

# Lightweight NLP libraries (always available)
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    logging.warning("VADER not available. Install with: pip install vaderSentiment")

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    logging.warning("TextBlob not available. Install with: pip install textblob")

# Optional: Hugging Face Transformers (lightweight models)
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
    # Use CPU to save memory
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    DEVICE = "cpu"
    logging.info("Transformers not available. Using VADER/TextBlob only.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _ensure_nlp_resources():
    """Best-effort download of lightweight NLTK/TextBlob corpora.

    This prevents runtime crashes like:
    """
    try:
        import nltk

        resources = {
            'tokenizers/punkt': 'punkt',
            'taggers/averaged_perceptron_tagger': 'averaged_perceptron_tagger',
            'corpora/wordnet': 'wordnet',
            'corpora/omw-1.4': 'omw-1.4',
        }

        for path, pkg in resources.items():
            try:
                nltk.data.find(path)
            except LookupError:
                nltk.download(pkg, quiet=True)
    except Exception as exc:  # pragma: no cover - best effort only
        logger.warning(f"NLTK/TextBlob resource setup skipped: {exc}")

# Lightweight model configurations
LIGHTWEIGHT_MODELS = {
    # ~22MB - Excellent for semantic similarity and understanding
    'embedding': 'sentence-transformers/all-MiniLM-L6-v2',
    # ~17MB - Tiny but effective for classification
    'classifier': 'prajjwal1/bert-tiny',  
    # ~265MB - Best lightweight sentiment (optional, falls back to VADER)
    'sentiment': 'distilbert-base-uncased-finetuned-sst-2-english',
    # ~82MB - Small text generation
    'generator': 'distilgpt2',
    # ~65MB - Fast zero-shot classification
    'zero_shot': 'typeform/distilbert-base-uncased-mnli',
}


class HuggingFaceNLPEngine:
    """
    Hybrid NLP engine: VADER/TextBlob base + optional Hugging Face enhancement
    Provides sentiment analysis, intent classification, and smart suggestions
    """
    
    def __init__(self, model_name: str = None, use_transformers: bool = True, use_kaggle: bool = True):
        """
        Initialize NLP engine with lightweight models
        
        Args:
            model_name: Optional custom model name
            use_transformers: Whether to load HuggingFace models (default: True)
            use_kaggle: Whether to use Kaggle datasets for recommendations (default: True)
        """
        self.model_name = model_name
        self.use_transformers = use_transformers and TRANSFORMERS_AVAILABLE
        self.use_kaggle = use_kaggle and KAGGLE_AVAILABLE
        
        # Initialize Kaggle dataset manager
        self.kaggle_manager = None
        self.kaggle_recommendations = None
        if self.use_kaggle:
            try:
                self.kaggle_manager = KaggleAQIDataset()
                # Try to load existing recommendations or create sample data
                import pickle
                from pathlib import Path
                cache_file = Path("data/kaggle/recommendations_db.pkl")
                if cache_file.exists():
                    with open(cache_file, 'rb') as f:
                        self.kaggle_recommendations = pickle.load(f)
                    logger.info("✓ Loaded Kaggle recommendations database")
                else:
                    logger.info("Building Kaggle recommendations database...")
                    self.kaggle_recommendations = self.kaggle_manager.build_recommendations_database()
                    logger.info("✓ Kaggle recommendations database ready")
            except Exception as e:
                logger.warning(f"Kaggle integration failed: {e}. Using basic recommendations.")
                self.use_kaggle = False
        
        # Always initialize VADER (lightweight, fast)
        if VADER_AVAILABLE:
            self.vader_analyzer = SentimentIntensityAnalyzer()
            logger.info("✓ VADER sentiment analyzer loaded")
        else:
            self.vader_analyzer = None
        
        # Optional: Initialize HuggingFace pipelines
        self.hf_sentiment = None
        self.hf_classifier = None
        self.hf_generator = None
        
        if self.use_transformers:
            self._initialize_hf_models()
        else:
            logger.info("Running in lightweight mode (VADER + TextBlob only)")
            
    def _initialize_hf_models(self):
        """Initialize lightweight Hugging Face models"""
        try:
            # Zero-shot classifier for intent detection (~65MB)
            logger.info("Loading zero-shot classifier (distilbert-mnli)...")
            self.hf_classifier = pipeline(
                "zero-shot-classification",
                model=LIGHTWEIGHT_MODELS['zero_shot'],
                device=-1  # CPU
            )
            logger.info("✓ Zero-shot classifier loaded")
            
            # Text generator for dynamic responses (~82MB)
            logger.info("Loading text generator (distilgpt2)...")
            self.hf_generator = pipeline(
                "text-generation",
                model=LIGHTWEIGHT_MODELS['generator'],
                device=-1,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                pad_token_id=50256
            )
            logger.info("✓ Text generator loaded")
            
            logger.info("🚀 Hugging Face models ready!")
            
        except Exception as e:
            logger.warning(f"Could not load HF models: {e}. Using VADER/TextBlob fallback.")
            self.hf_classifier = None
            self.hf_generator = None
    
    def classify_intent(self, query: str) -> Dict:
        """
        Classify user intent using zero-shot classification
        
        Args:
            query: User's question or input
            
        Returns:
            Intent classification with confidence scores
        """
        candidate_labels = [
            "safety inquiry",
            "health concern", 
            "activity recommendation",
            "air quality information",
            "protection advice",
            "timing question",
            "general question"
        ]
        
        if self.hf_classifier:
            try:
                result = self.hf_classifier(query, candidate_labels, multi_label=False)
                return {
                    'intent': result['labels'][0].replace(' ', '_'),
                    'confidence': result['scores'][0],
                    'all_scores': dict(zip(result['labels'], result['scores'])),
                    'source': 'huggingface'
                }
            except Exception as e:
                logger.warning(f"HF classifier error: {e}")
        
        # Fallback to keyword-based classification
        return self._keyword_intent_classification(query)
    
    def _keyword_intent_classification(self, query: str) -> Dict:
        """Fallback keyword-based intent classification"""
        query_lower = query.lower()
        
        intents = {
            'safety_inquiry': ['safe', 'can i', 'should i', 'okay to', 'dangerous'],
            'health_concern': ['health', 'symptom', 'sick', 'affect', 'impact', 'breathe'],
            'activity_recommendation': ['exercise', 'run', 'jog', 'workout', 'outdoor', 'activity'],
            'air_quality_information': ['what is', 'aqi', 'explain', 'pollutant', 'pm2.5'],
            'protection_advice': ['mask', 'protect', 'prevent', 'purifier', 'filter'],
            'timing_question': ['when', 'best time', 'what time', 'morning', 'evening'],
        }
        
        best_intent = 'general_question'
        best_score = 0
        
        for intent, keywords in intents.items():
            score = sum(1 for kw in keywords if kw in query_lower)
            if score > best_score:
                best_score = score
                best_intent = intent
        
        return {
            'intent': best_intent,
            'confidence': min(0.5 + best_score * 0.15, 0.95),
            'source': 'keyword'
        }
    
    def generate_dynamic_response(self, prompt: str, aqi_context: Dict) -> str:
        """
        Generate dynamic text using DistilGPT2
        
        Args:
            prompt: Starting prompt
            aqi_context: Current AQI data
            
        Returns:
            Generated response text
        """
        if self.hf_generator:
            try:
                # Create context-aware prompt
                aqi = aqi_context.get('aqi', 100)
                status = "good" if aqi <= 50 else "moderate" if aqi <= 100 else "unhealthy" if aqi <= 150 else "hazardous"
                
                full_prompt = f"Air quality is {status} with AQI {int(aqi)}. {prompt}"
                
                result = self.hf_generator(
                    full_prompt,
                    max_new_tokens=60,
                    num_return_sequences=1,
                    truncation=True
                )
                
                generated = result[0]['generated_text']
                # Clean up the response
                response = generated.replace(full_prompt, '').strip()
                
                # If generation is too short or nonsensical, use template
                if len(response) < 20:
                    return self._get_template_response(aqi, prompt)
                    
                return response
                
            except Exception as e:
                logger.warning(f"Generation error: {e}")
        
        return self._get_template_response(aqi_context.get('aqi', 100), prompt)
    
    def _get_template_response(self, aqi: float, context: str) -> str:
        """Template-based response fallback"""
        if aqi <= 50:
            return "The air quality is excellent! Perfect conditions for all outdoor activities."
        elif aqi <= 100:
            return "Air quality is acceptable. Most people can enjoy outdoor activities normally."
        elif aqi <= 150:
            return "Air quality is moderate. Sensitive individuals should limit prolonged outdoor exposure."
        elif aqi <= 200:
            return "Air quality is unhealthy. Everyone should reduce outdoor activities."
        else:
            return "Air quality is hazardous! Stay indoors and use air purification."
    
    def analyze_with_ai(self, text: str, aqi_context: Dict = None) -> Dict:
        """
        Comprehensive AI analysis of text using all available models
        
        Args:
            text: Text to analyze
            aqi_context: Optional AQI context
            
        Returns:
            Complete analysis with sentiment, intent, and suggestions
        """
        result = {
            'text': text,
            'timestamp': datetime.now().isoformat(),
            'models_used': []
        }
        
        # 1. Sentiment Analysis (VADER - always fast)
        if self.vader_analyzer:
            vader_scores = self.vader_analyzer.polarity_scores(text)
            result['sentiment'] = {
                'compound': vader_scores['compound'],
                'positive': vader_scores['pos'],
                'negative': vader_scores['neg'],
                'neutral': vader_scores['neu'],
                'label': 'POSITIVE' if vader_scores['compound'] >= 0.05 else 'NEGATIVE' if vader_scores['compound'] <= -0.05 else 'NEUTRAL'
            }
            result['models_used'].append('VADER')
        
        # 2. Intent Classification (HuggingFace or fallback)
        intent_result = self.classify_intent(text)
        result['intent'] = intent_result
        result['models_used'].append(f"Intent: {intent_result['source']}")
        
        # 3. Entity/Keyword Extraction (TextBlob)
        if TEXTBLOB_AVAILABLE:
            blob = TextBlob(text)
            result['keywords'] = list(blob.noun_phrases)
            result['entities'] = [{'word': word, 'tag': tag} for word, tag in blob.tags if tag.startswith('NN')]
            result['models_used'].append('TextBlob')
        
        # 4. Generate AI-enhanced suggestion if context provided
        if aqi_context and self.hf_generator:
            result['ai_suggestion'] = self.generate_dynamic_response(
                f"Based on the question: {text[:100]}",
                aqi_context
            )
            result['models_used'].append('DistilGPT2')
        
        return result
            
    def get_kaggle_recommendations(self, aqi: float, pollutants: Dict = None) -> Dict:
        """
        Get recommendations from Kaggle datasets
        
        Args:
            aqi: Current AQI value
            pollutants: Dictionary of pollutant levels
            
        Returns:
            Recommendations dictionary
        """
        if self.use_kaggle and self.kaggle_manager:
            try:
                return self.kaggle_manager.get_recommendations_for_aqi(aqi, pollutants)
            except Exception as e:
                logger.warning(f"Kaggle recommendation error: {e}")
        
        # Fallback to basic recommendations
        return self._get_basic_recommendations(aqi)
    
    def _get_basic_recommendations(self, aqi: float) -> Dict:
        """Fallback basic recommendations when Kaggle not available"""
        if aqi <= 50:
            category = 'good'
            activities = ['all outdoor activities', 'exercise', 'sports']
            precautions = ['none needed']
        elif aqi <= 100:
            category = 'moderate'
            activities = ['most outdoor activities', 'light exercise']
            precautions = ['sensitive groups limit prolonged exertion']
        elif aqi <= 150:
            category = 'unhealthy_sensitive'
            activities = ['reduced outdoor activities', 'indoor exercise']
            precautions = ['everyone limit prolonged outdoor activities', 'sensitive groups stay indoors']
        elif aqi <= 200:
            category = 'unhealthy'
            activities = ['indoor activities only']
            precautions = ['everyone avoid outdoor activities', 'wear mask if must go out']
        else:
            category = 'hazardous'
            activities = ['stay indoors']
            precautions = ['remain indoors', 'use air purifier', 'seal windows']
        
        return {
            'aqi': aqi,
            'category': category,
            'activities': activities,
            'precautions': precautions
        }
    
    def generate_health_explanation(
        self,
        aqi: float,
        pollutants: Dict[str, float],
        risk_level: str,
        user_profile: str,
        location: str
    ) -> str:
        """
        Generate natural language explanation of health impacts with Kaggle data
        
        Args:
            aqi: Current AQI value
            pollutants: Dictionary of pollutant levels
            risk_level: Computed risk level
            user_profile: User category
            location: Location name
            
        Returns:
            Natural language explanation
        """
        # Create context for generation
        context = self._create_health_context(
            aqi, pollutants, risk_level, user_profile, location
        )
        
        return self._generate_explanation(context)
            
    def _create_health_context(
        self,
        aqi: float,
        pollutants: Dict,
        risk_level: str,
        user_profile: str,
        location: str
    ) -> Dict:
        """Create structured context for explanation"""
        dominant_pollutant = max(pollutants.items(), key=lambda x: x[1])
        
        return {
            'aqi': aqi,
            'location': location,
            'risk_level': risk_level,
            'user_profile': user_profile,
            'dominant_pollutant': dominant_pollutant[0].upper(),
            'pollutant_value': dominant_pollutant[1],
            'time': datetime.now().strftime('%I:%M %p')
        }
        
    def _generate_explanation(self, context: Dict) -> str:
        """Generate high-quality explanation using templates"""
        aqi = context['aqi']
        
        # Health impact templates based on AQI
        if aqi > 300:
            health_impact = (
                f"🚨 HEALTH EMERGENCY in {context['location']}! "
                f"Current AQI is {int(aqi)} - This is hazardous to all populations. "
                f"{context['dominant_pollutant']} levels are critically high at {int(context['pollutant_value'])} µg/m³. "
                f"Immediate Action Required: Stay indoors, use air purifiers, and avoid all physical exertion. "
            )
        elif aqi > 200:
            health_impact = (
                f"⚠️ VERY UNHEALTHY air quality in {context['location']}. "
                f"AQI: {int(aqi)}. {context['dominant_pollutant']} is the primary concern. "
                f"Everyone should avoid outdoor activities. Sensitive groups must remain indoors. "
            )
        elif aqi > 150:
            health_impact = (
                f"🔴 UNHEALTHY air in {context['location']} (AQI: {int(aqi)}). "
                f"Elevated {context['dominant_pollutant']} levels detected. "
                f"Reduce prolonged outdoor exposure, especially for sensitive groups. "
            )
        elif aqi > 100:
            health_impact = (
                f"🟡 MODERATE air quality in {context['location']} (AQI: {int(aqi)}). "
                f"Sensitive individuals should consider limiting prolonged outdoor activities. "
            )
        else:
            health_impact = (
                f"✅ GOOD air quality in {context['location']} (AQI: {int(aqi)}). "
                f"Perfect conditions for outdoor activities and exercise! "
            )
            
        # Add user-specific guidance
        if context['user_profile'] == 'children':
            health_impact += "\n\n👶 For Children: "
            if aqi > 150:
                health_impact += "Keep children indoors. Cancel outdoor play and sports."
            elif aqi > 100:
                health_impact += "Limit outdoor play to 30-45 minutes. Watch for coughing or breathing difficulty."
            else:
                health_impact += "Great day for outdoor play!"
                
        elif context['user_profile'] == 'elderly':
            health_impact += "\n\n👴 For Seniors: "
            if aqi > 150:
                health_impact += "Stay indoors. Have respiratory medications ready."
            elif aqi > 100:
                health_impact += "Limit outdoor time. Take breaks frequently."
            else:
                health_impact += "Ideal for morning walks and light exercise."
                
        return health_impact
        
    def analyze_health_sentiment(self, text: str) -> Dict:
        """
        Analyze sentiment of health-related text using VADER
        VADER is specifically tuned for social media and health-related text
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment analysis results with compound score
        """
        if self.vader_analyzer and VADER_AVAILABLE:
            # Use VADER for accurate sentiment analysis
            scores = self.vader_analyzer.polarity_scores(text)
            compound = scores['compound']
            
            if compound >= 0.05:
                sentiment = 'POSITIVE'
                is_positive = True
            elif compound <= -0.05:
                sentiment = 'NEGATIVE'
                is_positive = False
            else:
                sentiment = 'NEUTRAL'
                is_positive = False
                
            return {
                'sentiment': sentiment,
                'confidence': abs(compound),
                'is_positive': is_positive,
                'scores': {
                    'positive': scores['pos'],
                    'negative': scores['neg'],
                    'neutral': scores['neu'],
                    'compound': compound
                }
            }
        
        # Fallback to keyword-based analysis
        positive_words = {'good', 'excellent', 'safe', 'healthy', 'clean', 'fresh', 'great', 'perfect'}
        negative_words = {'bad', 'unhealthy', 'dangerous', 'hazardous', 'polluted', 'toxic', 'poor', 'severe'}
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            return {'sentiment': 'POSITIVE', 'confidence': 0.7, 'is_positive': True}
        elif neg_count > pos_count:
            return {'sentiment': 'NEGATIVE', 'confidence': 0.7, 'is_positive': False}
        else:
            return {'sentiment': 'NEUTRAL', 'confidence': 0.5, 'is_positive': False}
    
    def analyze_user_query(self, query: str) -> Dict:
        """
        Analyze user query to understand intent and extract key information
        Uses TextBlob for NLP processing
        
        Args:
            query: User's question or input
            
        Returns:
            Analysis with intent, entities, and sentiment
        """
        result = {
            'original_query': query,
            'sentiment': self.analyze_health_sentiment(query),
            'intent': 'unknown',
            'entities': [],
            'keywords': []
        }
        
        if TEXTBLOB_AVAILABLE:
            blob = TextBlob(query)
            
            # Extract noun phrases (key topics)
            result['keywords'] = list(blob.noun_phrases)
            
            # Get word tags for entity extraction
            for word, tag in blob.tags:
                if tag in ['NN', 'NNP', 'NNS']:  # Nouns
                    result['entities'].append({'word': word, 'type': 'noun'})
                elif tag in ['CD']:  # Numbers (could be AQI values)
                    result['entities'].append({'word': word, 'type': 'number'})
        
        # Intent classification based on patterns
        query_lower = query.lower()
        
        if any(w in query_lower for w in ['safe', 'can i', 'should i', 'okay to']):
            result['intent'] = 'safety_inquiry'
        elif any(w in query_lower for w in ['what is', 'what does', 'explain', 'mean']):
            result['intent'] = 'information_request'
        elif any(w in query_lower for w in ['when', 'best time', 'what time']):
            result['intent'] = 'timing_inquiry'
        elif any(w in query_lower for w in ['exercise', 'run', 'jog', 'workout', 'outdoor']):
            result['intent'] = 'activity_inquiry'
        elif any(w in query_lower for w in ['health', 'symptom', 'affect', 'impact']):
            result['intent'] = 'health_inquiry'
        elif any(w in query_lower for w in ['protect', 'mask', 'prevent', 'avoid']):
            result['intent'] = 'protection_inquiry'
        elif any(w in query_lower for w in ['recommend', 'suggest', 'advice', 'tip']):
            result['intent'] = 'recommendation_request'
            
        return result
    
    def generate_smart_response(self, query: str, aqi_context: Dict) -> str:
        """
        Generate intelligent response based on query analysis and AQI context
        
        Args:
            query: User's question
            aqi_context: Current AQI data and context
            
        Returns:
            Smart, contextual response
        """
        analysis = self.analyze_user_query(query)
        intent = analysis['intent']
        aqi = aqi_context.get('aqi', 100)
        
        responses = {
            'safety_inquiry': self._generate_safety_response(aqi, aqi_context),
            'activity_inquiry': self._generate_activity_response(aqi, aqi_context),
            'health_inquiry': self._generate_health_response(aqi, aqi_context),
            'timing_inquiry': self._generate_timing_response(aqi, aqi_context),
            'protection_inquiry': self._generate_protection_response(aqi, aqi_context),
            'recommendation_request': self._generate_recommendation_response(aqi, aqi_context),
            'information_request': self._generate_info_response(query, aqi_context),
        }
        
        response = responses.get(intent, self._generate_default_response(aqi, aqi_context))
        
        # Add sentiment-aware closing
        if analysis['sentiment']['is_positive']:
            response += "\n\n😊 Great to see your interest in air quality awareness!"
        elif analysis['sentiment']['sentiment'] == 'NEGATIVE':
            response += "\n\n💪 Stay informed and take precautions. Your health matters!"
            
        return response
    
    def _generate_safety_response(self, aqi: float, context: Dict) -> str:
        if aqi <= 50:
            return f"✅ Yes, it's completely safe! With AQI at {int(aqi)}, air quality is excellent. Enjoy any outdoor activities!"
        elif aqi <= 100:
            return f"🟢 Generally safe for most people (AQI: {int(aqi)}). Sensitive individuals should monitor how they feel."
        elif aqi <= 150:
            return f"🟡 Caution advised (AQI: {int(aqi)}). Limit prolonged outdoor exposure, especially for children, elderly, and those with respiratory conditions."
        elif aqi <= 200:
            return f"🟠 Not recommended to go outside (AQI: {int(aqi)}). Everyone may experience health effects. Stay indoors if possible."
        else:
            return f"🔴 Stay indoors! AQI is {int(aqi)} - hazardous levels. Only go out for essentials with proper N95 protection."
    
    def _generate_activity_response(self, aqi: float, context: Dict) -> str:
        if aqi <= 50:
            return "🏃 Perfect for all activities! Running, cycling, sports - everything is great today!"
        elif aqi <= 100:
            return "🚶 Most activities are fine. Consider lighter intensity if exercising for extended periods."
        elif aqi <= 150:
            return "🏠 Switch to indoor activities. Gym workouts, yoga, or home exercises are better choices today."
        else:
            return "⛔ Avoid all strenuous activities. Even indoor exercise should be light. Rest and stay hydrated."
    
    def _generate_health_response(self, aqi: float, context: Dict) -> str:
        return f"""🏥 Health Impact at AQI {int(aqi)}:

{'✅ No significant health impacts expected.' if aqi <= 50 else ''}
{'⚠️ Sensitive groups may experience mild irritation.' if 50 < aqi <= 100 else ''}
{'🔸 Possible symptoms: throat irritation, mild coughing, eye discomfort.' if 100 < aqi <= 150 else ''}
{'🔶 Likely symptoms: breathing difficulty, coughing, fatigue. Seek fresh air.' if 150 < aqi <= 200 else ''}
{'🔴 Serious effects: respiratory distress, chest pain, severe coughing. Seek medical help if symptoms persist.' if aqi > 200 else ''}

At-risk groups: Children, elderly, pregnant women, and those with asthma/heart conditions should take extra precautions."""
    
    def _generate_timing_response(self, aqi: float, context: Dict) -> str:
        return """⏰ Best Times for Outdoor Activities:

🌅 Early Morning (6-8 AM): Usually the cleanest air before traffic builds up
🌙 Late Evening (after 8 PM): Pollution typically decreases after rush hour
☀️ Avoid 2-6 PM: Ozone levels peak in afternoon heat
🚗 Avoid Rush Hours: 7-9 AM and 5-7 PM have highest traffic pollution

💡 Pro Tip: Check real-time AQI before heading out - conditions can change quickly!"""
    
    def _generate_protection_response(self, aqi: float, context: Dict) -> str:
        return f"""😷 Protection Measures for AQI {int(aqi)}:

{'✅ No special protection needed. Enjoy the fresh air!' if aqi <= 50 else ''}
{'🏠 Keep windows open for ventilation.' if aqi <= 50 else '🏠 Keep windows closed to prevent outdoor air from entering.'}
{'😷 N95/KN95 masks recommended for outdoor activities.' if aqi > 100 else ''}
{'🌬️ Run HEPA air purifiers indoors.' if aqi > 100 else ''}
{'💧 Stay hydrated - helps your body flush out pollutants.' if aqi > 100 else ''}
{'🚿 Shower after outdoor exposure to remove particles from skin/hair.' if aqi > 150 else ''}
{'🏥 Have medications ready if you have respiratory conditions.' if aqi > 150 else ''}"""
    
    def _generate_recommendation_response(self, aqi: float, context: Dict) -> str:
        if aqi <= 50:
            activities = "outdoor running, cycling, hiking, picnics, outdoor sports"
        elif aqi <= 100:
            activities = "light outdoor walks, gardening, short outdoor exercises"
        elif aqi <= 150:
            activities = "indoor gym, yoga, home workouts, swimming (indoor pools)"
        else:
            activities = "reading, indoor games, streaming, light stretching at home"
            
        return f"""📋 Recommendations for Today (AQI: {int(aqi)}):

Suggested Activities: {activities}

Do's:
✅ {'Enjoy outdoor activities freely' if aqi <= 100 else 'Stay indoors as much as possible'}
✅ {'Stay hydrated' if aqi > 50 else 'Take advantage of the clean air'}
✅ Monitor air quality throughout the day

Don'ts:
❌ {'Ignore symptoms if you feel unwell' if aqi > 100 else 'Miss this great weather!'}
❌ {'Exercise outdoors' if aqi > 150 else 'Overexert in the afternoon heat' if aqi > 100 else 'Nothing to avoid today!'}"""
    
    def _generate_info_response(self, query: str, context: Dict) -> str:
        query_lower = query.lower()
        
        if 'aqi' in query_lower:
            return """📊 What is AQI (Air Quality Index)?

AQI is a standardized scale from 0-500 measuring air pollution levels:

| AQI Range | Category | Health Impact |
|-----------|----------|---------------|
| 0-50 | 🟢 Good | No risk |
| 51-100 | 🟡 Moderate | Sensitive groups at mild risk |
| 101-150 | 🟠 Unhealthy (Sensitive) | Sensitive groups affected |
| 151-200 | 🔴 Unhealthy | Everyone may experience effects |
| 201-300 | 🟣 Very Unhealthy | Health warnings for all |
| 301-500 | ⚫ Hazardous | Emergency conditions |

Main Pollutants Measured: PM2.5, PM10, Ozone (O₃), NO₂, SO₂, CO"""
        
        return f"ℹ️ Current AQI is {int(context.get('aqi', 0))}. Ask me about safety, activities, health impacts, or protection measures!"
    
    def _generate_default_response(self, aqi: float, context: Dict) -> str:
        return f"""📍 Current Air Quality Summary (AQI: {int(aqi)})

{'🟢 Air quality is GOOD. Enjoy your day outdoors!' if aqi <= 50 else ''}
{'🟡 Air quality is MODERATE. Most activities are safe.' if 50 < aqi <= 100 else ''}
{'🟠 Air quality is concerning. Limit outdoor exposure.' if 100 < aqi <= 150 else ''}
{'🔴 Air quality is UNHEALTHY. Stay indoors when possible.' if 150 < aqi <= 200 else ''}
{'⚫ HAZARDOUS conditions. Avoid all outdoor activities.' if aqi > 200 else ''}

Ask me about: safety, activities, health impacts, best times, or protection tips!"""
            
    def generate_activity_suggestion(
        self,
        aqi: float,
        activities: List[Dict],
        user_context: Dict
    ) -> str:
        """
        Generate natural language activity suggestions
        
        Args:
            aqi: Current AQI
            activities: List of recommended activities
            user_context: User preferences and constraints
            
        Returns:
            Natural language suggestions
        """
        if not activities:
            return "Current air quality limits outdoor activities. Consider indoor alternatives."
            
        # Create suggestion prompt
        top_activities = [a['activity'] for a in activities[:3]]
        
        if aqi <= 50:
            intro = f"✨ Perfect Air Quality! Here's what you can enjoy:"
        elif aqi <= 100:
            intro = f"🌤️ Good Conditions. Recommended activities:"
        elif aqi <= 150:
            intro = f"⚠️ Moderate Air. Safe indoor options:"
        else:
            intro = f"🏠 Stay Indoors. Best activities for today:"
            
        suggestions = intro + "\n\n"
        
        for i, activity_dict in enumerate(activities[:5], 1):
            activity = activity_dict['activity']
            safety = activity_dict['safety_level']
            recommendation = activity_dict['recommendation']
            
            emoji = "✅" if safety == "Safe" else "⚠️" if safety == "Caution" else "❌"
            suggestions += f"{i}. {emoji} {activity} - {recommendation}\n"
            
        return suggestions
        
    def summarize_air_quality_report(
        self,
        full_report: str,
        max_length: int = 150
    ) -> str:
        """
        Summarize lengthy air quality reports using extractive summarization
        
        Args:
            full_report: Full report text
            max_length: Maximum summary length
            
        Returns:
            Summarized text
        """
        if len(full_report) <= max_length:
            return full_report
        
        # Extract key sentences based on importance keywords
        sentences = full_report.replace('!', '.').replace('?', '.').split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Priority keywords for AQI reports
        priority_keywords = ['aqi', 'health', 'risk', 'warning', 'safe', 'danger', 'recommend']
        
        # Score sentences by keyword presence
        scored = []
        for s in sentences:
            score = sum(1 for kw in priority_keywords if kw in s.lower())
            scored.append((score, s))
        
        # Take highest scored sentences that fit
        scored.sort(reverse=True, key=lambda x: x[0])
        summary = ""
        for _, sentence in scored:
            if len(summary) + len(sentence) + 2 <= max_length:
                summary += sentence + ". "
            if len(summary) >= max_length * 0.8:
                break
        
        return summary.strip() if summary else full_report[:max_length] + "..."
        
    def generate_personalized_tips(
        self,
        aqi: float,
        pollutants: Dict,
        user_profile: Dict,
        location: str
    ) -> List[str]:
        """
        Generate AI-powered personalized health tips
        
        Args:
            aqi: Current AQI
            pollutants: Pollutant levels
            user_profile: User information
            location: Location name
            
        Returns:
            List of personalized tips
        """
        try:
            tips = []
            
            # Dominant pollutant (with fallback)
            if not pollutants or len(pollutants) == 0:
                pollutants = {'pm25': aqi * 0.6}  # Default estimate
            
            dominant = max(pollutants.items(), key=lambda x: x[1])
            pollutant_name = dominant[0].upper()
            
            # AQI-based tips
            if aqi > 250:
                tips.extend([
                    f"🚨 EMERGENCY: AQI {int(aqi)} is hazardous. Seal all windows and use HEPA air purifiers.",
                    f"🏥 Health Alert: Monitor for symptoms - chest pain, irregular heartbeat, or severe coughing.",
                    f"😷 Protection: N95/N99 masks are MANDATORY for any outdoor exposure.",
                    f"💊 Medication: Have emergency medications and doctor contact ready."
                ])
            elif aqi > 200:
                tips.extend([
                    f"⚠️ VERY UNHEALTHY: Avoid all outdoor activities. AQI is {int(aqi)}.",
                    f"🪟 Indoor Air: Keep windows closed. Run air purifiers on high.",
                    f"🚫 Exercise: NO outdoor exercise. Indoor activities only.",
                    f"👥 Check on Others: Call vulnerable family members - children, elderly."
                ])
            elif aqi > 150:
                tips.extend([
                    f"🔴 UNHEALTHY: Limit outdoor time. Current AQI: {int(aqi)}.",
                    f"😷 Masks: Wear N95 masks for outdoor errands.",
                    f"🏃 Exercise: Postpone intense outdoor workouts.",
                    f"💧 Hydration: Drink extra water to help flush pollutants."
                ])
            elif aqi > 100:
                tips.extend([
                    f"🟡 MODERATE: Sensitive groups should reduce prolonged exposure (AQI: {int(aqi)}).",
                    f"⏱️ Timing: Exercise during early morning when AQI is typically lower.",
                    f"😷 Optional: Consider masks during extended outdoor activities."
                ])
            else:
                tips.extend([
                    f"✅ EXCELLENT: Perfect air quality (AQI: {int(aqi)}). Enjoy outdoor activities!",
                    f"🏃 Exercise: Great conditions for running, cycling, and sports.",
                    f"🌳 Nature: Ideal time for parks, hiking, and outdoor recreation."
                ])
                
            # Pollutant-specific tips
            if pollutant_name == 'PM25' and dominant[1] > 50:
                tips.append(f"⚠️ PM2.5 Alert: Fine particles at {int(dominant[1])} µg/m³. These penetrate deep into lungs.")
            elif pollutant_name == 'O3' and dominant[1] > 60:
                tips.append(f"☀️ Ozone Alert: Avoid afternoon sun. Ozone peaks between 2-6 PM.")
            elif pollutant_name == 'NO2' and dominant[1] > 40:
                tips.append(f"🚗 Traffic Pollution: High NO₂ from vehicles. Avoid busy roads.")
                
            # User-specific tips
            age = user_profile.get('age', 30)
            has_respiratory = user_profile.get('respiratory_condition', False)
            has_heart = user_profile.get('heart_condition', False)
            
            if has_respiratory and aqi > 100:
                tips.append("🫁 Asthma/COPD: Have rescue inhaler ready. Monitor breathing closely.")
            if has_heart and aqi > 150:
                tips.append("❤️ Heart Condition: Avoid exertion. Watch for chest discomfort or fatigue.")
            if age < 12 and aqi > 100:
                tips.append("👶 Children: Kids breathe faster - more susceptible. Limit outdoor play.")
            if age > 65 and aqi > 100:
                tips.append("👴 Seniors: Elderly are high-risk. Rest frequently, stay hydrated.")
                
            # Time-based tips
            hour = datetime.now().hour
            if 6 <= hour <= 9 and aqi < 100:
                tips.append("🌅 Morning Freshness: Best air quality of the day. Perfect for morning walks!")
            elif 18 <= hour <= 21 and aqi > 100:
                tips.append("🌆 Evening Traffic: Rush hour increases pollution. Stay indoors if possible.")
                
            return tips[:8]  # Return top 8 tips
            
        except Exception as e:
            logger.error(f"Error generating tips: {e}")
            # Fallback: Return basic tips based on AQI
            if aqi > 150:
                return ["🔴 Poor air quality. Stay indoors.", "😷 Wear N95 mask if going outside."]
            elif aqi > 100:
                return ["🟡 Moderate air quality. Limit prolonged exposure.", "⏱️ Exercise during cleaner hours."]
            else:
                return ["✅ Good air quality. Enjoy outdoor activities!"]


class SmartQAEngine:
    """
    NLP-powered Q&A engine using semantic similarity and intent understanding
    Can answer any question about air quality, health, and activities
    """
    
    def __init__(self):
        logger.info("Smart NLP Q&A engine initializing...")
        
        # Initialize VADER for sentiment
        self.vader = SentimentIntensityAnalyzer() if VADER_AVAILABLE else None
        
        # Knowledge base - semantic topics with embeddings
        self.knowledge_base = self._build_knowledge_base()
        
        # Intent patterns with semantic variations
        self.intent_patterns = {
            'safety': {
                'keywords': ['safe', 'danger', 'risk', 'harm', 'okay', 'fine', 'alright', 'harmful', 'hazardous', 'hazard', 'unsafe', 'risky', 'bad', 'good', 'terrible', 'awful', 'toxic', 'children', 'child', 'kids', 'kid'],
                'phrases': ['is it safe', 'safe for', 'can i go', 'should i go', 'am i safe', 'is it okay', 'is it fine', 'dangerous to', 'risky to', 'is it hazardous', 'hazardous outside', 'is it bad', 'is it good', 'safe outside', 'unsafe outside', 'go outside', 'step outside', 'safe for children', 'safe for kids'],
                'semantic_score': 1.0
            },
            'activity': {
                'keywords': ['run', 'jog', 'walk', 'exercise', 'sport', 'cycle', 'bike', 'swim', 'play', 'workout', 'gym', 'hike', 'yoga', 'kid', 'child', 'children', 'baby'],
                'phrases': ['should i exercise', 'can i run', 'outdoor activities', 'go for a walk', 'play outside', 'physical activity', 'safe for kids', 'safe for children', 'children play', 'kids play'],
                'semantic_score': 1.0
            },
            'health': {
                'keywords': ['health', 'symptom', 'cough', 'breathe', 'asthma', 'lung', 'heart', 'allergy', 'sick', 'ill', 'disease', 'affect', 'effect', 'impact'],
                'phrases': ['health effects', 'health impact', 'affect my health', 'breathing problems', 'cause symptoms', 'health risks', 'how does it affect', 'what happens to'],
                'semantic_score': 1.0
            },
            'protection': {
                'keywords': ['mask', 'protect', 'prevent', 'filter', 'purifier', 'avoid', 'shield', 'defense', 'safety measures', 'n95', 'kn95'],
                'phrases': ['how to protect', 'stay safe', 'what should i wear', 'need a mask', 'air purifier', 'protect myself', 'what mask', 'which mask'],
                'semantic_score': 1.0
            },
            'timing': {
                'keywords': ['when', 'time', 'hour', 'morning', 'evening', 'afternoon', 'night', 'best', 'optimal', 'schedule'],
                'phrases': ['best time', 'when to go', 'what time', 'good time', 'optimal time', 'when should i', 'when can i'],
                'semantic_score': 1.0
            },
            'explanation': {
                'keywords': ['what', 'why', 'how', 'explain', 'meaning', 'definition', 'understand', 'describe', 'mean', 'pm2.5', 'pm25', 'pm10', 'ozone', 'pollutant'],
                'phrases': ['what is aqi', 'what does', 'how does', 'why is', 'explain the', 'tell me about', 'what are', 'what is pm', 'what is ozone', 'what is pm2.5', 'what is pm25', 'what is pm10'],
                'semantic_score': 1.0
            },
            'recommendation': {
                'keywords': ['recommend', 'suggest', 'advice', 'tip', 'should', 'best', 'ideal', 'prefer', 'do'],
                'phrases': ['what should i do', 'give me advice', 'any tips', 'recommend for', 'suggest me', 'what do you recommend', 'what can i do', 'what to do'],
                'semantic_score': 1.0
            },
            'comparison': {
                'keywords': ['better', 'worse', 'compare', 'versus', 'difference', 'than', 'prefer', 'vs'],
                'phrases': ['which is better', 'indoor vs outdoor', 'compare to', 'difference between', 'is it better', 'indoor or outdoor'],
                'semantic_score': 1.0
            },
            'forecast': {
                'keywords': ['tomorrow', 'later', 'forecast', 'predict', 'future', 'upcoming', 'next', 'will'],
                'phrases': ['will it get better', 'tomorrow\'s aqi', 'air quality later', 'going to improve', 'what about tomorrow'],
                'semantic_score': 1.0
            },
            'location': {
                'keywords': ['where', 'location', 'area', 'city', 'region', 'place', 'nearby', 'local'],
                'phrases': ['in my area', 'near me', 'in this city', 'around here', 'local air quality'],
                'semantic_score': 1.0
            }
        }
        
        # Pollutant knowledge
        self.pollutant_info = {
            'pm25': {
                'name': 'PM2.5 (Fine Particulate Matter)',
                'description': 'Tiny particles smaller than 2.5 micrometers that can penetrate deep into lungs and bloodstream',
                'sources': 'Vehicle emissions, industrial processes, wildfires, dust',
                'health_effects': 'Respiratory issues, heart disease, stroke, lung cancer with long-term exposure',
                'protection': 'N95/N99 masks filter PM2.5 effectively. HEPA air purifiers help indoors.'
            },
            'pm10': {
                'name': 'PM10 (Coarse Particulate Matter)',
                'description': 'Larger particles 2.5-10 micrometers, usually dust and pollen',
                'sources': 'Road dust, construction, agriculture, pollen',
                'health_effects': 'Throat and nose irritation, aggravates asthma',
                'protection': 'Regular masks help. Keep windows closed on dusty days.'
            },
            'o3': {
                'name': 'Ozone (O₃)',
                'description': 'Ground-level ozone formed by sunlight reacting with pollutants',
                'sources': 'Chemical reaction of NOx and VOCs in sunlight - highest on hot sunny days',
                'health_effects': 'Chest pain, coughing, throat irritation, worsens bronchitis and asthma',
                'protection': 'Stay indoors during afternoon peak (2-6 PM). Exercise early morning.'
            },
            'no2': {
                'name': 'Nitrogen Dioxide (NO₂)',
                'description': 'Reddish-brown gas from burning fuel',
                'sources': 'Vehicle emissions, power plants, industrial facilities',
                'health_effects': 'Airway inflammation, reduced lung function, respiratory infections',
                'protection': 'Avoid busy roads and traffic areas. Stay indoors during rush hour.'
            },
            'so2': {
                'name': 'Sulfur Dioxide (SO₂)',
                'description': 'Colorless gas with sharp smell',
                'sources': 'Burning coal and oil, industrial processes, volcanoes',
                'health_effects': 'Breathing difficulty, eye irritation, aggravates heart disease',
                'protection': 'Industrial masks for high exposure. Avoid areas near factories.'
            },
            'co': {
                'name': 'Carbon Monoxide (CO)',
                'description': 'Colorless, odorless poisonous gas',
                'sources': 'Vehicle exhaust, incomplete combustion, fires',
                'health_effects': 'Headaches, dizziness, confusion, fatal at high levels',
                'protection': 'Never idle cars in enclosed spaces. CO detectors for homes.'
            }
        }
        
        logger.info("✓ Smart NLP Q&A engine ready")
    
    def _build_knowledge_base(self) -> Dict:
        """Build comprehensive knowledge base for air quality Q&A"""
        return {
            'aqi_basics': {
                'topic': 'AQI Basics',
                'content': """The Air Quality Index (AQI) is a standardized measurement scale from 0-500 that indicates how polluted the air is. 
                    The scale is divided into categories:
                    • 0-50 (Good): Air quality is satisfactory, no health risk
                    • 51-100 (Moderate): Acceptable, sensitive people may experience minor effects
                    • 101-150 (Unhealthy for Sensitive Groups): Children, elderly, and those with respiratory conditions affected
                    • 151-200 (Unhealthy): Everyone may experience health effects
                    • 201-300 (Very Unhealthy): Health warnings for entire population
                    • 301-500 (Hazardous): Emergency conditions, everyone affected
                    
                    AQI is calculated from concentrations of major pollutants: PM2.5, PM10, Ozone, NO2, SO2, and CO.""",
                'keywords': ['aqi', 'air quality index', 'scale', 'measurement', 'pollution level', 'categories']
            },
            'health_impacts': {
                'topic': 'Health Impacts',
                'content': """Air pollution affects health in many ways:
                    
                    Short-term effects:
                    • Eye, nose, and throat irritation
                    • Coughing and wheezing
                    • Shortness of breath
                    • Chest tightness
                    • Headaches and fatigue
                    • Aggravation of asthma and allergies
                    
                    Long-term effects:
                    • Chronic respiratory diseases
                    • Reduced lung function
                    • Increased risk of lung cancer
                    • Cardiovascular disease
                    • Neurological effects
                    • Premature death
                    
                    Vulnerable groups: children, elderly, pregnant women, people with asthma/heart conditions.""",
                'keywords': ['health', 'effect', 'impact', 'symptom', 'disease', 'lung', 'heart', 'breathing']
            },
            'protection_measures': {
                'topic': 'Protection Measures',
                'content': """How to protect yourself from air pollution:
                    
                    Outdoor protection:
                    • Wear N95/N99 masks (filter 95-99% of particles)
                    • Check AQI before going out
                    • Avoid exercising near busy roads
                    • Plan outdoor activities for early morning when air is cleanest
                    • Shower and change clothes after being outdoors on high pollution days
                    
                    Indoor protection:
                    • Use HEPA air purifiers (removes 99.97% of particles)
                    • Keep windows and doors closed on high pollution days
                    • Avoid burning candles, incense, or smoking indoors
                    • Use exhaust fans when cooking
                    • Indoor plants can help (but aren't sufficient alone)
                    
                    Health precautions:
                    • Stay hydrated to help body flush toxins
                    • Eat antioxidant-rich foods (vitamins C, E)
                    • Have medications ready if you have respiratory conditions
                    • Monitor symptoms and seek medical help if severe""",
                'keywords': ['protect', 'mask', 'purifier', 'filter', 'prevent', 'safety', 'precaution', 'n95']
            },
            'best_times': {
                'topic': 'Best Times for Outdoor Activities',
                'content': """Optimal times for outdoor activities based on air quality patterns:
                    
                    Best times:
                    • Early morning (6-8 AM): Typically cleanest air before traffic builds
                    • Late evening (after 8 PM): Pollution decreases after rush hour
                    • After rain: Rain clears particles from air
                    • Weekends: Less traffic means lower pollution in cities
                    
                    Worst times to avoid:
                    • Rush hours (7-9 AM, 5-8 PM): Peak traffic pollution
                    • Hot afternoons (2-6 PM): Ozone levels peak
                    • During wildfires/dust storms: Stay indoors completely
                    
                    Always check real-time AQI before heading out - conditions can change quickly!""",
                'keywords': ['time', 'when', 'morning', 'evening', 'afternoon', 'best', 'optimal', 'schedule']
            },
            'exercise_guidance': {
                'topic': 'Exercise and Air Quality',
                'content': """Exercise guidelines based on air quality:
                    
                    AQI 0-50 (Good): All outdoor activities safe. Great for running, cycling, sports.
                    
                    AQI 51-100 (Moderate): Most activities fine. Consider shorter duration for intense exercise.
                    
                    AQI 101-150 (Unhealthy for Sensitive): 
                    • Healthy adults: Reduce intensity/duration
                    • Sensitive groups: Exercise indoors
                    
                    AQI 151-200 (Unhealthy):
                    • Switch to indoor exercise (gym, yoga, home workouts)
                    • Light walking okay with mask
                    • Avoid intense cardio outdoors
                    
                    AQI 201+ (Very Unhealthy/Hazardous):
                    • No outdoor exercise
                    • Even indoor exercise should be light
                    • Rest and stay hydrated
                    
                    Why exercise is riskier in bad air: You breathe deeper and faster, inhaling 10-20x more pollutants!""",
                'keywords': ['exercise', 'run', 'jog', 'workout', 'sport', 'gym', 'outdoor', 'fitness', 'cardio']
            },
            'children_elderly': {
                'topic': 'Children and Elderly',
                'content': """Special considerations for vulnerable groups:
                    
                    Children:
                    • Breathe faster than adults, inhaling more pollutants
                    • Lungs still developing, more susceptible to damage
                    • Play closer to ground where pollution concentrates
                    • Limit outdoor play when AQI > 100
                    • Schools should move recess indoors on high pollution days
                    
                    Elderly (65+):
                    • Often have underlying heart/lung conditions
                    • Immune system less effective
                    • Higher risk of serious health effects
                    • Should stay indoors when AQI > 100
                    • Have emergency medications accessible
                    • Check on elderly neighbors during pollution events
                    
                    Both groups should have regular checkups and respiratory health monitoring.""",
                'keywords': ['child', 'kid', 'baby', 'elderly', 'senior', 'old', 'age', 'vulnerable', 'young']
            },
            'indoor_air': {
                'topic': 'Indoor Air Quality',
                'content': """Managing indoor air quality:
                    
                    Indoor pollutant sources:
                    • Cooking (especially gas stoves) - use exhaust fans
                    • Cleaning products - use natural alternatives
                    • Furniture/carpets - can emit VOCs
                    • Mold and dust mites
                    • Smoking/vaping
                    • Candles and incense
                    
                    Improving indoor air:
                    • HEPA air purifiers (size for room)
                    • Regular ventilation when outdoor air is good
                    • Change HVAC filters regularly (MERV 13+ recommended)
                    • Keep humidity 30-50% to prevent mold
                    • Clean regularly to reduce dust
                    • Add some indoor plants (spider plant, peace lily)
                    
                    Indoor air can be 2-5x more polluted than outdoor air if not managed!""",
                'keywords': ['indoor', 'inside', 'home', 'house', 'room', 'purifier', 'ventilation', 'filter']
            }
        }
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts using word overlap and context"""
        text1_lower = text1.lower()
        text2_lower = text2.lower()
        
        # Tokenize and clean
        words1 = set(text1_lower.split())
        words2 = set(text2_lower.split())
        
        # Remove stop words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                      'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                      'should', 'may', 'might', 'must', 'shall', 'can', 'to', 'of', 'in',
                      'for', 'on', 'with', 'at', 'by', 'from', 'as', 'it', 'its', 'this',
                      'that', 'these', 'those', 'i', 'me', 'my', 'we', 'our', 'you', 'your'}
        
        words1 = words1 - stop_words
        words2 = words2 - stop_words
        
        if not words1 or not words2:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _detect_intent(self, question: str) -> Dict:
        """Detect user intent using NLP and pattern matching"""
        question_lower = question.lower()
        intent_scores = {}
        
        for intent, patterns in self.intent_patterns.items():
            score = 0.0
            
            # Check keywords
            keyword_matches = sum(1 for kw in patterns['keywords'] if kw in question_lower)
            score += keyword_matches * 0.3
            
            # Check phrases (higher weight)
            phrase_matches = sum(1 for phrase in patterns['phrases'] if phrase in question_lower)
            score += phrase_matches * 0.5
            
            # Use TextBlob for additional NLP if available
            if TEXTBLOB_AVAILABLE:
                blob = TextBlob(question)
                # Check noun phrases
                for np in blob.noun_phrases:
                    if any(kw in np for kw in patterns['keywords']):
                        score += 0.2
            
            intent_scores[intent] = score
        
        # Get top intent
        if intent_scores:
            top_intent = max(intent_scores, key=intent_scores.get)
            top_score = intent_scores[top_intent]
            return {
                'primary_intent': top_intent if top_score > 0 else 'general',
                'confidence': min(top_score / 2.0, 1.0),  # Normalize to 0-1
                'all_intents': intent_scores
            }
        
        return {'primary_intent': 'general', 'confidence': 0.5, 'all_intents': {}}
    
    def _find_relevant_knowledge(self, question: str) -> List[Dict]:
        """Find relevant knowledge base entries for the question"""
        question_lower = question.lower()
        relevant = []
        
        for key, kb_entry in self.knowledge_base.items():
            score = 0.0
            
            # Keyword matching
            for keyword in kb_entry['keywords']:
                if keyword in question_lower:
                    score += 0.3
            
            # Semantic similarity with topic
            topic_similarity = self._calculate_semantic_similarity(question, kb_entry['topic'])
            score += topic_similarity * 0.4
            
            # Content word matching
            content_words = set(kb_entry['content'].lower().split())
            question_words = set(question_lower.split())
            content_match = len(content_words & question_words) / max(len(question_words), 1)
            score += content_match * 0.3
            
            if score > 0.1:
                relevant.append({
                    'key': key,
                    'topic': kb_entry['topic'],
                    'content': kb_entry['content'],
                    'score': score
                })
        
        # Sort by relevance
        relevant.sort(key=lambda x: x['score'], reverse=True)
        return relevant[:3]  # Top 3 most relevant
    
    def _extract_aqi_context(self, context) -> Dict:
        """Extract AQI and other relevant info from context (dict or string)"""
        import re
        
        result = {
            'aqi': None,
            'category': 'unknown',
            'location': 'your location',
            'pollutant': None
        }
        
        # If context is a dict, extract directly
        if isinstance(context, dict):
            result['aqi'] = context.get('aqi')
            result['location'] = context.get('location', 'your location')
            result['pollutant'] = context.get('dominant_pollutant')
            return result
        
        # If context is a string, parse it
        context_str = str(context)
        
        # Extract AQI number - handles "Current AQI in X is 85" format
        aqi_match = re.search(r'aqi.*?is\s+([0-9]+)', context_str.lower())
        if aqi_match:
            result['aqi'] = int(aqi_match.group(1))
        
        # Extract category
        categories = ['good', 'moderate', 'unhealthy', 'hazardous', 'very unhealthy']
        for cat in categories:
            if cat in context_str.lower():
                result['category'] = cat.title()
                break
        
        # Extract pollutant
        pollutants = ['pm25', 'pm2.5', 'pm10', 'ozone', 'o3', 'no2', 'so2', 'co']
        for p in pollutants:
            if p in context_str.lower():
                result['pollutant'] = p.upper().replace('.', '')
                break
        
        return result
    
    def _detect_pollutant_question(self, question: str) -> Optional[str]:
        """Detect if question is about a specific pollutant"""
        question_lower = question.lower()
        
        pollutant_aliases = {
            'pm25': ['pm2.5', 'pm25', 'fine particle', 'fine particulate'],
            'pm10': ['pm10', 'coarse particle', 'dust'],
            'o3': ['ozone', 'o3', 'smog'],
            'no2': ['no2', 'nitrogen dioxide', 'nitrogen'],
            'so2': ['so2', 'sulfur dioxide', 'sulfur'],
            'co': ['carbon monoxide', 'co poisoning']
        }
        
        for pollutant, aliases in pollutant_aliases.items():
            if any(alias in question_lower for alias in aliases):
                return pollutant
        
        return None
    
    def answer_question(self, question: str, context) -> Dict:
        """
        Answer any question about air quality using NLP understanding
        
        Args:
            question: User's question (can be any natural language)
            context: Current AQI context information (dict or string)
            
        Returns:
            Comprehensive answer with confidence and sources
        """
        # Analyze question
        intent_result = self._detect_intent(question)
        intent = intent_result['primary_intent']
        aqi_context = self._extract_aqi_context(context)
        aqi = aqi_context['aqi'] if aqi_context['aqi'] is not None else 100  # Default if not found
        
        # Check for pollutant-specific question
        pollutant = self._detect_pollutant_question(question)
        if pollutant and pollutant in self.pollutant_info:
            p_info = self.pollutant_info[pollutant]
            answer = f"""{p_info['name']}

📖 What is it? {p_info['description']}

🏭 Sources: {p_info['sources']}

🏥 Health Effects: {p_info['health_effects']}

😷 Protection: {p_info['protection']}"""
            
            return {
                'answer': answer,
                'confidence': 0.92,
                'source': 'pollutant_knowledge',
                'intent': 'explanation'
            }
        
        # Find relevant knowledge
        relevant_kb = self._find_relevant_knowledge(question)
        
        # Generate answer based on intent and context
        if intent == 'safety':
            answer = self._generate_safety_answer(aqi, aqi_context, question)
        elif intent == 'activity':
            answer = self._generate_activity_answer(aqi, aqi_context, question)
        elif intent == 'health':
            answer = self._generate_health_answer(aqi, aqi_context, question)
        elif intent == 'protection':
            answer = self._generate_protection_answer(aqi, aqi_context)
        elif intent == 'timing':
            answer = self._generate_timing_answer(aqi)
        elif intent == 'explanation':
            answer = self._generate_explanation_answer(question, relevant_kb)
        elif intent == 'recommendation':
            answer = self._generate_recommendation_answer(aqi, aqi_context)
        elif intent == 'comparison':
            answer = self._generate_comparison_answer(question, aqi)
        elif intent == 'forecast':
            answer = self._generate_forecast_answer(aqi, aqi_context)
        elif intent == 'location':
            answer = self._generate_location_answer(aqi_context)
        else:
            # General answer using knowledge base
            answer = self._generate_general_answer(question, aqi, relevant_kb)
        
        # Calculate confidence based on intent detection and knowledge match
        confidence = intent_result['confidence']
        if relevant_kb:
            confidence = min(confidence + relevant_kb[0]['score'] * 0.3, 0.98)
        
        return {
            'answer': answer,
            'confidence': max(confidence, 0.7),
            'source': 'nlp_understanding',
            'intent': intent,
            'related_topics': [kb['topic'] for kb in relevant_kb[:2]]
        }
    
    def _generate_safety_answer(self, aqi: int, ctx: Dict, question: str) -> str:
        """Generate safety-related answer"""
        question_lower = question.lower()
        
        # Check if asking about children/kids
        is_children_question = any(word in question_lower for word in ['children', 'child', 'kids', 'kid', 'baby', 'toddler'])
        
        # Check for specific activity in question
        activities = {
            'run': 'running', 'jog': 'jogging', 'walk': 'walking',
            'cycle': 'cycling', 'bike': 'biking', 'exercise': 'exercising',
            'play': 'playing outside', 'swim': 'swimming outdoors'
        }
        
        specific_activity = None
        for key, activity in activities.items():
            if key in question_lower:
                specific_activity = activity
                break
        
        if aqi <= 50:
            if is_children_question:
                return f"""✅ Yes, perfectly safe for children!

With AQI at {aqi} ({ctx['category']}), the air quality is excellent. Children can:
• Play outside freely
• Participate in sports and outdoor activities
• No restrictions needed

👶 Great day for outdoor play! No precautions needed."""
            if specific_activity:
                return f"""✅ Yes, absolutely safe for {specific_activity}!

With AQI at {aqi} ({ctx['category']}), the air quality is excellent. This is perfect weather for {specific_activity} and all outdoor activities.

🏃 Enjoy your {specific_activity}! No precautions needed."""
            return f"""✅ Completely Safe!

Current AQI is {aqi} ({ctx['category']}) - excellent air quality. It's safe to:
• Exercise outdoors
• Let children play outside
• Keep windows open
• Enjoy any outdoor activity

No special precautions needed today!"""
        
        elif aqi <= 100:
            if is_children_question:
                return f"""🟡 Generally safe for children

AQI is {aqi} ({ctx['category']}). Most children can play outside normally.

⚠️ For children with asthma or respiratory issues:
• Limit very strenuous outdoor play
• Watch for any breathing difficulty
• Take breaks during prolonged activities

👍 Overall, normal outdoor play is fine for healthy children."""
            if specific_activity:
                return f"""🟡 Generally safe for {specific_activity}

AQI is {aqi} ({ctx['category']}). Most people can enjoy {specific_activity} without issues.

⚠️ Note for sensitive groups: If you have asthma, heart conditions, or are elderly, consider shorter duration or lower intensity."""
            return f"""🟡 Mostly Safe (Moderate AQI: {aqi})

Outdoor activities are generally fine for most people.

Sensitive groups (asthma, elderly, children, heart conditions) should:
• Limit prolonged outdoor exposure
• Monitor how they feel
• Take breaks if any discomfort"""
        
        elif aqi <= 150:
            if is_children_question:
                return f"""🟠 Caution for children (AQI: {aqi})

⚠️ Children should limit outdoor activities:
• Keep outdoor play short (under 30 minutes)
• Avoid strenuous activities like running or sports
• Watch for coughing or breathing difficulty
• Move activities indoors when possible

👶 Children's lungs are developing - they're more sensitive to poor air."""
            return f"""🟠 Caution Advised (AQI: {aqi})

{'Sensitive groups should avoid ' + specific_activity if specific_activity else 'Not ideal for prolonged outdoor exposure'}.

Healthy adults: Can do light outdoor activities, but avoid intense exercise
Sensitive groups: Stay indoors, use indoor alternatives

😷 Consider wearing a mask if you must go outside."""
        
        elif aqi <= 200:
            if is_children_question:
                return f"""🔴 Not safe for children outdoors (AQI: {aqi})

❌ Keep children inside:
• Cancel outdoor play and recess
• No outdoor sports or activities
• Keep windows closed
• Use air purifiers indoors if available

👶 Children are especially vulnerable at this AQI level. Indoor play only!"""
            return f"""🔴 Not Safe for Outdoor Activities (AQI: {aqi})

{'Avoid ' + specific_activity + ' completely' if specific_activity else 'Everyone should limit outdoor exposure'}.

Everyone may experience health effects. Stay indoors when possible.

✅ Switch to indoor activities
😷 N95 mask required for essential outdoor tasks"""
        
        else:
            if is_children_question:
                return f"""⚫ HAZARDOUS for children! (AQI: {aqi})

🚨 KEEP CHILDREN INDOORS:
• Do NOT let children go outside
• Close all windows and doors
• Use air purifiers on high
• Watch for breathing problems, coughing
• Seek medical help if symptoms appear

👶 This is a health emergency. Children must stay inside!"""
            return f"""⚫ HAZARDOUS - Stay Indoors! (AQI: {aqi})

{'DO NOT ' + specific_activity if specific_activity else 'Avoid ALL outdoor activities'}!

This is an emergency air quality situation. Only go outside for absolute essentials with proper N95/N99 protection.

🏥 Seek medical help if you experience symptoms."""
    
    def _generate_activity_answer(self, aqi: int, ctx: Dict, question: str) -> str:
        """Generate activity-specific answer"""
        if aqi <= 50:
            return f"""🏃 Perfect for All Activities! (AQI: {aqi})

Outdoor Activities - All Safe:
• Running & Jogging ✅
• Cycling & Biking ✅
• Team sports ✅
• Hiking & Nature walks ✅
• Children's outdoor play ✅

💪 Take advantage of this great air quality! No limitations today."""
        
        elif aqi <= 100:
            return f"""🚶 Most Activities OK (AQI: {aqi})

Recommended:
• Walking & Light jogging ✅
• Short cycling sessions ✅
• Outdoor yoga ✅
• Light gardening ✅

Consider Limiting:
• Marathon training ⚠️
• Intense sports (especially for sensitive groups) ⚠️

⏰ Best time: Early morning before traffic"""
        
        elif aqi <= 150:
            return f"""🏠 Switch to Indoor Activities (AQI: {aqi})

Safe Indoor Alternatives:
• Gym workouts 🏋️
• Indoor yoga/pilates 🧘
• Swimming (indoor pools) 🏊
• Home exercise videos 📱
• Mall walking 🚶

If you must be outside:
• Keep it brief (under 30 min)
• Low intensity only
• Wear a mask"""
        
        else:
            return f"""⛔ Avoid All Strenuous Activities (AQI: {aqi})

Today's Safe Options:
• Reading & relaxation 📚
• Board games & puzzles 🎲
• Streaming movies 📺
• Light stretching at home 🧘
• Indoor hobbies 🎨

⚠️ Even indoor exercise should be very light. Rest and stay hydrated.

🔄 Check AQI regularly - conditions may improve."""
    
    def _generate_health_answer(self, aqi: int, ctx: Dict, question: str) -> str:
        """Generate health-related answer"""
        question_lower = question.lower()
        
        # Check for specific conditions mentioned
        if 'asthma' in question_lower:
            severity = 'critical' if aqi > 100 else 'moderate' if aqi > 50 else 'low'
            return f"""🫁 Asthma & Air Quality (AQI: {aqi})

Current Risk Level: {'🔴 HIGH' if severity == 'critical' else '🟡 MODERATE' if severity == 'moderate' else '🟢 LOW'}

Recommendations:
{'• Stay indoors with air purifier running' if aqi > 100 else '• Monitor symptoms carefully'}
{'• Have rescue inhaler within reach at all times' if aqi > 50 else '• Enjoy outdoor activities normally'}
{'• Take preventive medication if prescribed' if aqi > 100 else ''}
{'• Consider wearing N95 mask outdoors' if aqi > 100 else ''}

⚠️ Triggers: PM2.5 and ozone are worst for asthma. {'Current levels are concerning.' if aqi > 100 else 'Current levels are manageable.'}"""
        
        if 'allergy' in question_lower or 'allergies' in question_lower:
            return f"""🤧 Allergies & Air Quality (AQI: {aqi})

Air pollution can worsen allergy symptoms!

Current Impact: {'High - pollution aggravates allergies' if aqi > 100 else 'Moderate' if aqi > 50 else 'Low'}

Tips:
• Take antihistamines as prescribed
• Use saline nasal rinse after outdoor exposure
• Keep windows closed on high pollution/pollen days
• Shower before bed to remove allergens
• Use HEPA air purifier in bedroom"""
        
        # General health impact answer
        return f"""🏥 Health Impact Summary (AQI: {aqi})

Current Air Quality: {ctx['category']}

Possible Symptoms at This Level:
{'✅ No symptoms expected - healthy air!' if aqi <= 50 else ''}
{'⚠️ Sensitive individuals may notice mild throat/eye irritation' if 50 < aqi <= 100 else ''}
{'🔸 Possible: coughing, throat irritation, reduced stamina' if 100 < aqi <= 150 else ''}
{'🔶 Likely: breathing difficulty, fatigue, chest tightness' if 150 < aqi <= 200 else ''}
{'🔴 Serious: respiratory distress, chest pain, severe coughing' if aqi > 200 else ''}

Who's Most Affected:
• Children (lungs still developing)
• Elderly (weaker immune systems)
• Pregnant women
• People with asthma, COPD, heart disease

💡 If symptoms persist, move to cleaner air and consult a doctor."""
    
    def _generate_protection_answer(self, aqi: int, ctx: Dict) -> str:
        """Generate protection advice"""
        return f"""😷 Protection Guide (AQI: {aqi})

Outdoor Protection:
{'✅ No special protection needed - enjoy fresh air!' if aqi <= 50 else ''}
{'😷 Optional: Surgical mask for extended outdoor time' if 50 < aqi <= 100 else ''}
{'😷 Recommended: N95/KN95 mask outdoors' if 100 < aqi <= 150 else ''}
{'😷 Required: N95/N99 mask for any outdoor exposure' if aqi > 150 else ''}

Indoor Protection:
• {'Open windows for ventilation' if aqi <= 50 else 'Keep windows closed'}
• {'Air purifier optional' if aqi <= 100 else 'Run HEPA air purifier (size it for your room)'}
• {'Standard cleaning routine' if aqi <= 100 else 'Change clothes after outdoor exposure'}

Health Precautions:
• Stay hydrated (helps flush toxins)
• Eat antioxidant-rich foods (vitamin C, E)
• {'No special medication needed' if aqi <= 100 else 'Have respiratory medications ready'}
• {'Monitor symptoms' if aqi > 100 else 'Enjoy your day!'}

Mask Effectiveness:
• N95: Filters 95% of particles
• N99: Filters 99% of particles
• Surgical: ~60-70% (better than nothing)
• Cloth: ~20-30% (not recommended for high AQI)"""
    
    def _generate_timing_answer(self, aqi: int) -> str:
        """Generate timing-related answer"""
        return """⏰ Best Times for Outdoor Activities

🌅 BEST Times (typically cleanest air):
• Early morning: 6:00 - 8:00 AM
• Late evening: After 8:00 PM
• After rainfall (rain clears particles)
• Weekends (less traffic)

⛔ WORST Times (highest pollution):
• Morning rush: 7:00 - 9:00 AM
• Evening rush: 5:00 - 8:00 PM
• Hot afternoons: 2:00 - 6:00 PM (ozone peaks)
• During wildfires/dust storms

💡 Pro Tips:
• Check real-time AQI before heading out
• Pollution patterns can change - monitor hourly
• Wind direction affects local air quality
• Rain is your friend for clean air!

📱 Set up AQI alerts on your phone for your exercise times."""
    
    def _generate_explanation_answer(self, question: str, relevant_kb: List[Dict]) -> str:
        """Generate explanatory answer from knowledge base"""
        if relevant_kb:
            # Use most relevant knowledge base entry
            best_match = relevant_kb[0]
            content = best_match['content'].strip()
            
            # Clean up the content formatting
            lines = content.split('\n')
            cleaned_lines = [line.strip() for line in lines if line.strip()]
            formatted_content = '\n'.join(cleaned_lines)
            
            return f"""📖 {best_match['topic']}

{formatted_content}"""
        
        # Default explanation about AQI
        return """📊 Air Quality Index (AQI) Explained

AQI is a standardized scale (0-500) measuring air pollution:

| AQI | Category | Health Impact |
|-----|----------|---------------|
| 0-50 | 🟢 Good | No health risk |
| 51-100 | 🟡 Moderate | Sensitive groups mild effects |
| 101-150 | 🟠 USG | Sensitive groups affected |
| 151-200 | 🔴 Unhealthy | Everyone may have effects |
| 201-300 | 🟣 Very Unhealthy | Health warnings for all |
| 301-500 | ⚫ Hazardous | Emergency conditions |

Main Pollutants Measured:
PM2.5, PM10, Ozone (O₃), NO₂, SO₂, CO

Ask me about any specific pollutant for more details!"""
    
    def _generate_recommendation_answer(self, aqi: int, ctx: Dict) -> str:
        """Generate personalized recommendations"""
        if aqi <= 50:
            activities = "outdoor running, cycling, hiking, picnics, sports, beach activities"
            dos = "Take advantage of great air quality, exercise outdoors, open windows"
            donts = "Nothing to avoid - enjoy your day!"
        elif aqi <= 100:
            activities = "light outdoor walks, gardening, short jogs, outdoor yoga"
            dos = "Stay hydrated, exercise in morning/evening, monitor how you feel"
            donts = "Avoid intense midday exercise, don't ignore symptoms"
        elif aqi <= 150:
            activities = "indoor gym, home workouts, swimming (indoor), yoga, mall walking"
            dos = "Stay indoors mostly, use air purifier, wear mask outside"
            donts = "Avoid prolonged outdoor exposure, no intense outdoor exercise"
        else:
            activities = "reading, streaming, indoor games, light stretching, rest"
            dos = "Stay indoors, run air purifier, keep windows sealed, stay hydrated"
            donts = "NO outdoor activities, avoid all exertion, don't open windows"
        
        return f"""📋 Today's Recommendations (AQI: {aqi})

{ctx['category']} Air Quality

✨ Suggested Activities:
{activities}

✅ Do:
• {dos.replace(', ', chr(10) + '• ')}

❌ Don't:
• {donts.replace(', ', chr(10) + '• ')}

{'🎉 Great day to be outdoors!' if aqi <= 50 else '😷 Take precautions!' if aqi <= 150 else '🏠 Indoor day - stay safe!'}"""
    
    def _generate_comparison_answer(self, question: str, aqi: int) -> str:
        """Generate comparison-based answer"""
        question_lower = question.lower()
        
        if 'indoor' in question_lower and 'outdoor' in question_lower:
            if aqi <= 50:
                return """🏠 vs 🌳 Indoor vs Outdoor (AQI: Good)

Both are great options today! Outdoor air is actually excellent.

Outdoor advantages today:
• Fresh, clean air
• Natural vitamin D
• Better for mood and energy

Recommendation: Go outside and enjoy! Save indoor workouts for bad air days."""
            else:
                return f"""🏠 vs 🌳 Indoor vs Outdoor (AQI: {aqi})

Winner today: Indoor activities 🏆

Why indoor is better right now:
• HEPA-filtered air is cleaner
• No exposure to outdoor pollutants
• Better for respiratory health

If you choose outdoor:
• Keep it brief
• Lower intensity
• Wear a mask
• Morning is best time"""
        
        return f"""📊 Comparison for AQI {aqi}

Based on current air quality ({aqi}):

{'✅ Outdoor activities win - great air today!' if aqi <= 50 else '⚖️ Balance indoor and outdoor, lean toward indoor for intense exercise' if aqi <= 100 else '🏠 Indoor activities strongly preferred'}

Ask about specific activities to compare!"""
    
    def _generate_forecast_answer(self, aqi: int, ctx: Dict) -> str:
        """Generate forecast-related answer"""
        return f"""🔮 Air Quality Patterns & Forecast

Current: AQI {aqi} ({ctx['category']})

Typical Daily Pattern:
• 🌅 6-8 AM: Usually cleanest (before traffic)
• 🌆 8-10 AM: Increases with rush hour
• ☀️ 12-4 PM: Often peaks (heat + traffic)
• 🌇 4-7 PM: High (evening rush)
• 🌙 8 PM+: Usually improves

What Affects Tomorrow:
• Weather (rain clears pollution)
• Wind (disperses pollutants)
• Temperature (heat creates ozone)
• Traffic patterns
• Industrial activity

💡 Best Practice: Check AQI first thing in the morning and before outdoor activities. Conditions can change quickly!

📱 Set up notifications for AQI alerts in your area."""
    
    def _generate_location_answer(self, ctx: Dict) -> str:
        """Generate location-specific answer"""
        return f"""📍 Air Quality in {ctx.get('location', 'Your Area')}

Current Status:
• AQI: {ctx.get('aqi', 'N/A')}
• Category: {ctx.get('category', 'Unknown')}
• Main Pollutant: {ctx.get('pollutant', 'Not specified')}

Local Factors That May Affect Air Quality:
• Traffic density and rush hours
• Industrial areas nearby
• Elevation and geography
• Coastal vs inland location
• Urban heat island effects

💡 Tips for Your Area:
• Learn your local pollution patterns
• Note which wind directions bring cleaner air
• Identify less-polluted routes for exercise
• Find parks away from major roads"""
    
    def _generate_general_answer(self, question: str, aqi: int, relevant_kb: List[Dict]) -> str:
        """Generate general answer for unclassified questions"""
        # Try to extract useful info from knowledge base
        if relevant_kb:
            best_match = relevant_kb[0]
            excerpt = best_match['content'][:500] + "..." if len(best_match['content']) > 500 else best_match['content']
            return f"""📍 Current AQI: {aqi}

Based on your question, here's relevant information:

{best_match['topic']}:
{excerpt.strip()}

---
💬 Ask me more specific questions about:
• Safety for activities
• Health impacts
• Protection measures
• Best times to go outside
• Specific pollutants"""
        
        # Fallback conversational response
        category = 'excellent' if aqi <= 50 else 'moderate' if aqi <= 100 else 'concerning' if aqi <= 150 else 'poor'
        return f"""🌍 Current Air Quality: AQI {aqi} ({category.title()})

I can help you understand:
• 🏃 Is it safe for outdoor activities?
• 🏥 How might this affect my health?
• 😷 What protection should I use?
• ⏰ When is the best time to go out?
• 📊 What do the pollutant levels mean?

Try asking questions like:
• "Is it safe to go jogging now?"
• "What's the best time for outdoor exercise?"
• "How does PM2.5 affect asthma?"
• "What mask should I wear?"

I'm here to help! 💬"""


# Keep alias for backward compatibility
QuestionAnsweringEngine = SmartQAEngine


# Singleton instances
_nlp_engine_instance = None
_qa_engine_instance = None


def get_nlp_engine(model_name: str = None, use_transformers: bool = True) -> HuggingFaceNLPEngine:
    """
    Get or create NLP engine instance
    
    Args:
        model_name: Optional custom model name
        use_transformers: Whether to use HuggingFace models (default: True)
                         Set to False for lightweight mode (VADER + TextBlob only)
    
    Returns:
        HuggingFaceNLPEngine instance
    """
    _ensure_nlp_resources()
    global _nlp_engine_instance
    if _nlp_engine_instance is None:
        _nlp_engine_instance = HuggingFaceNLPEngine(
            model_name=model_name, 
            use_transformers=use_transformers
        )
    return _nlp_engine_instance


def get_nlp_engine_lightweight() -> HuggingFaceNLPEngine:
    """Get NLP engine in lightweight mode (no HuggingFace models)"""
    _ensure_nlp_resources()
    return HuggingFaceNLPEngine(use_transformers=False)


def get_qa_engine() -> QuestionAnsweringEngine:
    """Get or create QA engine instance"""
    _ensure_nlp_resources()
    global _qa_engine_instance
    if _qa_engine_instance is None:
        _qa_engine_instance = QuestionAnsweringEngine()
    return _qa_engine_instance


class UniversalQueryHandler:
    """
    Universal query handler that responds to EVERY user query
    Now powered by Advanced Conversational AI with Transformers
    Provides human-like, context-aware responses
    """
    
    def __init__(self):
        """Initialize universal query handler with Conversational AI"""
        # Import the new conversational AI
        try:
            from models.conversational_ai import get_conversational_ai
            self.conversational_ai = get_conversational_ai()
            self.use_advanced_ai = True
            logger.info("✓ Advanced Conversational AI loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load Conversational AI: {e}. Using fallback.")
            self.conversational_ai = None
            self.use_advanced_ai = False
        
        # Keep original engines as fallback
        self.nlp_engine = get_nlp_engine()
        self.qa_engine = get_qa_engine()
        
        # Try to load Kaggle dataset manager
        self.kaggle_manager = None
        self.kaggle_db = None
        if KAGGLE_AVAILABLE:
            try:
                # Ensure data directory exists
                import os
                os.makedirs("data/kaggle", exist_ok=True)
                
                self.kaggle_manager = KaggleAQIDataset()
                import pickle
                from pathlib import Path
                cache_file = Path("data/kaggle/recommendations_db.pkl")
                if cache_file.exists():
                    with open(cache_file, 'rb') as f:
                        self.kaggle_db = pickle.load(f)
                else:
                    logger.info("Building Kaggle recommendations database (first run)...")
                    self.kaggle_db = self.kaggle_manager.build_recommendations_database()
                logger.info("✓ Universal query handler with Kaggle data initialized")
            except ImportError:
                logger.warning("Kaggle dataset module not available (optional feature)")
            except Exception as e:
                logger.warning(f"Kaggle integration unavailable: {e}")
    
    def handle_query(self, query: str, aqi_data: Dict = None) -> Dict:
        """
        Handle ANY user query with Advanced AI - provides human-like responses
        
        Args:
            query: User's question/statement (anything!)
            aqi_data: Current AQI context {aqi, pollutants, location, etc.}
            
        Returns:
            Comprehensive response with answer, confidence, intent, and more
        """
        if not query or not query.strip():
            return {
                'answer': "I didn't catch that. Could you ask me about air quality, health impacts, or activities?",
                'confidence': 1.0,
                'type': 'empty_query'
            }
        
        # Initialize default AQI data if not provided
        if aqi_data is None:
            aqi_data = {'aqi': 100, 'pollutants': {}, 'location': 'your area'}
        
        # USE ADVANCED CONVERSATIONAL AI if available
        if self.use_advanced_ai and self.conversational_ai:
            try:
                logger.info(f"Processing query with Conversational AI: '{query[:50]}...'")
                response = self.conversational_ai.chat(query, aqi_data)
                
                # Enhance with Kaggle data if available
                if self.kaggle_db and aqi_data.get('aqi'):
                    aqi = aqi_data.get('aqi')
                    kaggle_recs = self._get_kaggle_enhancement(aqi, aqi_data.get('pollutants'))
                    if kaggle_recs:
                        response['kaggle_data'] = kaggle_recs
                
                return response
            except Exception as e:
                logger.error(f"Conversational AI error: {e}. Falling back to basic handler.")
                import traceback
                traceback.print_exc()
        
        # FALLBACK: Use original Q&A engine
        return self._handle_query_fallback(query, aqi_data)
    
    def _handle_query_fallback(self, query: str, aqi_data: Dict) -> Dict:
        """Fallback query handler when AI is unavailable"""
        aqi = aqi_data.get('aqi', 100)
        
        # Try Q&A engine
        try:
            qa_result = self.qa_engine.answer_question(query, aqi_data)
            
            # Enhance with Kaggle data if available
            if self.kaggle_db and aqi:
                kaggle_recs = self._get_kaggle_enhancement(aqi, aqi_data.get('pollutants'))
                if kaggle_recs:
                    qa_result['kaggle_data'] = kaggle_recs
                    qa_result['answer'] += f"\\n\\n📊 Data-Driven Insight:\\n{self._format_kaggle_insight(kaggle_recs, aqi)}"
            
            return qa_result
            
        except Exception as e:
            logger.warning(f"Q&A engine error: {e}")
        
        # Ultimate fallback: Use NLP analysis and generate contextual response
        return self._generate_fallback_response(query, aqi_data)
    
    def _handle_conversation(self, query: str, aqi: float) -> Optional[Dict]:
        """Handle conversational/social queries"""
        # Greetings
        if any(greet in query for greet in self.conversation_patterns['greeting']):
            category = self._get_aqi_category(aqi)
            return {
                'answer': f"👋 Hello! Today's air quality is {category} (AQI: {int(aqi)}). How can I help you stay safe and healthy?",
                'confidence': 1.0,
                'type': 'greeting',
                'suggestions': [
                    'Is it safe to exercise outside?',
                    'What activities do you recommend?',
                    'Tell me about the current air quality'
                ]
            }
        
        # Thanks
        if any(thank in query for thank in self.conversation_patterns['thanks']):
            return {
                'answer': "😊 You're welcome! Stay safe and breathe easy! Feel free to ask anything else about air quality.",
                'confidence': 1.0,
                'type': 'acknowledgment'
            }
        
        # Help requests
        if any(help_word in query for help_word in self.conversation_patterns['help']):
            return {
                'answer': """🤝 I can help you with:

1. Safety Questions - "Is it safe to run outside?"
2. Activity Recommendations - "What can I do today?"
3. Health Information - "How does PM2.5 affect health?"
4. Best Times - "When is the best time to exercise?"
5. Protection Advice - "What mask should I wear?"
6. Pollutant Explanations - "What is ozone?"
7. Location-specific Info - "How's the air quality here?"

Just ask naturally - I understand all kinds of questions! 💬""",
                'confidence': 1.0,
                'type': 'help'
            }
        
        return None
    
    def _get_kaggle_enhancement(self, aqi: float, pollutants: Dict = None) -> Optional[Dict]:
        """Get Kaggle dataset enhancement"""
        if not self.kaggle_manager:
            return None
        
        try:
            return self.kaggle_manager.get_recommendations_for_aqi(aqi, pollutants)
        except:
            return None
    
    def _format_kaggle_insight(self, kaggle_data: Dict, aqi: float) -> str:
        """Format Kaggle insights for display"""
        insights = []
        
        # Category info
        if 'category_info' in kaggle_data:
            cat_info = kaggle_data['category_info']
            if 'frequency' in cat_info:
                insights.append(f"This AQI level occurs {cat_info['frequency']*100:.1f}% of the time globally")
        
        # Pollutant alerts
        if 'pollutant_alerts' in kaggle_data and kaggle_data['pollutant_alerts']:
            top_alert = kaggle_data['pollutant_alerts'][0]
            insights.append(f"{top_alert['pollutant']} is {top_alert['level']:.1f} µg/m³ (safe level: {top_alert['safe_level']} µg/m³)")
        
        # Health statistics
        if 'health_statistics' in self.kaggle_db:
            stats = self.kaggle_db['health_statistics']
            if aqi > 150:
                insights.append(stats.get('global_summary', ''))
        
        return " • ".join(insights) if insights else "Based on global air quality patterns"
    
    def _generate_fallback_response(self, query: str, aqi_data: Dict) -> Dict:
        """Generate intelligent fallback response for any query"""
        aqi = aqi_data.get('aqi', 100)
        category = self._get_aqi_category(aqi)
        
        # Analyze query with NLP
        analysis = self.nlp_engine.analyze_with_ai(query, aqi_context=aqi_data)
        
        # Extract intent
        intent = analysis.get('intent', {}).get('intent', 'unknown')
        sentiment = analysis.get('sentiment', {}).get('label', 'NEUTRAL')
        
        # Build contextual response
        response = f"📍 Current AQI: {int(aqi)} ({category})\\n\\n"
        
        # Address the query contextually
        if sentiment == 'NEGATIVE' or any(word in query.lower() for word in ['worry', 'concern', 'scared', 'afraid']):
            if aqi <= 100:
                response += "Don't worry! Air quality is acceptable for most activities. "
            else:
                response += "I understand your concern. Here's what you should know: "
        
        # Add relevant information based on AQI
        if aqi <= 50:
            response += "✅ Excellent air quality! All outdoor activities are safe. Perfect day to enjoy fresh air!"
        elif aqi <= 100:
            response += "🟡 Moderate air quality. Most people can proceed with normal activities. Sensitive individuals should be cautious during prolonged outdoor exertion."
        elif aqi <= 150:
            response += "🟠 Unhealthy for sensitive groups. Consider reducing prolonged outdoor activities, especially if you have respiratory conditions."
        elif aqi <= 200:
            response += "🔴 Unhealthy air. Everyone should limit outdoor activities. Wear a mask if you must go outside."
        else:
            response += "⚫ Hazardous conditions! Stay indoors, use air purifiers, and avoid all outdoor activities."
        
        # Add specific advice based on keywords in query
        query_lower = query.lower()
        if any(word in query_lower for word in ['exercise', 'run', 'jog', 'workout', 'gym']):
            response += f"\\n\\n🏃 Exercise Advice: "
            if aqi <= 100:
                response += "Safe for outdoor exercise. Stay hydrated!"
            else:
                response += "Consider indoor exercise today. Outdoor workout not recommended."
        
        if any(word in query_lower for word in ['child', 'kids', 'baby', 'toddler']):
            response += f"\\n\\n👶 For Children: "
            if aqi <= 75:
                response += "Safe for children's outdoor play."
            else:
                response += "Limit children's outdoor time. Keep activities indoors."
        
        if any(word in query_lower for word in ['mask', 'protection', 'protect']):
            response += f"\\n\\n😷 Protection: "
            if aqi > 150:
                response += "N95/N99 masks recommended for any outdoor exposure."
            elif aqi > 100:
                response += "Consider wearing a mask during prolonged outdoor activities."
            else:
                response += "Mask not necessary at current levels."
        
        # Add helpful suggestions
        response += "\\n\\n💡 Ask me about: Safety for activities, health impacts, best times to go out, or specific pollutants."
        
        return {
            'answer': response,
            'confidence': 0.75,
            'type': 'contextual_fallback',
            'intent': intent,
            'sentiment': sentiment,
            'keywords': analysis.get('keywords', [])
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


# Singleton for universal query handler
_universal_handler_instance = None

def get_universal_query_handler() -> UniversalQueryHandler:
    """Get or create universal query handler instance"""
    _ensure_nlp_resources()
    global _universal_handler_instance
    if _universal_handler_instance is None:
        _universal_handler_instance = UniversalQueryHandler()
    return _universal_handler_instance
