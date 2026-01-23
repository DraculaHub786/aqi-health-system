# Auto-download NLTK & TextBlob data on startup (critical for Colab)
import nltk
import ssl

# Handle SSL certificate issue (common in some environments)
try:
    _create_unverified_https_context = ssl._create_unverified_context
    ssl._create_default_https_context = _create_unverified_https_context
except AttributeError:
    pass

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4', quiet=True)

# Download TextBlob corpora
try:
    from textblob import TextBlob
    TextBlob("test").correct()  # Will fail if corpora not present
except Exception:
    try:
        import subprocess
        subprocess.run(['python', '-m', 'textblob.download_corpora'], 
                      stdout=subprocess.DEVNULL, 
                      stderr=subprocess.DEVNULL,
                      timeout=30)
    except:
        pass

import streamlit as st
import logging
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import sys
import os

# Add paths for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.advanced_ml import get_predictor, get_classifier, get_recommender
from models.nlp_engine import get_nlp_engine, get_qa_engine, get_universal_query_handler
from utils.data_manager import get_data_manager
from utils.config import APP_CONFIG, AQI_CATEGORIES, POLLUTANT_INFO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="AQI Health & Activity Planner",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI with animations
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .aqi-card {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
    }
    .stAlert {
        border-radius: 10px;
    }
    .tip-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
        border-radius: 4px;
    }
    
    /* Sliding text animation for long text */
    .sliding-text-container {
        overflow: hidden;
        white-space: nowrap;
        position: relative;
        background: linear-gradient(90deg, transparent, #f0f2f6 5%, #f0f2f6 95%, transparent);
        border-radius: 8px;
        padding: 8px 12px;
        margin: 4px 0;
    }
    
    .sliding-text {
        display: inline-block;
        animation: slideText 8s linear infinite;
        padding-right: 50px;
    }
    
    .sliding-text:hover {
        animation-play-state: paused;
    }
    
    @keyframes slideText {
        0% { transform: translateX(0%); }
        100% { transform: translateX(-50%); }
    }
    
    /* Static text that fits */
    .static-text {
        display: block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 10px 15px;
        border-radius: 8px;
        text-align: center;
        font-weight: 600;
        font-size: 1.1em;
    }
    
    /* Risk card styling */
    .risk-card {
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .risk-card h2 {
        margin: 0 0 10px 0;
        font-size: 2em;
    }
    
    .risk-card p {
        margin: 5px 0;
        font-size: 1.1em;
    }
    
    .risk-low { background: linear-gradient(135deg, #d4edda 0%, #a8e6cf 100%); border-left: 5px solid #28a745; }
    .risk-moderate { background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%); border-left: 5px solid #ffc107; }
    .risk-high { background: linear-gradient(135deg, #ffe0b2 0%, #ffcc80 100%); border-left: 5px solid #ff9800; }
    .risk-very-high { background: linear-gradient(135deg, #ffcdd2 0%, #ef9a9a 100%); border-left: 5px solid #f44336; }
    .risk-severe { background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%); border-left: 5px solid #9c27b0; }
    
    /* Forecast card */
    .forecast-simple {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #1f77b4;
    }
    
    .forecast-simple h4 {
        color: #1f77b4;
        margin: 0 0 10px 0;
    }
    
    /* Activity card with animation */
    .activity-card {
        padding: 12px 15px;
        border-radius: 10px;
        margin: 8px 0;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .activity-card:hover {
        transform: translateX(5px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .safe-activity { background: linear-gradient(90deg, #d4edda, #e8f5e9); border-left: 4px solid #28a745; }
    .caution-activity { background: linear-gradient(90deg, #fff3cd, #fffde7); border-left: 4px solid #ffc107; }
    .unsafe-activity { background: linear-gradient(90deg, #f8d7da, #ffebee); border-left: 4px solid #dc3545; }
</style>
""", unsafe_allow_html=True)


def get_aqi_color(aqi: float) -> str:
    """Get color based on AQI value"""
    if aqi <= 50:
        return "#00e400"
    elif aqi <= 100:
        return "#ffff00"
    elif aqi <= 150:
        return "#ff7e00"
    elif aqi <= 200:
        return "#ff0000"
    elif aqi <= 300:
        return "#8f3f97"
    else:
        return "#7e0023"


def get_aqi_category(aqi: float) -> str:
    """Get AQI category"""
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


def create_aqi_gauge(aqi: float) -> go.Figure:
    """Create AQI gauge chart"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=aqi,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Current AQI", 'font': {'size': 24}},
        delta={'reference': 50, 'increasing': {'color': "red"}},
        gauge={
            'axis': {'range': [None, 500], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': get_aqi_color(aqi)},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#e8f5e9'},
                {'range': [50, 100], 'color': '#fff9c4'},
                {'range': [100, 150], 'color': '#ffe0b2'},
                {'range': [150, 200], 'color': '#ffcdd2'},
                {'range': [200, 300], 'color': '#f3e5f5'},
                {'range': [300, 500], 'color': '#ffebee'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 150
            }
        }
    ))
    
    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=60, b=20),
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig


def create_pollutant_chart(pollutants: dict) -> go.Figure:
    """Create pollutant bar chart"""
    pollutant_names = list(pollutants.keys())
    values = list(pollutants.values())
    
    colors = [get_aqi_color(v) for v in values]
    
    fig = go.Figure(data=[
        go.Bar(
            x=pollutant_names,
            y=values,
            marker_color=colors,
            text=values,
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Level: %{y:.1f} µg/m³<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title="Pollutant Levels",
        xaxis_title="Pollutant",
        yaxis_title="Concentration (µg/m³)",
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False
    )
    
    return fig


def create_forecast_chart(predictions: list) -> go.Figure:
    """Create hourly AQI forecast chart"""
    hours = [p['hour'] for p in predictions]
    aqi_values = [p['aqi'] for p in predictions]
    lower_bounds = [p['confidence_lower'] for p in predictions]
    upper_bounds = [p['confidence_upper'] for p in predictions]
    
    fig = go.Figure()
    
    # Confidence interval
    fig.add_trace(go.Scatter(
        x=hours + hours[::-1],
        y=upper_bounds + lower_bounds[::-1],
        fill='toself',
        fillcolor='rgba(31, 119, 180, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Confidence Interval',
        hoverinfo='skip'
    ))
    
    # AQI prediction line
    fig.add_trace(go.Scatter(
        x=hours,
        y=aqi_values,
        mode='lines+markers',
        name='Predicted AQI',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8),
        hovertemplate='<b>%{x}</b><br>AQI: %{y}<extra></extra>'
    ))
    
    fig.update_layout(
        title="24-Hour AQI Forecast",
        xaxis_title="Time",
        yaxis_title="AQI",
        height=400,
        hovermode='x unified',
        margin=dict(l=20, r=20, t=40, b=60)
    )
    
    return fig


def create_historical_chart(historical_data: list) -> go.Figure:
    """Create historical trend chart"""
    if not historical_data:
        return None
        
    df = pd.DataFrame(historical_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('AQI Trend', 'PM2.5 & PM10 Levels'),
        vertical_spacing=0.15,
        row_heights=[0.6, 0.4]
    )
    
    # AQI trend
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['aqi'],
            mode='lines',
            name='AQI',
            line=dict(color='#1f77b4', width=2),
            fill='tozeroy',
            fillcolor='rgba(31, 119, 180, 0.3)'
        ),
        row=1, col=1
    )
    
    # PM2.5 and PM10
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['pm25'],
            mode='lines',
            name='PM2.5',
            line=dict(color='#ff7f0e', width=2)
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['pm10'],
            mode='lines',
            name='PM10',
            line=dict(color='#2ca02c', width=2)
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=600,
        hovermode='x unified',
        showlegend=True,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="AQI", row=1, col=1)
    fig.update_yaxes(title_text="µg/m³", row=2, col=1)
    
    return fig


def render_sidebar():
    """Render sidebar with user inputs"""
    st.sidebar.title("🌍 AQI Health Planner")
    st.sidebar.markdown("---")
    
    # Location input
    prev_location = st.session_state.get('location', 'Delhi')
    location = st.sidebar.text_input(
        "📍 Enter Location",
        value=prev_location,
        help="Enter city name or coordinates",
        key="location_input"
    )
    # Detect location change
    location_changed = location != prev_location
    
    # User profile
    st.sidebar.markdown("### 👤 User Profile")
    
    age = st.sidebar.slider("Age", 1, 100, st.session_state.get('age', 30))
    
    profile_type = st.sidebar.selectbox(
        "Category",
        ["General", "Children", "Elderly", "Outdoor Worker"],
        index=0
    )
    
    # Health conditions
    st.sidebar.markdown("### 🏥 Health Conditions")
    respiratory = st.sidebar.checkbox(
        "Respiratory condition (Asthma/COPD)",
        value=st.session_state.get('respiratory', False)
    )
    heart = st.sidebar.checkbox(
        "Heart condition",
        value=st.session_state.get('heart', False)
    )
    
    # Activity preferences
    st.sidebar.markdown("### 🏃 Activity Preferences")
    intensity = st.sidebar.select_slider(
        "Preferred Intensity",
        options=["Sedentary", "Low", "Medium", "High"],
        value="Medium"
    )
    
    outdoor_hours = st.sidebar.slider(
        "Daily Outdoor Hours",
        0, 12,
        st.session_state.get('outdoor_hours', 2)
    )
    
    outdoor_worker = st.sidebar.checkbox(
        "Outdoor Worker",
        value=st.session_state.get('outdoor_worker', False)
    )
    
    # Fetch button
    fetch_data = st.sidebar.button("🔄 Fetch AQI Data", type="primary", use_container_width=True)
    # If location changed, set a session flag
    if location_changed:
        st.session_state.fetch_on_location_change = True
    
    # Settings
    st.sidebar.markdown("---")
    with st.sidebar.expander("⚙️ Advanced Settings"):
        force_refresh = st.checkbox("Force refresh (skip cache)", value=False)
        show_debug = st.checkbox("Show debug info", value=False)
        
    return {
        'location': location,
        'age': age,
        'profile_type': profile_type.lower(),
        'respiratory_condition': respiratory,
        'heart_condition': heart,
        'preferred_intensity': intensity.lower(),
        'daily_outdoor_hours': outdoor_hours,
        'outdoor_worker': outdoor_worker,
        'fetch_data': fetch_data,
        'force_refresh': force_refresh,
        'show_debug': show_debug,
        'location_changed': location_changed
    }


def main():
    """Main application"""
    
    # Initialize session state
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.location = 'Delhi'
        st.session_state.aqi_data = None
        st.session_state.age = 30
        st.session_state.respiratory = False
        st.session_state.heart = False
        st.session_state.outdoor_hours = 2
        st.session_state.outdoor_worker = False
        
    # Header
    st.markdown('<div class="main-header">🌍 AI-Powered AQI Health & Activity Planner</div>', unsafe_allow_html=True)
    st.markdown("#### Real-time air quality monitoring with ML predictions and NLP insights")
    
    # Render sidebar and get inputs
    user_input = render_sidebar()
    
    # Initialize services
    with st.spinner('Initializing AI models...'):
        data_manager = get_data_manager()
        nlp_engine = get_nlp_engine()
        predictor = get_predictor()
        classifier = get_classifier()
        recommender = get_recommender()
    
    # Fetch data on button click, first load, or location change
    fetch_on_location_change = st.session_state.pop('fetch_on_location_change', False)
    if user_input['fetch_data'] or st.session_state.aqi_data is None or fetch_on_location_change:
        with st.spinner(f"Fetching AQI data for {user_input['location']}..."):
            try:
                aqi_data = data_manager.get_aqi_data(
                    user_input['location'],
                    force_refresh=user_input['force_refresh']
                )
                
                if aqi_data:
                    st.session_state.aqi_data = aqi_data
                    st.session_state.location = user_input['location']
                    st.success(f"✅ Data fetched from {aqi_data['source']}")
                else:
                    st.error("❌ Unable to fetch AQI data. Please try another location.")
                    return
                    
            except Exception as e:
                st.error(f"❌ Error fetching data: {str(e)}")
                logger.error(f"Data fetch error: {e}", exc_info=True)
                return
    
    # Display data if available
    if st.session_state.aqi_data:
        aqi_data = st.session_state.aqi_data
        aqi = aqi_data['aqi']
        pollutants = aqi_data['pollutants']
        
        # Main metrics row
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        
        with col1:
            st.plotly_chart(create_aqi_gauge(aqi), use_container_width=True)
            
        with col2:
            # Category with sliding animation for long text
            category = get_aqi_category(aqi)
            location_name = aqi_data.get('location', user_input['location'])
            
            st.markdown("📊 Category")
            if len(category) > 15:
                st.markdown(f'''
                <div class="sliding-text-container">
                    <span class="sliding-text">{category} &nbsp;&nbsp;&nbsp; {category}</span>
                </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="static-text">{category}</div>', unsafe_allow_html=True)
            
            st.markdown("📍 Location")
            if len(location_name) > 12:
                st.markdown(f'''
                <div class="sliding-text-container">
                    <span class="sliding-text">{location_name} &nbsp;&nbsp;&nbsp; {location_name}</span>
                </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="static-text">{location_name}</div>', unsafe_allow_html=True)
            
        with col3:
            st.metric(
                "Dominant Pollutant",
                aqi_data.get('dominant_pollutant', 'PM2.5')
            )
            st.metric(
                "Data Source",
                aqi_data.get('source', 'API')
            )
            
        with col4:
            timestamp = datetime.fromisoformat(aqi_data['timestamp'])
            st.metric(
                "Last Updated",
                timestamp.strftime("%I:%M %p")
            )
            st.metric(
                "Date",
                timestamp.strftime("%b %d, %Y")
            )
        
        st.markdown("---")
        
        # Health explanation (NLP generated) - DYNAMIC based on location & AQI
        st.markdown(f"### 🤖 AI Health Analysis for {user_input['location']}")
        with st.spinner("Generating AI explanation..."):
            explanation = nlp_engine.generate_health_explanation(
                aqi,
                pollutants,
                get_aqi_category(aqi),
                user_input['profile_type'],
                user_input['location']
            )
            
            # Dynamic color based on AQI
            if aqi <= 50:
                st.success(explanation)
            elif aqi <= 100:
                st.info(explanation)
            elif aqi <= 150:
                st.warning(explanation)
            else:
                st.error(explanation)
            
            # Add real-time context
            current_hour = datetime.now().hour
            if 6 <= current_hour <= 9:
                time_advice = "🌅 Morning Update: Air quality typically improves in early morning. Good time for outdoor activities if AQI permits."
            elif 12 <= current_hour <= 16:
                time_advice = "☀️ Afternoon Alert: Ozone levels peak during afternoon hours. Consider indoor activities."
            elif 17 <= current_hour <= 20:
                time_advice = "🌆 Evening Advisory: Rush hour traffic may worsen air quality temporarily."
            else:
                time_advice = "🌙 Night Note: Air quality typically stabilizes during night hours."
            
            st.caption(time_advice)
        
        # Two column layout
        col_left, col_right = st.columns([1, 1])
        
        with col_left:
            # Pollutant chart
            st.plotly_chart(create_pollutant_chart(pollutants), use_container_width=True)
            
            # Risk classification - USER FRIENDLY VERSION
            st.markdown("### 🎯 Your Personal Health Risk")
            with st.spinner("Analyzing health risk..."):
                user_profile = {
                    'age': user_input['age'],
                    'respiratory_condition': user_input['respiratory_condition'],
                    'heart_condition': user_input['heart_condition'],
                    'daily_outdoor_hours': user_input['daily_outdoor_hours'],
                    'outdoor_worker': user_input['outdoor_worker'],
                    'pm25': pollutants.get('pm25', aqi * 0.6)
                }
                
                risk_result = classifier.classify_risk(aqi, user_profile)
                
                # Display risk level
                risk_level = risk_result['risk_level']
                risk_score = risk_result['risk_score']
                
                # User-friendly risk explanations
                risk_explanations = {
                    'Low': {
                        'emoji': '😊',
                        'class': 'risk-low',
                        'message': "You're in the clear!",
                        'action': "Enjoy outdoor activities freely. No special precautions needed.",
                        'icon': '✅'
                    },
                    'Moderate': {
                        'emoji': '😐',
                        'class': 'risk-moderate', 
                        'message': "Some caution advised",
                        'action': "Most activities are safe. Take breaks if you feel uncomfortable.",
                        'icon': '⚠️'
                    },
                    'High': {
                        'emoji': '😟',
                        'class': 'risk-high',
                        'message': "Be careful today",
                        'action': "Limit outdoor time to 1-2 hours. Wear a mask if going out.",
                        'icon': '🟠'
                    },
                    'Very High': {
                        'emoji': '😰',
                        'class': 'risk-very-high',
                        'message': "Stay indoors if possible",
                        'action': "Avoid outdoor activities. Use air purifiers indoors.",
                        'icon': '🔴'
                    },
                    'Severe': {
                        'emoji': '🚨',
                        'class': 'risk-severe',
                        'message': "Health emergency!",
                        'action': "Do NOT go outside. Seal windows. Seek medical help if symptoms appear.",
                        'icon': '⛔'
                    }
                }
                
                risk_info = risk_explanations.get(risk_level, risk_explanations['Moderate'])
                
                # Beautiful risk card
                st.markdown(f'''
                <div class="risk-card {risk_info['class']}">
                    <h2>{risk_info['emoji']} {risk_level} Risk</h2>
                    <p style="font-size: 1.3em; font-weight: bold;">{risk_info['message']}</p>
                    <p>{risk_info['icon']} {risk_info['action']}</p>
                </div>
                ''', unsafe_allow_html=True)
                
                # Simple visual meter
                meter_color = '#28a745' if risk_score < 30 else '#ffc107' if risk_score < 60 else '#dc3545'
                st.markdown(f'''
                <div style="background: #e9ecef; border-radius: 10px; height: 20px; margin: 10px 0;">
                    <div style="background: {meter_color}; width: {risk_score}%; height: 100%; border-radius: 10px; transition: width 0.5s;"></div>
                </div>
                <p style="text-align: center; color: gray;">Risk Score: {risk_score}/100</p>
                ''', unsafe_allow_html=True)
                
                # Personalized factors (expandable)
                with st.expander("📋 Why this risk level? (Click to see factors)"):
                    factors = []
                    if user_input['age'] < 12:
                        factors.append("👶 Children are more sensitive to air pollution")
                    elif user_input['age'] > 60:
                        factors.append("👴 Seniors have higher vulnerability")
                    if user_input['respiratory_condition']:
                        factors.append("🫁 Respiratory condition increases sensitivity")
                    if user_input['heart_condition']:
                        factors.append("❤️ Heart condition requires extra caution")
                    if user_input['outdoor_worker']:
                        factors.append("👷 Outdoor work means prolonged exposure")
                    if aqi > 100:
                        factors.append(f"🌫️ High AQI ({int(aqi)}) is the main concern")
                    if pollutants.get('pm25', 0) > 50:
                        factors.append(f"💨 PM2.5 is elevated ({pollutants.get('pm25', 0):.0f} µg/m³)")
                    
                    if factors:
                        for f in factors:
                            st.markdown(f"- {f}")
                    else:
                        st.markdown("✅ No major risk factors detected!")
        
        with col_right:
            # Activity recommendations - DYNAMIC based on AQI
            st.markdown(f"### 🏃 What Can You Do Right Now?")
            
            # Quick summary based on AQI
            if aqi <= 50:
                st.success("🎉 Perfect day for any activity! Air is clean and fresh.")
            elif aqi <= 100:
                st.info("👍 Good conditions for most outdoor activities.")
            elif aqi <= 150:
                st.warning("⚠️ Limit strenuous outdoor activities. Indoor options recommended.")
            else:
                st.error("🏠 Stay indoors. Only essential outdoor activities.")
            
            with st.spinner("Finding best activities for you..."):
                recommendations = recommender.get_recommendations(
                    aqi,
                    user_profile,
                    weather=aqi_data.get('weather'),
                    current_time=datetime.now()
                )
                
                if recommendations:
                    for rec in recommendations[:6]:
                        activity = rec['activity']
                        safety = rec['safety_level']
                        score = rec['score']
                        rec_text = rec['recommendation']
                        
                        # Dynamic recommendation text based on AQI
                        if safety == 'Safe':
                            safety_class = 'safe-activity'
                            icon = "✅"
                            status = "Go for it!"
                        elif safety == 'Caution':
                            safety_class = 'caution-activity'
                            icon = "⚠️"
                            status = "Be careful"
                        else:
                            safety_class = 'unsafe-activity'
                            icon = "❌"
                            status = "Not today"
                        
                        st.markdown(f'''
                        <div class="activity-card {safety_class}">
                            <strong>{icon} {activity}</strong> <span style="float:right; font-size:0.9em;">{status}</span><br>
                            <small style="color: #666;">{rec_text}</small>
                        </div>
                        ''', unsafe_allow_html=True)
                else:
                    st.warning("No suitable activities found. Consider indoor rest.")
        
        st.markdown("---")
        
        # Hourly forecast - USER FRIENDLY VERSION
        st.markdown("### 🔮 When Will Air Be Better?")
        
        # Simple summary first
        st.markdown(f'''
        <div class="forecast-simple">
            <h4>📊 Quick Look at Next 24 Hours</h4>
            <p>We use AI to predict how air quality will change throughout the day.</p>
        </div>
        ''', unsafe_allow_html=True)
        
        with st.spinner("Predicting air quality..."):
            predictions = predictor.predict_hourly(
                aqi,
                pollutants,
                hours_ahead=24,
                weather_data=aqi_data.get('weather')
            )
            
            # User-friendly forecast summary
            safe_times = [p for p in predictions if p['aqi'] < 100]
            moderate_times = [p for p in predictions if 100 <= p['aqi'] < 150]
            bad_times = [p for p in predictions if p['aqi'] >= 150]
            
            col_f1, col_f2, col_f3 = st.columns(3)
            with col_f1:
                st.markdown(f'''
                <div style="background: #d4edda; padding: 15px; border-radius: 10px; text-align: center;">
                    <h3 style="color: #28a745; margin: 0;">✅ {len(safe_times)}</h3>
                    <p style="margin: 5px 0;">Good Hours</p>
                    <small>Safe for outdoor</small>
                </div>
                ''', unsafe_allow_html=True)
            with col_f2:
                st.markdown(f'''
                <div style="background: #fff3cd; padding: 15px; border-radius: 10px; text-align: center;">
                    <h3 style="color: #ffc107; margin: 0;">⚠️ {len(moderate_times)}</h3>
                    <p style="margin: 5px 0;">Moderate Hours</p>
                    <small>Be cautious</small>
                </div>
                ''', unsafe_allow_html=True)
            with col_f3:
                st.markdown(f'''
                <div style="background: #f8d7da; padding: 15px; border-radius: 10px; text-align: center;">
                    <h3 style="color: #dc3545; margin: 0;">❌ {len(bad_times)}</h3>
                    <p style="margin: 5px 0;">Unhealthy Hours</p>
                    <small>Stay indoors</small>
                </div>
                ''', unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Show the chart
            st.plotly_chart(create_forecast_chart(predictions), use_container_width=True)
            
            # Best times with simple language
            if safe_times:
                best_hours = [p['hour'] for p in safe_times[:3]]
                st.success(f"🌟 Best times to go outside: {', '.join(best_hours)}")
                if len(safe_times) >= 3:
                    st.caption("💡 Plan your outdoor activities during these windows for healthiest experience.")
            elif moderate_times:
                ok_hours = [p['hour'] for p in moderate_times[:3]]
                st.warning(f"⚠️ Acceptable times (use caution): {', '.join(ok_hours)}")
                st.caption("💡 Keep outdoor activities short and take breaks.")
            else:
                st.error("🏠 No good outdoor times tomorrow. Plan indoor activities instead.")
                st.caption("💡 Use this time for indoor exercise, reading, or relaxation.")
        
        # Health tips - DYNAMIC based on AQI
        st.markdown(f"### 💡 Your Health Tips for AQI {int(aqi)}")
        
        # Dynamic intro based on conditions
        if aqi <= 50:
            st.success("🎉 Great air today! Here are some tips to make the most of it:")
        elif aqi <= 100:
            st.info("👍 Decent air quality. Some helpful tips for you:")
        elif aqi <= 150:
            st.warning("⚠️ Air isn't great today. Important tips to protect yourself:")
        else:
            st.error("🚨 Poor air quality! Follow these tips carefully:")
        
        # ALWAYS show health tips (no empty sections!)
        tips = None
        try:
            tips = nlp_engine.generate_personalized_tips(
                aqi,
                pollutants,
                user_profile,
                user_input['location']
            )
        except Exception as e:
            logger.error(f"Health tips error: {e}")
        
        # Display tips or fallback
        if tips and len(tips) > 0:
            tip_cols = st.columns(2)
            for i, tip in enumerate(tips):
                with tip_cols[i % 2]:
                    if aqi > 150 and i < 2:
                        st.error(f"🔴 {tip}")
                    elif aqi > 100 and i < 3:
                        st.warning(f"🟡 {tip}")
                    else:
                        st.info(f"💡 {tip}")
        else:
            # ALWAYS show fallback tips (guaranteed display)
            if aqi > 200:
                st.error("🔴 Stay indoors with windows closed")
                st.error("🔴 Use air purifiers if available")
                st.error("🔴 Wear N95/N99 mask for any outdoor exposure")
                st.error("🔴 Avoid all physical exertion")
            elif aqi > 150:
                st.warning("🟡 Limit all outdoor activities")
                st.warning("🟡 Wear N95 mask when going outside")
                st.info("💡 Keep windows and doors closed")
                st.info("💡 Use air purifier indoors")
            elif aqi > 100:
                st.info("💡 Reduce prolonged outdoor exertion")
                st.info("💡 Sensitive groups should be cautious")
                st.info("💡 Consider indoor activities")
                st.success("✅ Most people can proceed normally")
            else:
                st.success("✅ Safe for all outdoor activities!")
                st.success("✅ Enjoy the fresh air")
                st.info("💡 Great day for exercise and outdoor play")
                st.info("💡 Perfect conditions for all age groups")
        
        # Historical data
        st.markdown("---")
        st.markdown("### 📊 How Has Air Quality Changed?")
        

        try:
            historical_data = data_manager.get_historical_data(user_input['location'], hours=24)
            num_points = len(historical_data) if historical_data else 0
            if historical_data and num_points > 0:
                fig_history = create_historical_chart(historical_data)
                if fig_history:
                    st.plotly_chart(fig_history, use_container_width=True)
                    if num_points < 3:
                        st.info(f"ℹ️ Only {num_points} data point{'s' if num_points > 1 else ''} available. Check AQI for this location a few more times to see richer history!")
                else:
                    st.warning("⚠️ Unable to create historical chart.")
            else:
                st.info("📝 No historical AQI data yet for this location. Check AQI a few times to build up history!")
            if user_input.get('show_debug'):
                st.caption(f"Debug: Retrieved {num_points} data points")
        except Exception as e:
            st.error(f"❌ Error loading historical data: {str(e)}")
            logger.error(f"Historical data error: {e}", exc_info=True)
        
        # Smart Q&A Section
        st.markdown("---")
        st.markdown("### 💬 Ask Me Anything About Air Quality")
        st.caption("Powered by NLP - I can understand natural language questions!")
        
        query_handler = get_universal_query_handler()
        # Debug: Show if advanced NLP/ML models are loaded
        try:
            from models.conversational_ai import get_conversational_ai
            ai = get_conversational_ai()
            st.caption(f"[DEBUG] Advanced NLP loaded: {getattr(ai, 'is_advanced_ai_loaded', lambda: False)()}")
        except Exception as e:
            st.caption(f"[DEBUG] Advanced NLP not loaded: {e}")
        
        # Initialize session state for Q&A
        if 'pending_qa_question' not in st.session_state:
            st.session_state.pending_qa_question = ''
        if 'last_processed_question' not in st.session_state:
            st.session_state.last_processed_question = ''
        
        # Check if there's a pending question to display
        default_value = st.session_state.pending_qa_question
        
        # Suggested questions - these set pending question and rerun
        st.markdown("💡 Try asking:")
        col_q1, col_q2, col_q3 = st.columns(3)
        with col_q1:
            if st.button("🏃 Is it safe to jog?", key="q1", use_container_width=True):
                st.session_state.pending_qa_question = "Is it safe to go jogging right now?"
                st.rerun()
        with col_q2:
            if st.button("😷 What protection?", key="q2", use_container_width=True):
                st.session_state.pending_qa_question = "What kind of mask should I wear today?"
                st.rerun()
        with col_q3:
            if st.button("📊 What is PM2.5?", key="q3", use_container_width=True):
                st.session_state.pending_qa_question = "What is PM2.5 and how does it affect health?"
                st.rerun()
        
        # Additional quick questions
        col_q4, col_q5, col_q6 = st.columns(3)
        with col_q4:
            if st.button("⚠️ Is it hazardous?", key="q4", use_container_width=True):
                st.session_state.pending_qa_question = "Is it hazardous outside right now?"
                st.rerun()
        with col_q5:
            if st.button("👶 Safe for kids?", key="q5", use_container_width=True):
                st.session_state.pending_qa_question = "Is it safe for children to play outside?"
                st.rerun()
        with col_q6:
            if st.button("⏰ Best time?", key="q6", use_container_width=True):
                st.session_state.pending_qa_question = "What's the best time to go outside today?"
                st.rerun()
        
        # Text input for custom questions - use default_value for pre-filled questions
        question = st.text_input(
            "Your question:",
            value=default_value,
            placeholder="Ask anything - e.g., Can I exercise outside? What's the best time to go out? Is it hazardous?"
        )
        
        # Clear pending question after displaying it
        if default_value and question == default_value:
            st.session_state.pending_qa_question = ''
        
        if question and question != st.session_state.last_processed_question:
            # Prepare AQI context for universal query handler
            context_aqi_data = {
                'aqi': aqi,
                'pollutants': aqi_data.get('pollutants', {}),
                'location': user_input['location'],
                'dominant_pollutant': aqi_data.get('dominant_pollutant'),
                'source': aqi_data.get('source', 'unknown')
            }
            
            # Mark question as processed
            st.session_state.last_processed_question = question
            
            try:
                with st.spinner("🤔 Analyzing your question with AI..."):
                    answer = query_handler.handle_query(question, context_aqi_data)
                
                # Validate answer structure
                if not answer or not isinstance(answer, dict):
                    st.error("❌ Received invalid response from NLP engine.")
                    if user_input.get('show_debug'):
                        st.caption(f"Debug: answer type = {type(answer)}, value = {answer}")
                elif not answer.get('answer'):
                    st.warning("⚠️ NLP engine didn't generate an answer. Try rephrasing your question.")
                    if user_input.get('show_debug'):
                        st.caption(f"Debug: answer keys = {answer.keys()}")
                else:
                    # Display answer in a nice format
                    st.markdown("---")
                    st.markdown("#### 🤖 AI Response")
                    
                    # Answer box with styling
                    answer_text = str(answer['answer'])
                    answer_html = f"""
                    <div style='background: linear-gradient(135deg, #667eea22 0%, #764ba222 100%); 
                                padding: 20px; border-radius: 15px; margin: 10px 0; 
                                border-left: 4px solid #667eea;'>
                        {answer_text.replace(chr(10), '<br>')}
                    </div>
                    """
                    st.markdown(answer_html, unsafe_allow_html=True)
                    
                    # Confidence and metadata
                    confidence_pct = answer.get('confidence', 0.5) * 100
                    confidence_color = '#27ae60' if confidence_pct >= 80 else '#f39c12' if confidence_pct >= 60 else '#e74c3c'
                    
                    col_conf, col_intent, col_topics = st.columns([1, 1, 2])
                    with col_conf:
                        st.markdown(f"""
                        <span style='color: {confidence_color}; font-weight: bold;'>
                            📊 Confidence: {confidence_pct:.0f}%
                        </span>
                        """, unsafe_allow_html=True)
                    with col_intent:
                        intent_emoji = {'safety': '🛡️', 'activity': '🏃', 'health': '🏥', 
                                       'protection': '😷', 'timing': '⏰', 'explanation': '📖',
                                       'recommendation': '💡', 'comparison': '⚖️', 
                                       'forecast': '🔮', 'location': '📍',
                                       'greeting': '👋', 'acknowledgment': '😊', 'help': '🤝'}.get(answer.get('intent') or answer.get('type', ''), '💬')
                        st.markdown(f"{intent_emoji} Intent: {answer.get('intent', 'general').title()}")
                    with col_topics:
                        if answer.get('related_topics'):
                            st.markdown(f"📚 Related: {', '.join(answer['related_topics'])}")
            
            except Exception as e:
                st.error(f"❌ Error processing your question: {str(e)}")
                logger.error(f"NLP Q&A error for question '{question}': {e}", exc_info=True)
                if user_input.get('show_debug'):
                    st.code(f"Exception type: {type(e).__name__}\nDetails: {str(e)}")
                st.info("💡 Try asking in a different way or use the suggested questions above.")
        
        # Debug info
        if user_input['show_debug']:
            st.markdown("---")
            st.markdown("### 🔍 Debug Information")
            with st.expander("Raw AQI Data"):
                st.json(aqi_data)
            with st.expander("User Profile"):
                st.json(user_profile)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; padding: 20px;'>
        <p><strong>🌍 AQI Health & Activity Planner</strong></p>
        <p>Powered by Advanced ML Models & Hugging Face NLP | Real-time API Integration</p>
        <p><small>Data sources: WAQI, OpenWeather, ML Simulations | Updates every 5 minutes</small></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error(f"Application error: {e}", exc_info=True)
        
        if st.button("🔄 Restart Application"):
            st.rerun()
