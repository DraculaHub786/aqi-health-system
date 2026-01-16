"""
Demo script to showcase Kaggle integration and enhanced NLP
"""

def demo_universal_query_handler():
    """Demonstrate universal query handler"""
    print("\n" + "=" * 70)
    print("🤖 UNIVERSAL QUERY HANDLER DEMO")
    print("=" * 70)
    
    from models.nlp_engine import get_universal_query_handler
    
    # Initialize handler
    handler = get_universal_query_handler()
    
    # Sample AQI data
    aqi_data = {
        'aqi': 125,
        'pollutants': {
            'PM2.5': 55.5,
            'PM10': 85.2,
            'NO2': 45.2,
            'O3': 68.5
        },
        'location': 'Your City'
    }
    
    # Test various query types
    test_queries = [
        "Hi!",
        "Is it safe to run outside?",
        "What can I do today?",
        "How does PM2.5 affect my health?",
        "I'm worried about my kids playing outside",
        "Should I wear a mask?",
        "What's the best time to exercise?",
        "Indoor or outdoor workout?",
        "Tell me about the air quality",
        "Thanks for your help!"
    ]
    
    print("\n📝 Testing with AQI: 125 (Unhealthy for Sensitive Groups)")
    print(f"   Pollutants: PM2.5={aqi_data['pollutants']['PM2.5']}, NO2={aqi_data['pollutants']['NO2']}")
    print("\n" + "-" * 70)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. USER: \"{query}\"")
        print("-" * 70)
        
        result = handler.handle_query(query, aqi_data)
        
        print(f"ASSISTANT ({result.get('type', 'unknown')}):")
        print(result['answer'])
        print(f"\n   [Confidence: {result['confidence']:.2f}]")
        
        if 'suggestions' in result:
            print(f"   Suggestions: {result['suggestions']}")
        
        print("-" * 70)


def demo_kaggle_recommendations():
    """Demonstrate Kaggle dataset recommendations"""
    print("\n" + "=" * 70)
    print("📊 KAGGLE DATASET RECOMMENDATIONS DEMO")
    print("=" * 70)
    
    try:
        from utils.kaggle_dataset import KaggleAQIDataset
        
        manager = KaggleAQIDataset()
        
        # Build or load database
        print("\n📦 Loading Kaggle recommendations database...")
        db = manager.build_recommendations_database()
        print(f"✓ Database loaded with {len(db)} categories\n")
        
        # Test different AQI levels
        test_cases = [
            {
                'aqi': 45,
                'pollutants': {'PM2.5': 10.5, 'PM10': 25.3, 'NO2': 18.5},
                'label': 'Good Air Quality'
            },
            {
                'aqi': 125,
                'pollutants': {'PM2.5': 55.5, 'PM10': 85.2, 'NO2': 45.2},
                'label': 'Unhealthy for Sensitive Groups'
            },
            {
                'aqi': 225,
                'pollutants': {'PM2.5': 125.0, 'PM10': 185.0, 'O3': 85.0},
                'label': 'Very Unhealthy'
            }
        ]
        
        for case in test_cases:
            print("\n" + "=" * 70)
            print(f"📍 TEST CASE: AQI {case['aqi']} - {case['label']}")
            print("=" * 70)
            
            recs = manager.get_recommendations_for_aqi(
                case['aqi'],
                case['pollutants']
            )
            
            print(f"\n🏷️  Category: {recs['category'].upper()}")
            print(f"📊 AQI Range: {recs['category_info']['range']}")
            
            print(f"\n✅ Recommended Activities:")
            for activity in recs['activities']:
                print(f"   • {activity}")
            
            if recs.get('precautions'):
                print(f"\n⚠️  Precautions:")
                for precaution in recs['precautions']:
                    print(f"   • {precaution}")
            
            if recs.get('pollutant_alerts'):
                print(f"\n🔴 Pollutant Alerts:")
                for alert in recs['pollutant_alerts']:
                    print(f"\n   {alert['pollutant']}:")
                    print(f"   • Current: {alert['level']:.1f} µg/m³")
                    print(f"   • Safe Level: {alert['safe_level']} µg/m³")
                    print(f"   • Health Effects: {', '.join(alert['health_effects'][:2])}")
                    print(f"   • High Risk Groups: {', '.join(alert['high_risk_groups'])}")
            
            print("\n" + "-" * 70)
        
        # Show activity thresholds
        print("\n" + "=" * 70)
        print("🏃 ACTIVITY SAFETY THRESHOLDS (From Kaggle Data)")
        print("=" * 70 + "\n")
        
        thresholds = db['activity_safety_thresholds']
        for activity_type, info in thresholds.items():
            if 'max_aqi' in info:
                print(f"• {activity_type.replace('_', ' ').title()}: Max AQI {info['max_aqi']}")
                print(f"  Examples: {', '.join(info['examples'])}\n")
        
        # Show health statistics
        print("\n" + "=" * 70)
        print("📈 GLOBAL HEALTH STATISTICS (WHO Data)")
        print("=" * 70 + "\n")
        
        stats = db['health_statistics']
        for key, value in stats.items():
            print(f"• {value}")
        
    except Exception as e:
        print(f"\n⚠️  Kaggle integration not available: {e}")
        print("   Run: python setup_kaggle.py")


def demo_nlp_analysis():
    """Demonstrate NLP analysis capabilities"""
    print("\n" + "=" * 70)
    print("🧠 NLP ANALYSIS DEMO")
    print("=" * 70)
    
    from models.nlp_engine import get_nlp_engine
    
    nlp = get_nlp_engine(use_transformers=False)  # Lightweight mode
    
    test_texts = [
        "I'm really worried about the air quality today. Is it safe for my kids?",
        "Great air quality! Perfect day for a run in the park.",
        "The pollution is terrible. I have asthma and feel awful.",
    ]
    
    aqi_context = {'aqi': 125, 'location': 'City'}
    
    print("\n📝 Analyzing user queries with NLP...\n")
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n{i}. Text: \"{text}\"")
        print("-" * 70)
        
        analysis = nlp.analyze_with_ai(text, aqi_context)
        
        if 'sentiment' in analysis:
            sent = analysis['sentiment']
            print(f"   Sentiment: {sent['label']} (score: {sent['compound']:.2f})")
        
        if 'intent' in analysis:
            intent = analysis['intent']
            print(f"   Intent: {intent['intent']} (confidence: {intent['confidence']:.2f})")
        
        if 'keywords' in analysis:
            print(f"   Keywords: {', '.join(analysis['keywords'][:5])}")
        
        print("-" * 70)


def main():
    """Run all demos"""
    print("\n\n")
    print("=" * 70)
    print("  AQI HEALTH SYSTEM - ENHANCED NLP & KAGGLE INTEGRATION DEMO")
    print("=" * 70)
    
    try:
        # Demo 1: Universal Query Handler
        demo_universal_query_handler()
        
        # Demo 2: Kaggle Recommendations
        demo_kaggle_recommendations()
        
        # Demo 3: NLP Analysis
        demo_nlp_analysis()
        
        print("\n\n" + "=" * 70)
        print("✅ ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("\n💡 The system is ready to use with:")
        print("   • Universal query understanding")
        print("   • Real-world Kaggle datasets")
        print("   • Evidence-based recommendations")
        print("   • Advanced NLP analysis")
        print("\n🚀 Run the app: streamlit run streamlit_app.py")
        print("\n")
        
    except Exception as e:
        print(f"\n\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 Try running: python setup_kaggle.py first\n")


if __name__ == "__main__":
    main()
