"""
Test the Advanced Conversational NLP Engine
This script demonstrates the AI-powered chat capabilities
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_conversational_ai():
    """Test the conversational AI with various queries"""
    print("=" * 70)
    print("🤖 Testing Advanced Conversational AI for AQI Health System")
    print("=" * 70)
    print()
    
    # Import the conversational AI
    try:
        from models.conversational_ai import get_conversational_ai
        print("✓ Successfully imported Conversational AI module")
    except ImportError as e:
        print(f"❌ Error importing: {e}")
        print("\nPlease install required packages:")
        print("pip install transformers torch sentencepiece sentence-transformers")
        return
    
    # Initialize the AI
    print("\n🔄 Initializing Conversational AI models...")
    print("   (This may take a minute on first run - models are being downloaded)")
    print()
    
    try:
        ai = get_conversational_ai()
        print("✓ Conversational AI initialized successfully!")
    except Exception as e:
        print(f"❌ Error initializing AI: {e}")
        return
    
    # Test AQI context
    test_aqi_context = {
        'aqi': 85,
        'pollutants': {
            'pm25': 35.5,
            'pm10': 45.2,
            'o3': 65.0,
            'no2': 25.0
        },
        'location': 'Test City'
    }
    
    # Test queries
    test_queries = [
        "Hello!",
        "Is it safe for kids to play outside?",
        "What is PM2.5 and how does it affect health?",
        "Can I go jogging right now?",
        "What's the best time to exercise today?",
        "Should I wear a mask?",
        "Thank you for your help!",
    ]
    
    print("\n" + "=" * 70)
    print("🧪 Testing Various Queries")
    print("=" * 70)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'─' * 70}")
        print(f"Query #{i}: {query}")
        print(f"{'─' * 70}")
        
        try:
            response = ai.chat(query, test_aqi_context)
            
            print(f"\n🤖 AI Response:")
            print(response.get('answer', 'No answer'))
            print(f"\n📊 Metadata:")
            print(f"   Intent: {response.get('intent', 'unknown')}")
            print(f"   Confidence: {response.get('confidence', 0):.2%}")
            
            if 'suggestions' in response:
                print(f"\n💡 Suggestions:")
                for suggestion in response.get('suggestions', []):
                    print(f"   • {suggestion}")
        
        except Exception as e:
            print(f"❌ Error processing query: {e}")
            import traceback
            traceback.print_exc()
    
    # Test conversation memory
    print("\n" + "=" * 70)
    print("🧠 Testing Conversation Memory")
    print("=" * 70)
    
    summary = ai.get_conversation_summary()
    print(f"\n📝 Conversation Summary:")
    print(f"   Total messages: {summary['message_count']}")
    print(f"   User context detected: {summary['user_context']}")
    print(f"   Recent intents: {summary['recent_intents']}")
    
    print("\n" + "=" * 70)
    print("✅ Testing Complete!")
    print("=" * 70)
    print("\n💡 The NLP system is working! Try it in the Streamlit app:")
    print("   streamlit run streamlit_app.py")
    print()


if __name__ == "__main__":
    test_conversational_ai()
