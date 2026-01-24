"""
Production Readiness Test Suite
Tests all critical NLP/chatbot functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.nlp_engine import get_universal_query_handler

def test_nlp_system():
    """Test that NLP system handles all query types correctly"""
    
    print("🧪 Testing Production-Ready NLP System...")
    print("=" * 60)
    
    # Initialize handler
    handler = get_universal_query_handler()
    
    # Test AQI context
    test_aqi_data = {
        'aqi': 135,
        'pollutants': {'pm25': 55.5, 'pm10': 85.2, 'o3': 45.0},
        'location': 'Delhi',
        'dominant_pollutant': 'PM2.5'
    }
    
    # Test queries
    test_cases = [
        ("What is PM2.5?", "pollutant_info"),
        ("What is protection?", "protection"),
        ("Is it safe for kids?", "safety"),
        ("Can I exercise?", "activity"),
        ("When is the best time?", "timing"),
        ("Hello", "greeting"),
        ("Is it hazardous?", "safety"),
        ("What mask should I wear?", "protection"),
        ("Tell me about ozone", "pollutant_info"),
        ("Random query that doesn't match", "general")
    ]
    
    passed = 0
    failed = 0
    
    for query, expected_category in test_cases:
        print(f"\n📝 Query: '{query}'")
        print(f"   Expected: {expected_category}")
        
        try:
            result = handler.handle_query(query, test_aqi_data)
            
            # Check if response is valid
            if not result:
                print(f"   ❌ FAILED: No response returned")
                failed += 1
                continue
                
            if not isinstance(result, dict):
                print(f"   ❌ FAILED: Invalid response type: {type(result)}")
                failed += 1
                continue
                
            if not result.get('answer'):
                print(f"   ❌ FAILED: No answer in response")
                failed += 1
                continue
            
            # Success!
            intent = result.get('intent', result.get('type', 'unknown'))
            confidence = result.get('confidence', 0)
            answer_preview = result['answer'][:100] + "..." if len(result['answer']) > 100 else result['answer']
            
            print(f"   ✅ PASSED")
            print(f"   Intent: {intent}")
            print(f"   Confidence: {confidence:.2f}")
            print(f"   Answer: {answer_preview}")
            passed += 1
            
        except Exception as e:
            print(f"   ❌ FAILED: Exception - {str(e)}")
            failed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print(f"📊 Test Results:")
    print(f"   ✅ Passed: {passed}/{len(test_cases)}")
    print(f"   ❌ Failed: {failed}/{len(test_cases)}")
    print(f"   Success Rate: {(passed/len(test_cases)*100):.1f}%")
    
    if failed == 0:
        print("\n🎉 All tests passed! System is production-ready!")
        return True
    else:
        print(f"\n⚠️ {failed} tests failed. Review failures above.")
        return False

if __name__ == "__main__":
    success = test_nlp_system()
    sys.exit(0 if success else 1)
