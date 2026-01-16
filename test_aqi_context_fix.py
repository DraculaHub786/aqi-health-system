#!/usr/bin/env python3
"""
Comprehensive test for AQI context extraction fix
This verifies that the NLP engine correctly uses fetched AQI values
"""

import re
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_regex_extraction():
    """Test that the regex pattern correctly extracts AQI values"""
    print("=" * 60)
    print("TEST 1: Regex Pattern Extraction")
    print("=" * 60)
    
    # Test contexts that streamlit_app.py would create
    test_cases = [
        {
            'context': '''
            Current AQI in New York is 85, which is Moderate.
            The dominant pollutant is PM2.5.
            Health risk level is Moderate for the user profile.
            Location: New York
            ''',
            'expected_aqi': 85,
            'description': 'Standard streamlit format'
        },
        {
            'context': '''
            Current AQI in Delhi is 150, which is Unhealthy.
            The dominant pollutant is PM10.
            Health risk level is High for the user profile.
            Location: Delhi
            ''',
            'expected_aqi': 150,
            'description': 'High AQI (Unhealthy)'
        },
        {
            'context': '''
            Current AQI in California is 45, which is Good.
            The dominant pollutant is Ozone.
            Health risk level is Low for the user profile.
            Location: California
            ''',
            'expected_aqi': 45,
            'description': 'Low AQI (Good)'
        },
    ]
    
    pattern = r'aqi.*?is\s+([0-9]+)'
    
    all_passed = True
    for i, test in enumerate(test_cases, 1):
        match = re.search(pattern, test['context'].lower())
        if match:
            extracted_aqi = int(match.group(1))
            passed = extracted_aqi == test['expected_aqi']
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"\nTest 1.{i}: {test['description']}")
            print(f"  Expected AQI: {test['expected_aqi']}")
            print(f"  Extracted AQI: {extracted_aqi}")
            print(f"  {status}")
            all_passed = all_passed and passed
        else:
            print(f"\nTest 1.{i}: {test['description']}")
            print(f"  ❌ FAIL - No AQI found in context")
            all_passed = False
    
    return all_passed


def test_nlp_engine_extraction():
    """Test that the NLP engine correctly extracts and uses AQI"""
    print("\n" + "=" * 60)
    print("TEST 2: NLP Engine AQI Context Extraction")
    print("=" * 60)
    
    try:
        from models.nlp_engine import get_qa_engine
        
        qa_engine = get_qa_engine()
        
        # Test context similar to what streamlit_app.py sends
        context = '''
        Current AQI in New York is 95, which is Moderate.
        The dominant pollutant is PM2.5.
        Health risk level is Moderate for the user profile.
        Location: New York
        '''
        
        # Extract AQI using the NLP engine's method
        result = qa_engine._extract_aqi_context(context)
        
        print(f"\nContext: {context.strip()[:50]}...")
        print(f"Extracted AQI: {result['aqi']}")
        print(f"Expected AQI: 95")
        
        if result['aqi'] == 95:
            print("✅ PASS - AQI correctly extracted by NLP engine")
            return True
        else:
            print(f"❌ FAIL - Expected 95 but got {result['aqi']}")
            return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


def test_nlp_response_with_different_aqi_levels():
    """Test that NLP responses differ based on AQI level"""
    print("\n" + "=" * 60)
    print("TEST 3: NLP Response Variation with AQI Levels")
    print("=" * 60)
    
    try:
        from models.nlp_engine import get_qa_engine
        
        qa_engine = get_qa_engine()
        question = "Is it safe to go jogging?"
        
        test_cases = [
            {'aqi': 50, 'category': 'Good', 'expected_recommendation': 'safe'},
            {'aqi': 100, 'category': 'Moderate', 'expected_recommendation': 'caution'},
            {'aqi': 200, 'category': 'Very Unhealthy', 'expected_recommendation': 'avoid'},
        ]
        
        all_passed = True
        for test in test_cases:
            context = f'''
            Current AQI in TestCity is {test['aqi']}, which is {test['category']}.
            The dominant pollutant is PM2.5.
            Health risk level is Moderate for the user profile.
            Location: TestCity
            '''
            
            answer = qa_engine.answer_question(question, context)
            answer_text = answer['answer'].lower()
            
            # Check if response mentions the expected recommendation
            contains_keyword = test['expected_recommendation'] in answer_text or (
                test['expected_recommendation'] == 'caution' and 'caution' in answer_text
            )
            
            status = "✅ PASS" if contains_keyword else "❌ FAIL"
            print(f"\nAQI {test['aqi']} ({test['category']}): {status}")
            print(f"  Expected keyword: '{test['expected_recommendation']}'")
            print(f"  Answer preview: {answer_text[:80]}...")
            
            all_passed = all_passed and contains_keyword
        
        return all_passed
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n🔧 AQI Context Extraction Fix Verification\n")
    
    test1_passed = test_regex_extraction()
    test2_passed = test_nlp_engine_extraction()
    test3_passed = test_nlp_response_with_different_aqi_levels()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Test 1 (Regex): {'✅ PASS' if test1_passed else '❌ FAIL'}")
    print(f"Test 2 (Engine): {'✅ PASS' if test2_passed else '❌ FAIL'}")
    print(f"Test 3 (Responses): {'✅ PASS' if test3_passed else '❌ FAIL'}")
    
    all_passed = test1_passed and test2_passed and test3_passed
    print(f"\n{'✅ ALL TESTS PASSED!' if all_passed else '❌ SOME TESTS FAILED'}")
    print("=" * 60 + "\n")
    
    exit(0 if all_passed else 1)
