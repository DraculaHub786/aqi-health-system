#!/usr/bin/env python3
"""Test that AQI context extraction works correctly"""

from models.nlp_engine import get_qa_engine

def test_aqi_extraction():
    qa_engine = get_qa_engine()
    
    # Test with the exact context format from streamlit_app.py
    context = """
    Current AQI in New York is 85, which is Moderate.
    The dominant pollutant is PM2.5.
    Health risk level is Moderate for the user profile.
    Location: New York
    """
    
    # Try to extract AQI
    result = qa_engine._extract_aqi_context(context)
    print(f"Extracted AQI context: {result}")
    
    # Test a question with this context
    question = "Is it safe to go jogging?"
    answer = qa_engine.answer_question(question, context)
    
    print(f"\nQuestion: {question}")
    print(f"AQI used: {result['aqi']}")
    print(f"Answer: {answer['answer'][:200]}...")
    
    # Verify AQI was extracted correctly
    if result['aqi'] == 85:
        print("\n✅ SUCCESS: AQI 85 correctly extracted and used!")
        return True
    else:
        print(f"\n❌ FAILED: Expected AQI 85 but got {result['aqi']}")
        return False

if __name__ == "__main__":
    success = test_aqi_extraction()
    exit(0 if success else 1)
