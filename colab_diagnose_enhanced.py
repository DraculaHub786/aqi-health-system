"""
Enhanced Colab Diagnostic Script
Run this to identify why sections don't show in Colab
"""

import os
import sys
import traceback

print("\n" + "="*70)
print("🔍 ENHANCED COLAB DIAGNOSTICS")
print("="*70)

# TEST 1: NLTK Data
print("\n📚 Test 1: NLTK Data Packages")
print("-" * 70)
try:
    import nltk
    packages = {
        'punkt': 'tokenizers/punkt',
        'wordnet': 'corpora/wordnet',
        'omw-1.4': 'corpora/omw-1.4',
        'vader_lexicon': 'sentiment/vader_lexicon',
        'stopwords': 'corpora/stopwords',
        'averaged_perceptron_tagger': 'taggers/averaged_perceptron_tagger'
    }
    
    missing = []
    for name, path in packages.items():
        try:
            nltk.data.find(path)
            print(f"  ✓ {name}")
        except LookupError:
            print(f"  ✗ {name} - MISSING!")
            missing.append(name)
    
    if missing:
        print(f"\n  ⚠ ISSUE FOUND: Missing NLTK packages: {', '.join(missing)}")
        print(f"  💡 FIX: Run this in a Colab cell:")
        print(f"     import nltk")
        for pkg in missing:
            print(f"     nltk.download('{pkg}')")
    else:
        print(f"\n  ✅ All NLTK packages present")
        
except Exception as e:
    print(f"  ✗ NLTK Error: {e}")
    traceback.print_exc()

# TEST 2: TextBlob
print("\n📖 Test 2: TextBlob Corpora")
print("-" * 70)
try:
    from textblob import TextBlob
    test = TextBlob("test sentence")
    sentiment = test.sentiment
    print(f"  ✓ TextBlob working (polarity: {sentiment.polarity})")
except Exception as e:
    print(f"  ✗ TextBlob Error: {e}")
    print(f"  💡 FIX: Run in Colab:")
    print(f"     !python -m textblob.download_corpora")

# TEST 3: Database
print("\n💾 Test 3: Database & Historical Data")
print("-" * 70)
try:
    import sqlite3
    conn = sqlite3.connect('data/aqi_history.db')
    cursor = conn.cursor()
    
    # Check table
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='aqi_history'")
    if cursor.fetchone():
        print(f"  ✓ Table 'aqi_history' exists")
        
        # Check records
        cursor.execute("SELECT COUNT(*) FROM aqi_history")
        count = cursor.fetchone()[0]
        print(f"  ✓ {count} records in database")
        
        if count == 0:
            print(f"  ⚠ ISSUE FOUND: Database is empty!")
            print(f"  💡 FIX: Run colab_setup.py to populate sample data")
        else:
            # Get sample
            cursor.execute("SELECT * FROM aqi_history LIMIT 1")
            sample = cursor.fetchone()
            print(f"  ✓ Sample record exists")
    else:
        print(f"  ✗ Table doesn't exist!")
        print(f"  💡 FIX: Run colab_setup.py to create database")
    
    conn.close()
except Exception as e:
    print(f"  ✗ Database Error: {e}")
    traceback.print_exc()

# TEST 4: Health Tips Function
print("\n💡 Test 4: Health Tips Generation")
print("-" * 70)
try:
    print("  Importing NLP engine...")
    from models.nlp_engine import HuggingFaceNLPEngine
    print("  ✓ Import successful")
    
    print("  Initializing engine...")
    nlp = HuggingFaceNLPEngine(
        model_name='distilbert-base-uncased-finetuned-sst-2-english',
        use_transformers=False,
        use_kaggle=True
    )
    print("  ✓ Engine initialized")
    
    print("  Generating tips...")
    tips = nlp.generate_personalized_tips(
        aqi=120,
        pollutants={'PM25': 45, 'PM10': 60, 'O3': 30},
        user_profile={'age': 30, 'health_conditions': []},
        location='Delhi'
    )
    
    print(f"  ✓ Function returned: type={type(tips)}, length={len(tips) if tips else 0}")
    
    if tips and len(tips) > 0:
        print(f"  ✅ Health Tips WORKING - {len(tips)} tips generated")
        print(f"\n  Sample tips:")
        for i, tip in enumerate(tips[:3], 1):
            print(f"    {i}. {tip[:100]}...")
    else:
        print(f"  ⚠ ISSUE FOUND: Function returns empty list!")
        print(f"  💡 Possible causes:")
        print(f"     - Missing NLTK/TextBlob data")
        print(f"     - Exception being caught silently")
        print(f"     - Check logs in streamlit_app.py")
        
except Exception as e:
    print(f"  ✗ Health Tips Error: {e}")
    print(f"\n  Full traceback:")
    traceback.print_exc()

# TEST 5: Historical Data Function
print("\n📊 Test 5: Historical Data Retrieval")
print("-" * 70)
try:
    print("  Importing data manager...")
    from utils.data_manager import DataManager
    print("  ✓ Import successful")
    
    print("  Initializing manager...")
    dm = DataManager()
    print("  ✓ Manager initialized")
    
    print("  Getting historical data...")
    data = dm.get_historical_data('Delhi', hours=24)
    
    print(f"  ✓ Function returned: type={type(data)}, length={len(data) if data else 0}")
    
    if data and len(data) > 0:
        print(f"  ✅ Historical Data WORKING - {len(data)} records retrieved")
        
        # Check if it meets UI threshold
        if len(data) > 2:
            print(f"  ✓ Meets UI threshold (>2 records)")
        else:
            print(f"  ⚠ Below UI threshold (needs >2, has {len(data)})")
        
        # Sample record
        print(f"\n  Sample record:")
        sample = data[0]
        print(f"    AQI: {sample.get('aqi', 'N/A')}")
        print(f"    PM2.5: {sample.get('pm25', 'N/A')}")
        print(f"    Timestamp: {sample.get('timestamp', 'N/A')}")
        
        # Check required fields
        required = ['aqi', 'pm25', 'timestamp']
        missing_fields = [f for f in required if f not in sample]
        if missing_fields:
            print(f"  ⚠ Missing fields: {', '.join(missing_fields)}")
        else:
            print(f"  ✓ All required fields present")
    else:
        print(f"  ⚠ ISSUE FOUND: Function returns empty list!")
        print(f"  💡 Check database has records for location 'Delhi'")
        
except Exception as e:
    print(f"  ✗ Historical Data Error: {e}")
    print(f"\n  Full traceback:")
    traceback.print_exc()

# SUMMARY
print("\n" + "="*70)
print("📋 DIAGNOSTIC SUMMARY")
print("="*70)
print("""
If you see ✅ for Health Tips and Historical Data:
  → Functions work! Issue is in Streamlit UI rendering.
  → Check streamlit_app.py for conditional logic issues
  → Add debug prints before rendering sections

If you see ⚠ or ✗:
  → Follow the 💡 FIX instructions above
  → Re-run this diagnostic after fixes
  → Once all tests pass, restart Streamlit app

Common Fixes:
  1. Missing NLTK: import nltk; nltk.download(['wordnet', 'vader_lexicon'])
  2. Empty database: Run colab_setup.py to add sample data
  3. TextBlob: !python -m textblob.download_corpora
""")
print("="*70 + "\n")
