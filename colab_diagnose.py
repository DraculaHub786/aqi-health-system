"""
Colab Diagnostic Script
Run this in Colab to diagnose Health Tips and Historical Data issues
"""

import os
import sys

print("🔍 Diagnosing Colab Environment...")
print("=" * 70)

# 1. Check directories
print("\n📁 Directory Check:")
directories = ['data', 'data/kaggle', 'models', 'utils']
for dir_path in directories:
    exists = os.path.exists(dir_path)
    is_writable = os.access(dir_path, os.W_OK) if exists else False
    print(f"  {dir_path}: {'✓ EXISTS' if exists else '✗ MISSING'} | {'✓ WRITABLE' if is_writable else '✗ READ-ONLY'}")

# 2. Check database
print("\n💾 Database Check:")
db_path = 'data/aqi_history.db'
if os.path.exists(db_path):
    print(f"  ✓ Database exists: {db_path}")
    import sqlite3
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM aqi_history")
        count = cursor.fetchone()[0]
        print(f"  ✓ Records in database: {count}")
        conn.close()
    except Exception as e:
        print(f"  ✗ Database error: {e}")
else:
    print(f"  ✗ Database missing: {db_path}")

# 3. Test Health Tips function
print("\n💡 Health Tips Test:")
try:
    from models.nlp_engine import get_nlp_engine
    nlp_engine = get_nlp_engine()
    
    test_pollutants = {'pm25': 75, 'pm10': 100, 'o3': 40}
    test_profile = {
        'age': 30,
        'respiratory_condition': False,
        'heart_condition': False,
        'daily_outdoor_hours': 2,
        'outdoor_worker': False
    }
    
    tips = nlp_engine.generate_personalized_tips(
        aqi=130,
        pollutants=test_pollutants,
        user_profile=test_profile,
        location='Delhi'
    )
    
    if tips and len(tips) > 0:
        print(f"  ✓ Tips generated: {len(tips)} tips")
        for i, tip in enumerate(tips[:3], 1):
            print(f"    {i}. {tip[:60]}...")
    else:
        print(f"  ✗ No tips generated (returned: {tips})")
        
except Exception as e:
    print(f"  ✗ Error generating tips: {e}")
    import traceback
    traceback.print_exc()

# 4. Test Historical Data function
print("\n📊 Historical Data Test:")
try:
    from utils.data_manager import get_data_manager
    data_manager = get_data_manager()
    
    historical_data = data_manager.get_historical_data('Delhi', hours=24)
    
    if historical_data:
        print(f"  ✓ Historical data retrieved: {len(historical_data)} records")
        if len(historical_data) > 0:
            print(f"    Sample record: {historical_data[0]}")
    else:
        print(f"  ✗ No historical data (returned: {historical_data})")
        
except Exception as e:
    print(f"  ✗ Error getting historical data: {e}")
    import traceback
    traceback.print_exc()

# 5. Check NLTK data
print("\n📚 NLTK Data Check:")
try:
    import nltk
    packages = ['punkt', 'averaged_perceptron_tagger', 'wordnet', 'omw-1.4', 'vader_lexicon']
    for package in packages:
        try:
            nltk.data.find(f'tokenizers/{package}' if package == 'punkt' else 
                          f'corpora/{package}' if package not in ['vader_lexicon', 'averaged_perceptron_tagger'] else
                          f'sentiment/{package}' if package == 'vader_lexicon' else
                          f'taggers/{package}')
            print(f"  ✓ {package}")
        except LookupError:
            print(f"  ✗ {package} MISSING")
except Exception as e:
    print(f"  ✗ NLTK check failed: {e}")

# 6. Check Python path
print("\n🐍 Python Environment:")
print(f"  Python version: {sys.version}")
print(f"  Working directory: {os.getcwd()}")
print(f"  Python path:")
for path in sys.path[:5]:
    print(f"    - {path}")

# 7. Memory check
print("\n💾 Memory Check:")
try:
    import psutil
    mem = psutil.virtual_memory()
    print(f"  Total: {mem.total / 1024**3:.1f} GB")
    print(f"  Available: {mem.available / 1024**3:.1f} GB")
    print(f"  Used: {mem.percent}%")
except ImportError:
    print("  ⚠ psutil not installed (optional)")

print("\n" + "=" * 70)
print("✅ Diagnostic complete!")
print("\nIf you see errors above, that's the root cause.")
print("Copy the error messages and we can fix them!")
