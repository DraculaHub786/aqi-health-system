"""
Google Colab Setup Script
Run this before starting the Streamlit app in Colab
"""

import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_colab_environment():
    """Set up necessary directories and files for Colab"""
    
    print("🔧 Setting up Google Colab environment...")
    
    # 1. Create necessary directories
    directories = [
        'data',
        'data/kaggle',
        'models/__pycache__',
        'utils/__pycache__',
        'static/css',
        'static/js',
        'templates'
    ]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"✓ Created/verified directory: {directory}")
        except Exception as e:
            logger.error(f"✗ Failed to create {directory}: {e}")
    
    # 2. Initialize empty __init__.py files if missing
    init_files = [
        'models/__init__.py',
        'utils/__init__.py'
    ]
    
    for init_file in init_files:
        try:
            if not os.path.exists(init_file):
                with open(init_file, 'w') as f:
                    f.write('"""Package initialization"""')
                logger.info(f"✓ Created {init_file}")
            else:
                logger.info(f"✓ {init_file} exists")
        except Exception as e:
            logger.error(f"✗ Failed to create {init_file}: {e}")
    
    # 3. Check if database can be created
    try:
        import sqlite3
        db_path = 'data/aqi_history.db'
        conn = sqlite3.connect(db_path)
        conn.close()
        logger.info(f"✓ Database connection test successful: {db_path}")
    except Exception as e:
        logger.error(f"✗ Database test failed: {e}")
    
    # 4. Verify critical files exist
    critical_files = [
        'streamlit_app.py',
        'models/nlp_engine.py',
        'models/advanced_ml.py',
        'ml_model.py',
        'utils/data_manager.py',
        'utils/config.py'
    ]
    
    missing_files = []
    for file in critical_files:
        if os.path.exists(file):
            logger.info(f"✓ {file}")
        else:
            logger.error(f"✗ MISSING: {file}")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n⚠️ WARNING: Missing {len(missing_files)} critical files!")
        print("Make sure all files are uploaded to Colab.")
        return False
    
    # 5. Check Python dependencies
    print("\n📦 Checking dependencies...")
    try:
        import streamlit
        import plotly
        import pandas
        import numpy
        import sklearn
        import nltk
        import textblob
        import vaderSentiment
        logger.info("✓ All core dependencies installed")
    except ImportError as e:
        logger.error(f"✗ Missing dependency: {e}")
        print("Run: !pip install -r requirements.txt")
        return False
    
    # 6. Download NLTK data
    print("\n📚 Setting up NLTK data...")
    try:
        import nltk
        import ssl
        
        # Handle SSL
        try:
            _create_unverified_https_context = ssl._create_unverified_context
            ssl._create_default_https_context = _create_unverified_https_context
        except AttributeError:
            pass
        
        # Download required packages
        for package in ['punkt', 'averaged_perceptron_tagger', 'wordnet', 'omw-1.4', 'vader_lexicon']:
            try:
                nltk.data.find(f'tokenizers/{package}' if package == 'punkt' else f'corpora/{package}' if package != 'vader_lexicon' else f'sentiment/{package}')
                logger.info(f"✓ NLTK {package} already downloaded")
            except LookupError:
                nltk.download(package, quiet=True)
                logger.info(f"✓ Downloaded NLTK {package}")
        
        # TextBlob
        from textblob import TextBlob
        try:
            TextBlob("test").correct()
            logger.info("✓ TextBlob corpora ready")
        except:
            import subprocess
            subprocess.run(['python', '-m', 'textblob.download_corpora'], 
                          stdout=subprocess.DEVNULL, 
                          stderr=subprocess.DEVNULL,
                          timeout=30)
            logger.info("✓ Downloaded TextBlob corpora")
            
    except Exception as e:
        logger.error(f"✗ NLTK setup failed: {e}")
    
    # 7. Set environment variables
    print("\n🔑 Environment check...")
    if not os.getenv('WAQI_API_KEY'):
        logger.warning("⚠️ WAQI_API_KEY not set (app will use fallback/demo data)")
    else:
        logger.info("✓ WAQI_API_KEY configured")
    
    print("\n✅ Colab environment setup complete!")
    print("\nYou can now run:")
    print("!streamlit run streamlit_app.py --server.headless=true")
    
    return True


if __name__ == "__main__":
    success = setup_colab_environment()
    if not success:
        print("\n❌ Setup failed. Please fix errors above before running the app.")
    else:
        print("\n🎉 Ready to deploy!")
