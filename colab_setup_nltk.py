"""
NLTK & TextBlob Data Setup for Colab
Run this BEFORE starting Streamlit app
"""
import nltk
import ssl

# Fix SSL certificate issue
try:
    _create_unverified_https_context = ssl._create_unverified_context
    ssl._create_default_https_context = _create_unverified_https_context
except AttributeError:
    pass

print("📦 Downloading NLTK data...")

# Download all required NLTK packages
packages = ['punkt', 'averaged_perceptron_tagger', 'wordnet', 'omw-1.4', 'vader_lexicon', 'stopwords']
for package in packages:
    try:
        nltk.data.find(f'tokenizers/{package}' if package == 'punkt' else 
                      f'taggers/{package}' if 'tagger' in package else 
                      f'corpora/{package}' if package in ['wordnet', 'omw-1.4', 'stopwords'] else 
                      f'sentiment/{package}')
        print(f"✅ {package} already downloaded")
    except LookupError:
        print(f"⬇️ Downloading {package}...")
        nltk.download(package)
        print(f"✅ {package} downloaded")

print("\n📦 Downloading TextBlob corpora...")

# Download TextBlob corpora
try:
    from textblob import TextBlob
    # Test if corpora is available
    test = TextBlob("test")
    test.tags  # Will fail if corpora missing
    print("✅ TextBlob corpora already available")
except Exception as e:
    print(f"⬇️ Downloading TextBlob corpora... (error: {e})")
    import subprocess
    result = subprocess.run(
        ['python', '-m', 'textblob.download_corpora'],
        capture_output=True,
        text=True,
        timeout=60
    )
    if result.returncode == 0:
        print("✅ TextBlob corpora downloaded successfully")
    else:
        print(f"⚠️ TextBlob download had issues: {result.stderr}")
        # Alternative download method
        print("Trying alternative download...")
        nltk.download('brown')
        nltk.download('punkt')
        print("✅ Alternative download complete")

print("\n✅ All NLP data ready!")
print("You can now run: !streamlit run streamlit_app.py")
