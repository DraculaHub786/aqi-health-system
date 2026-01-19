# Colab NLTK Data Cleanup and Forced Download
import shutil
import nltk

# Clean up any broken NLTK data
shutil.rmtree('/root/nltk_data', ignore_errors=True)
shutil.rmtree('/content/nltk_data', ignore_errors=True)

# Download required NLTK corpora
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')

print('✅ NLTK data cleaned and downloaded successfully.')
