import sys
sys.path.insert(0, 'c:/Users/afjal/Documents/Mini-Projects/aqi-health-system')

from utils.kaggle_dataset import KaggleAQIDataset

# Create sample datasets
print("Creating sample datasets...")
m = KaggleAQIDataset()
m._create_sample_datasets()

# Build recommendations database
print("Building recommendations database...")
db = m.build_recommendations_database()

print(f"\nSuccess! Created recommendations database with {len(db)} categories")
print(f"Categories: {', '.join(db.keys())}")

# Test recommendations
recs = m.get_recommendations_for_aqi(125, {'PM2.5': 55.5})
print(f"\nTest: AQI 125")
print(f"Category: {recs['category']}")
print(f"Activities: {recs['activities']}")
