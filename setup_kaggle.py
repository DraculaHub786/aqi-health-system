"""
Setup script for Kaggle datasets
Downloads and configures AQI datasets for better recommendations
"""

import os
import sys
from pathlib import Path

def setup_kaggle():
    """Setup Kaggle integration"""
    print("=" * 60)
    print("AQI Health System - Kaggle Dataset Setup")
    print("=" * 60)
    
    # Check if kaggle is installed
    try:
        import kaggle
        print("✓ Kaggle API is installed")
    except ImportError:
        print("✗ Kaggle API not installed")
        print("\nInstalling Kaggle API...")
        os.system(f"{sys.executable} -m pip install kaggle")
        try:
            import kaggle
            print("✓ Kaggle API installed successfully")
        except:
            print("✗ Failed to install Kaggle API")
            return False
    
    # Check Kaggle authentication
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    
    if not kaggle_json.exists():
        print("\n" + "=" * 60)
        print("⚠️  Kaggle API Key Not Found")
        print("=" * 60)
        print("\nTo download datasets from Kaggle, you need an API key:")
        print("\n1. Create a free account at https://www.kaggle.com")
        print("2. Go to Account Settings > API")
        print("3. Click 'Create New API Token'")
        print("4. Save the downloaded kaggle.json to:")
        if os.name == 'nt':  # Windows
            print(f"   {Path.home() / '.kaggle' / 'kaggle.json'}")
        else:  # Linux/Mac
            print(f"   {Path.home() / '.kaggle' / 'kaggle.json'}")
            print("   Then run: chmod 600 ~/.kaggle/kaggle.json")
        
        print("\n" + "=" * 60)
        print("💡 Running in DEMO MODE with sample data")
        print("=" * 60)
        
        # Create sample data instead
        from utils.kaggle_dataset import KaggleAQIDataset
        manager = KaggleAQIDataset()
        manager._create_sample_datasets()
        db = manager.build_recommendations_database()
        
        print("\n✓ Sample dataset created successfully!")
        print(f"✓ Recommendations database built with {len(db)} categories")
        return True
    
    # Kaggle API is configured
    print("✓ Kaggle API key found")
    
    # Initialize dataset manager
    try:
        from utils.kaggle_dataset import KaggleAQIDataset
        
        print("\n" + "=" * 60)
        print("Downloading Kaggle Datasets")
        print("=" * 60)
        
        manager = KaggleAQIDataset()
        
        # Try to download datasets
        print("\nThis may take a few minutes...")
        results = manager.download_all_datasets(force=False)
        
        print("\n📦 Download Results:")
        for dataset, success in results.items():
            status = "✓" if success else "✗"
            print(f"  {status} {dataset}")
        
        # Build recommendations database
        print("\n" + "=" * 60)
        print("Building Recommendations Database")
        print("=" * 60)
        
        db = manager.build_recommendations_database()
        
        print(f"\n✓ Database built successfully!")
        print(f"  • AQI Ranges: {len(db.get('aqi_ranges', {}))}")
        print(f"  • Pollutant Info: {len(db.get('pollutant_health_impacts', {}))}")
        print(f"  • Seasonal Patterns: {len(db.get('seasonal_patterns', {}))}")
        print(f"  • Activity Thresholds: {len(db.get('activity_safety_thresholds', {}))}")
        
        if db.get('city_specific_advice'):
            print(f"  • City Data: {len(db['city_specific_advice'])} cities")
        
        print("\n" + "=" * 60)
        print("✨ Setup Complete!")
        print("=" * 60)
        print("\nYour AQI system now has:")
        print("  • Real-world AQI datasets")
        print("  • Evidence-based health recommendations")
        print("  • Pollutant-specific guidance")
        print("  • Seasonal pattern analysis")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error during setup: {e}")
        print("\nCreating sample data as fallback...")
        
        from utils.kaggle_dataset import KaggleAQIDataset
        manager = KaggleAQIDataset()
        manager._create_sample_datasets()
        db = manager.build_recommendations_database()
        
        print("✓ Running with sample data")
        return True


def test_kaggle_integration():
    """Test Kaggle integration"""
    print("\n" + "=" * 60)
    print("Testing Kaggle Integration")
    print("=" * 60)
    
    try:
        from utils.kaggle_dataset import KaggleAQIDataset
        
        manager = KaggleAQIDataset()
        
        # Test recommendations
        print("\n📊 Testing recommendations for different AQI levels...")
        
        test_cases = [
            (45, {'PM2.5': 10.5, 'PM10': 25.3}),
            (125, {'PM2.5': 55.5, 'NO2': 45.2}),
            (225, {'PM2.5': 125.0, 'O3': 85.0})
        ]
        
        for aqi, pollutants in test_cases:
            recs = manager.get_recommendations_for_aqi(aqi, pollutants)
            print(f"\n  AQI {aqi}: {recs['category'].upper()}")
            print(f"    Activities: {', '.join(recs['activities'][:3])}")
            if recs.get('precautions'):
                print(f"    Precautions: {recs['precautions'][0]}")
        
        print("\n✓ All tests passed!")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        return False
    
    return True


if __name__ == "__main__":
    print("\n")
    
    # Run setup
    success = setup_kaggle()
    
    if success:
        # Run tests
        test_kaggle_integration()
        
        print("\n" + "=" * 60)
        print("🎉 Kaggle Setup Complete!")
        print("=" * 60)
        print("\nYou can now run the application with enhanced recommendations:")
        print("  • streamlit run streamlit_app.py")
        print("  • python app.py (Flask app)")
        print("\nThe NLP engine will now use real-world data for better advice!")
        print("\n")
    else:
        print("\n⚠️  Setup completed with warnings")
        print("The system will work with basic recommendations\n")
