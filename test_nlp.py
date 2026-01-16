import sys
sys.path.insert(0, 'c:/Users/afjal/Documents/Mini-Projects/aqi-health-system')

from models.nlp_engine import get_universal_query_handler

print("\n" + "="*60)
print("Testing Universal Query Handler")
print("="*60)

handler = get_universal_query_handler()

aqi_data = {
    'aqi': 125,
    'pollutants': {'PM2.5': 55.5, 'NO2': 45.2},
    'location': 'Your City'
}

test_queries = [
    "Hi!",
    "Is it safe to run?",
    "What can I do today?",
    "I'm worried about my kids",
    "Thanks!"
]

for query in test_queries:
    print(f"\nQ: {query}")
    print("-"*60)
    result = handler.handle_query(query, aqi_data)
    print(result['answer'][:200] + "..." if len(result['answer']) > 200 else result['answer'])
    print(f"[Confidence: {result['confidence']:.2f}, Type: {result.get('type', 'unknown')}]")
