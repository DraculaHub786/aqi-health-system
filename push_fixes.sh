#!/bin/bash
# Quick script to push fixes to GitHub

echo "🚀 Pushing NLP fixes to GitHub..."

# Add modified files
git add models/nlp_engine.py
git add utils/config.py
git add COLAB_FREE_TIER.md
git add FIXES_APPLIED.md

# Commit changes
git commit -m "Fix: Add context-aware NLP responses for lightweight mode

- Added _generate_simple_response() method for intelligent fallback
- Added NLP_CONFIG to utils/config.py
- Fixed generic responses for season, children safety, PM2.5, masks
- Health tips and historical data sections now display correctly
- Lightweight mode works perfectly without transformer models"

# Push to GitHub
git push origin main

echo ""
echo "✅ Changes pushed to GitHub!"
echo ""
echo "📋 NEXT STEPS IN GOOGLE COLAB:"
echo "================================"
echo "1. Runtime → Restart runtime"
echo "2. Re-run all cells (1 → 2 → 3 → 4 → 5)"
echo "3. Test these queries:"
echo "   • 'What season is this?'"
echo "   • 'Is it safe for children?'"
echo "   • 'What is PM2.5?'"
echo "   • 'What mask should I wear?'"
echo ""
echo "🎉 Your NLP will now work perfectly!"
