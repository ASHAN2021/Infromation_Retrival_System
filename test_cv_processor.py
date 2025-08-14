"""
Test script for CV Processor functionality
Run this to test the CV processing capabilities without the Streamlit interface
"""

import os
from src.cv_processor import CVProcessor

def test_cv_processor():
    print("🔍 Testing CV Processor...")
    
    # Initialize processor
    cv_processor = CVProcessor()
    
    # Test statistics
    stats = cv_processor.get_statistics()
    print(f"\n📊 Current Database Stats:")
    print(f"Total CVs: {stats['total_cvs']}")
    print(f"Categories: {list(stats['categories'].keys()) if stats['categories'] else 'None'}")
    print(f"Experience Levels: {list(stats['experience_levels'].keys()) if stats['experience_levels'] else 'None'}")
    
    # Test categories
    categories = cv_processor.get_all_categories()
    print(f"\n📋 Available Categories: {categories}")
    
    # Test search functionality if we have data
    if stats['total_cvs'] > 0:
        print("\n🔎 Testing Search Functionality:")
        
        # Test category search
        if categories:
            first_category = categories[0]
            results = cv_processor.search_cvs_by_category(first_category, limit=3)
            print(f"Found {len(results)} CVs in '{first_category}' category")
        
        # Test semantic search
        test_queries = [
            "software engineer",
            "python developer",
            "data scientist",
            "machine learning",
            "project manager"
        ]
        
        for query in test_queries:
            results = cv_processor.semantic_search(query, limit=2)
            print(f"Query '{query}': Found {len(results)} relevant CVs")
            
            if results:
                for result in results[:1]:  # Show first result
                    print(f"  - {result['info']['name']} ({result['info']['category']})")
    
    else:
        print("\n📝 No CVs in database yet. Upload some CVs using the Streamlit app!")
    
    print("\n✅ Test completed!")

if __name__ == "__main__":
    # Make sure we're in the right environment
    try:
        test_cv_processor()
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure you're in the 'genai' conda environment and have set your GROQ_API_KEY in .env file")
