import sys
import os
import pandas as pd
from src.data_loader import load_chat_data
from src.poster_builder import generate_poster_report

def test_gen():
    file_path = "data/qun.json"
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    print("Loading data...")
    df, session = load_chat_data(file_path)
    print(f"Data loaded: {len(df)} messages.")
    
    # Optional: Filter to last 4 weeks to speed up test if needed, 
    # but let's try full first to see if it works.
    # df = df.tail(5000) 
    
    print("Generating poster...")
    
    try:
        output = generate_poster_report(
            session_info=session,
            df=df,
            memories_data={'golden_quotes': []}, # Empty for test
            music_url=None
        )
        print(f"Test complete. Output: {output}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_gen()
