#!/usr/bin/env python3
"""
Standalone script to test memory database viewing functionality.
"""

import sqlite3
import json
from datetime import datetime
import os

def view_memory_database(db_path: str = "model_memory.db"):
    """Utility function to view contents of the memory database."""
    try:
        if not os.path.exists(db_path):
            print(f"No memory database found at: {db_path}")
            print("The database will be created when you run the main system for the first time.")
            return
            
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        print(f"\n=== Memory Database Contents: {db_path} ===")
        
        # Check Qwen-Image memory
        cursor.execute("SELECT COUNT(*) FROM qwen_image_memory")
        qwen_image_count = cursor.fetchone()[0]
        print(f"\nQwen-Image Memory Entries: {qwen_image_count}")
        
        if qwen_image_count > 0:
            cursor.execute('''
                SELECT timestamp, image_index, original_prompt, evaluation_score, 
                       confidence_score, regeneration_count, good_things, bad_things
                FROM qwen_image_memory 
                ORDER BY timestamp DESC 
                LIMIT 3
            ''')
            results = cursor.fetchall()
            for i, row in enumerate(results):
                print(f"\n  Entry {i+1}:")
                print(f"    Timestamp: {row[0]}")
                print(f"    Image Index: {row[1]}")
                print(f"    Original Prompt: {row[2][:100]}...")
                print(f"    Evaluation Score: {row[3]}")
                print(f"    Confidence Score: {row[4]}")
                print(f"    Regeneration Count: {row[5]}")
                print(f"    Good Things: {row[6][:100]}...")
                print(f"    Bad Things: {row[7][:100]}...")
        
        # Check Qwen-Image-Edit memory
        cursor.execute("SELECT COUNT(*) FROM qwen_image_edit_memory")
        qwen_edit_count = cursor.fetchone()[0]
        print(f"\nQwen-Image-Edit Memory Entries: {qwen_edit_count}")
        
        if qwen_edit_count > 0:
            cursor.execute('''
                SELECT timestamp, image_index, original_prompt, evaluation_score, 
                       confidence_score, regeneration_count, reference_image, good_things, bad_things
                FROM qwen_image_edit_memory 
                ORDER BY timestamp DESC 
                LIMIT 3
            ''')
            results = cursor.fetchall()
            for i, row in enumerate(results):
                print(f"\n  Entry {i+1}:")
                print(f"    Timestamp: {row[0]}")
                print(f"    Image Index: {row[1]}")
                print(f"    Original Prompt: {row[2][:100]}...")
                print(f"    Evaluation Score: {row[3]}")
                print(f"    Confidence Score: {row[4]}")
                print(f"    Regeneration Count: {row[5]}")
                print(f"    Reference Image: {row[6]}")
                print(f"    Good Things: {row[7][:100]}...")
                print(f"    Bad Things: {row[8][:100]}...")
        
        conn.close()
        print(f"\n=== End Database Contents ===\n")
        
    except Exception as e:
        print(f"Error viewing database: {str(e)}")

if __name__ == "__main__":
    import sys
    db_path = sys.argv[1] if len(sys.argv) > 1 else "model_memory.db"
    view_memory_database(db_path)
