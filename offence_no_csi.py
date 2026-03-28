import csv
import os
from collections import Counter

# 1. Define the file path
# Using a raw string (r"...") to handle Windows backslashes correctly
file_path = r"project\DB_csv\Open_Consolidated_Data.csv"

def is_blank(value):
    """
    Helper function to determine if a value is effectively empty.
    Handles None, empty strings, and strings containing only whitespace.
    """
    if value is None:
        return True
    return str(value).strip() == ""

def analyze_data(filepath):
    # Counters to store frequencies
    csi_without_event = Counter()
    event_without_csi = Counter()
    
    # Check if file exists to avoid crashes
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return

    try:
        with open(filepath, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            # Ensure the expected columns exist in the CSV header
            required_cols = {'CSI_CATEGORY', 'EVENT_TYPE'}
            if not required_cols.issubset(set(reader.fieldnames)):
                print(f"Error: CSV is missing required columns. Found: {reader.fieldnames}")
                return

            for row in reader:
                csi = row.get('CSI_CATEGORY', '')
                event = row.get('EVENT_TYPE', '')
                
                # Check if values are blank using our helper
                csi_is_blank = is_blank(csi)
                event_is_blank = is_blank(event)
                
                # Task 1: Has CSI_Category but NO Event_Type
                if not csi_is_blank and event_is_blank:
                    csi_without_event[csi.strip()] += 1
                    
                # Task 2: Has Event_Type but NO CSI_Category
                elif not event_is_blank and csi_is_blank:
                    event_without_csi[event.strip()] += 1
                    
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return

    # --- Output Results ---
    print("-" * 60)
    print("ANALYSIS RESULTS")
    print("-" * 60)
    
    # Result 1
    print(f"\n1. CSI_CATEGORY present, but EVENT_TYPE missing:")
    print(f"   Total rows found: {sum(csi_without_event.values())}")
    if csi_without_event:
        print(f"   {'Category':<30} | {'Count':<10}")
        print("   " + "-" * 45)
        # Sort by most frequent
        for category, count in csi_without_event.most_common():
            print(f"   {category:<30} | {count:<10}")
    else:
        print("   No rows found matching this criteria.")

    # Result 2
    print(f"\n2. EVENT_TYPE present, but CSI_CATEGORY missing:")
    print(f"   Total rows found: {sum(event_without_csi.values())}")
    if event_without_csi:
        print(f"   {'Event Type':<30} | {'Count':<10}")
        print("   " + "-" * 45)
        # Sort by most frequent
        for event, count in event_without_csi.most_common():
            print(f"   {event:<30} | {count:<10}")
    else:
        print("   No rows found matching this criteria.")
        
    print("-" * 60)

if __name__ == "__main__":
    analyze_data(file_path)