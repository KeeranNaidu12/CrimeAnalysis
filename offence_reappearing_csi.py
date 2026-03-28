import csv
import os
from collections import Counter, defaultdict

# Define the file path
file_path = r"project\DB_csv\Open_Consolidated_Data.csv"

def is_blank(value):
    """Helper function to determine if a value is effectively empty."""
    if value is None:
        return True
    return str(value).strip() == ""

def analyze_event_csi_relationship(filepath):
    # Check if file exists
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return

    # Data structures
    event_id_to_rows = defaultdict(list)  # Store all rows for each EVENT_UNIQUE_ID
    event_ids_with_type_no_csi = set()    # EVENT_UNIQUE_IDs that have EVENT_TYPE but no CSI_CATEGORY
    
    print("=" * 80)
    print("PASS 1: Reading all data and organizing by EVENT_UNIQUE_ID")
    print("=" * 80)
    
    try:
        with open(filepath, mode='r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            
            required_cols = {'EVENT_UNIQUE_ID', 'EVENT_TYPE', 'CSI_CATEGORY'}
            if not required_cols.issubset(set(reader.fieldnames)):
                print(f"Error: CSV is missing required columns. Found: {reader.fieldnames}")
                return
            
            row_count = 0
            for row in reader:
                row_count += 1
                event_id = row.get('EVENT_UNIQUE_ID', '')
                event_type = row.get('EVENT_TYPE', '')
                csi_category = row.get('CSI_CATEGORY', '')
                
                if not is_blank(event_id):
                    event_id_clean = event_id.strip()
                    # Store the entire row for this EVENT_UNIQUE_ID
                    event_id_to_rows[event_id_clean].append({
                        'event_type': event_type.strip() if not is_blank(event_type) else '',
                        'csi_category': csi_category.strip() if not is_blank(csi_category) else ''
                    })
                    
                    # Step 1: Check if row has EVENT_TYPE but NO CSI_CATEGORY
                    if not is_blank(event_type) and is_blank(csi_category):
                        event_ids_with_type_no_csi.add(event_id_clean)
            
            print(f"   Total rows read: {row_count}")
            print(f"   Unique EVENT_UNIQUE_IDs: {len(event_id_to_rows)}")
            print(f"   EVENT_UNIQUE_IDs with EVENT_TYPE but no CSI_CATEGORY: {len(event_ids_with_type_no_csi)}")
            
    except Exception as e:
        print(f"An error occurred during Pass 1: {e}")
        return

    # Step 2 & 3: Find EVENT_UNIQUE_IDs that appear more than once
    print("\n" + "=" * 80)
    print("PASS 2: Finding EVENT_UNIQUE_IDs that appear more than once")
    print("=" * 80)
    
    event_ids_multiple_occurrences = []
    for event_id in event_ids_with_type_no_csi:
        if len(event_id_to_rows[event_id]) > 1:
            event_ids_multiple_occurrences.append(event_id)
    
    print(f"   EVENT_UNIQUE_IDs with EVENT_TYPE but no CSI_CATEGORY: {len(event_ids_with_type_no_csi)}")
    print(f"   Of those, appearing more than once: {len(event_ids_multiple_occurrences)}")
    
    # Step 4: Print details for each EVENT_UNIQUE_ID that appears multiple times
    print("\n" + "=" * 80)
    print("PASS 3: Detailed view of EVENT_UNIQUE_IDs appearing multiple times")
    print("=" * 80)
    
    csi_counter = Counter()
    
    if event_ids_multiple_occurrences:
        for event_id in event_ids_multiple_occurrences:
            rows = event_id_to_rows[event_id]
            print(f"\n{'EVENT_UNIQUE_ID:':<20} {event_id}")
            print(f"{'Total occurrences:':<20} {len(rows)}")
            print("-" * 80)
            print(f"{'Row #':<10} | {'EVENT_TYPE':<25} | {'CSI_CATEGORY':<40}")
            print("-" * 80)
            
            for i, row_data in enumerate(rows, start=1):
                event_type = row_data['event_type'] if row_data['event_type'] else '(empty)'
                csi_category = row_data['csi_category'] if row_data['csi_category'] else '(empty)'
                print(f"{i:<10} | {event_type:<25} | {csi_category:<40}")
                
                # Count CSI_CATEGORY (only if not empty)
                if row_data['csi_category']:
                    csi_counter[row_data['csi_category']] += 1
            
            print("-" * 80)
    else:
        print("\n   No EVENT_UNIQUE_IDs found that appear more than once.")
    
    # Step 5: Return all unique CSI_CATEGORY and their counts
    print("\n" + "=" * 80)
    print("FINAL RESULTS: Unique CSI_CATEGORY Values and Counts")
    print("=" * 80)
    
    if csi_counter:
        print(f"\n{'CSI_CATEGORY':<50} | {'Count':<10}")
        print("-" * 65)
        
        total_csi_entries = sum(csi_counter.values())
        
        for category, count in csi_counter.most_common():
            print(f"{category:<50} | {count:<10}")
            
        print("-" * 65)
        print(f"{'TOTAL':<50} | {total_csi_entries:<10}")
        print(f"\nUnique CSI_CATEGORY values: {len(csi_counter)}")
        print(f"Total CSI_CATEGORY entries: {total_csi_entries}")
    else:
        print("\n   No CSI_CATEGORY values found in the matching rows.")
    
    print("=" * 80)

if __name__ == "__main__":
    analyze_event_csi_relationship(file_path)