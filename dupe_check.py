import csv
import os
import hashlib
from collections import Counter, defaultdict
from datetime import datetime

# Define the file path
file_path = r"project\DB_csv\Open_Consolidated_Data_updated.csv"
output_txt_path = r"project\DB_csv\duplicate_summary.txt"

def is_blank(value):
    """Helper function to determine if a value is effectively empty."""
    if value is None:
        return True
    return str(value).strip() == ""

def row_to_hash(row, fieldnames):
    """Create a hash of the entire row for full row duplicate detection."""
    row_string = '|'.join(str(row.get(field, '')) for field in fieldnames)
    return hashlib.md5(row_string.encode('utf-8')).hexdigest()

def find_duplicates(filepath, output_txt=None):
    # Check if file exists
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return

    # Data structures
    event_id_counter = Counter()           # Count occurrences of each EVENT_UNIQUE_ID
    event_id_to_rows = defaultdict(list)   # Store row numbers for each EVENT_UNIQUE_ID
    row_hash_counter = Counter()           # Count occurrences of each full row hash
    row_hash_to_rows = defaultdict(list)   # Store row numbers for each row hash
    total_rows = 0
    fieldnames = None
    
    print("=" * 80)
    print("READING CSV FILE AND COUNTING DUPLICATES")
    print("=" * 80)
    
    try:
        with open(filepath, mode='r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            
            required_cols = {'EVENT_UNIQUE_ID'}
            if not required_cols.issubset(set(fieldnames)):
                print(f"Error: CSV is missing required columns. Found: {fieldnames}")
                return
            
            # Count occurrences
            for row_num, row in enumerate(reader, start=2):
                total_rows += 1
                event_id = row.get('EVENT_UNIQUE_ID', '')
                
                # Track EVENT_UNIQUE_ID duplicates
                if not is_blank(event_id):
                    event_id_clean = event_id.strip()
                    event_id_counter[event_id_clean] += 1
                    event_id_to_rows[event_id_clean].append(row_num)
                
                # Track full row duplicates
                row_hash = row_to_hash(row, fieldnames)
                row_hash_counter[row_hash] += 1
                row_hash_to_rows[row_hash].append(row_num)
            
            print(f"   Total rows read: {total_rows}")
            print(f"   Unique EVENT_UNIQUE_IDs: {len(event_id_counter)}")
            print(f"   Unique row hashes: {len(row_hash_counter)}")
            
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return

    # Find EVENT_UNIQUE_ID duplicates
    event_id_duplicates = {event_id: count for event_id, count in event_id_counter.items() if count > 1}
    event_id_unique = {event_id: count for event_id, count in event_id_counter.items() if count == 1}
    
    # Find full row duplicates
    row_duplicates = {row_hash: count for row_hash, count in row_hash_counter.items() if count > 1}
    row_unique = {row_hash: count for row_hash, count in row_hash_counter.items() if count == 1}
    
    # Calculate statistics
    total_event_id_duplicate_ids = len(event_id_duplicates)
    total_event_id_duplicate_rows = sum(event_id_duplicates.values())
    total_event_id_unique_ids = len(event_id_unique)
    
    total_row_duplicate_hashes = len(row_duplicates)
    total_row_duplicate_rows = sum(row_duplicates.values())
    total_row_unique_hashes = len(row_unique)
    
    # Rows that are "extra" (could be removed)
    extra_event_id_rows = total_event_id_duplicate_rows - total_event_id_duplicate_ids
    extra_row_duplicates = total_row_duplicate_rows - total_row_duplicate_hashes
    
    print("\n" + "=" * 80)
    print("DUPLICATE ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"   Total rows in file: {total_rows}")
    print("")
    print("   --- EVENT_UNIQUE_ID Duplicates ---")
    print(f"   EVENT_UNIQUE_IDs appearing only once: {total_event_id_unique_ids}")
    print(f"   EVENT_UNIQUE_IDs with duplicates: {total_event_id_duplicate_ids}")
    print(f"   Total rows for duplicate EVENT_UNIQUE_IDs: {total_event_id_duplicate_rows}")
    print(f"   Extra rows (could be removed): {extra_event_id_rows}")
    print("")
    print("   --- Full Row Duplicates (All Columns Identical) ---")
    print(f"   Unique rows: {total_row_unique_hashes}")
    print(f"   Rows with exact duplicates: {total_row_duplicate_hashes}")
    print(f"   Total rows that are exact duplicates: {total_row_duplicate_rows}")
    print(f"   Extra rows (could be removed): {extra_row_duplicates}")
    print("=" * 80)
    
    # Prepare output content
    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append("DUPLICATE ANALYSIS SUMMARY")
    output_lines.append("=" * 80)
    output_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output_lines.append(f"Source File: {filepath}")
    output_lines.append("")
    output_lines.append(f"Total rows in file: {total_rows}")
    output_lines.append("")
    output_lines.append("--- EVENT_UNIQUE_ID Duplicates ---")
    output_lines.append(f"EVENT_UNIQUE_IDs appearing only once: {total_event_id_unique_ids}")
    output_lines.append(f"EVENT_UNIQUE_IDs with duplicates: {total_event_id_duplicate_ids}")
    output_lines.append(f"Total rows for duplicate EVENT_UNIQUE_IDs: {total_event_id_duplicate_rows}")
    output_lines.append(f"Extra rows (could be removed): {extra_event_id_rows}")
    output_lines.append("")
    output_lines.append("--- Full Row Duplicates (All Columns Identical) ---")
    output_lines.append(f"Unique rows: {total_row_unique_hashes}")
    output_lines.append(f"Rows with exact duplicates: {total_row_duplicate_hashes}")
    output_lines.append(f"Total rows that are exact duplicates: {total_row_duplicate_rows}")
    output_lines.append(f"Extra rows (could be removed): {extra_row_duplicates}")
    output_lines.append("")
    
    # Show EVENT_UNIQUE_ID duplicate details
    if event_id_duplicates:
        print("\n" + "=" * 80)
        print("EVENT_UNIQUE_ID DUPLICATES (Sorted by Count)")
        print("=" * 80)
        
        output_lines.append("=" * 80)
        output_lines.append("EVENT_UNIQUE_ID DUPLICATES (Sorted by Count)")
        output_lines.append("=" * 80)
        output_lines.append(f"{'EVENT_UNIQUE_ID':<30} | {'Count':<10} | {'Row Numbers'}")
        output_lines.append("-" * 80)
        
        print(f"\n{'EVENT_UNIQUE_ID':<30} | {'Count':<10} | {'Row Numbers'}")
        print("-" * 80)
        
        for event_id, count in sorted(event_id_duplicates.items(), key=lambda x: x[1], reverse=True):
            row_nums = event_id_to_rows[event_id]
            row_nums_str = ', '.join(map(str, row_nums[:10]))
            if len(row_nums) > 10:
                row_nums_str += f' ... and {len(row_nums) - 10} more'
            
            print(f"{event_id:<30} | {count:<10} | {row_nums_str}")
            output_lines.append(f"{event_id:<30} | {count:<10} | {row_nums_str}")
        
        print("-" * 80)
        output_lines.append("-" * 80)
    
    # Show full row duplicate details
    if row_duplicates:
        print("\n" + "=" * 80)
        print("FULL ROW DUPLICATES (All Columns Identical)")
        print("=" * 80)
        
        output_lines.append("")
        output_lines.append("=" * 80)
        output_lines.append("FULL ROW DUPLICATES (All Columns Identical)")
        output_lines.append("=" * 80)
        output_lines.append(f"{'Row Hash':<35} | {'Count':<10} | {'Row Numbers'}")
        output_lines.append("-" * 80)
        
        print(f"\n{'Row Hash':<35} | {'Count':<10} | {'Row Numbers'}")
        print("-" * 80)
        
        for row_hash, count in sorted(row_duplicates.items(), key=lambda x: x[1], reverse=True):
            row_nums = row_hash_to_rows[row_hash]
            row_nums_str = ', '.join(map(str, row_nums[:10]))
            if len(row_nums) > 10:
                row_nums_str += f' ... and {len(row_nums) - 10} more'
            
            print(f"{row_hash[:35]:<35} | {count:<10} | {row_nums_str}")
            output_lines.append(f"{row_hash[:35]:<35} | {count:<10} | {row_nums_str}")
        
        print("-" * 80)
        output_lines.append("-" * 80)
    
    # Write to TXT file
    if output_txt:
        try:
            with open(output_txt, mode='w', encoding='utf-8') as f:
                f.write('\n'.join(output_lines))
            print(f"\n✓ Summary saved to: {output_txt}")
        except Exception as e:
            print(f"\n✗ Error saving to TXT file: {e}")
    
    print("=" * 80)
    
    return {
        'total_rows': total_rows,
        'event_id_duplicates': event_id_duplicates,
        'row_duplicates': row_duplicates
    }

if __name__ == "__main__":
    find_duplicates(file_path, output_txt_path)