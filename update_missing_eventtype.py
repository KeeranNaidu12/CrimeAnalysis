import csv
import os
from collections import Counter, defaultdict
from datetime import datetime

# Define the file path
file_path = r"project\DB_csv\Open_Consolidated_Data.csv"

def is_blank(value):
    """Helper function to determine if a value is effectively empty."""
    if value is None:
        return True
    return str(value).strip() == ""

def analyze_and_update_event_csi_relationship(filepath):
    # Check if file exists
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return

    # Data structures
    event_id_to_rows = defaultdict(list)      # Store all rows for each EVENT_UNIQUE_ID
    event_id_to_event_type = {}               # Map EVENT_UNIQUE_ID to its EVENT_TYPE
    event_ids_with_type_no_csi = set()        # EVENT_UNIQUE_IDs that have EVENT_TYPE but no CSI_CATEGORY
    rows_to_update = []                       # Store rows that need EVENT_TYPE updates
    rows_to_delete = []                       # Store rows that have EVENT_TYPE but no CSI_CATEGORY
    all_rows = []                             # Store all rows for final output
    
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
            
            fieldnames = reader.fieldnames
            row_count = 0
            
            for row_num, row in enumerate(reader, start=2):  # Start at 2 (row 1 is header)
                row_count += 1
                event_id = row.get('EVENT_UNIQUE_ID', '')
                event_type = row.get('EVENT_TYPE', '')
                csi_category = row.get('CSI_CATEGORY', '')
                
                # Store all rows for final output
                all_rows.append({
                    'row_num': row_num,
                    'row_data': row,
                    'event_id': event_id.strip() if not is_blank(event_id) else '',
                    'event_type': event_type.strip() if not is_blank(event_type) else '',
                    'csi_category': csi_category.strip() if not is_blank(csi_category) else ''
                })
                
                if not is_blank(event_id):
                    event_id_clean = event_id.strip()
                    
                    # Store the entire row for this EVENT_UNIQUE_ID
                    event_id_to_rows[event_id_clean].append({
                        'row_num': row_num,
                        'event_type': event_type.strip() if not is_blank(event_type) else '',
                        'csi_category': csi_category.strip() if not is_blank(csi_category) else '',
                        'original_row': row
                    })
                    
                    # Step 1: Check if row has EVENT_TYPE but NO CSI_CATEGORY
                    if not is_blank(event_type) and is_blank(csi_category):
                        event_ids_with_type_no_csi.add(event_id_clean)
                        # Store the EVENT_TYPE for this EVENT_UNIQUE_ID
                        if event_id_clean not in event_id_to_event_type:
                            event_id_to_event_type[event_id_clean] = event_type.strip()
                        # Mark this row for deletion
                        rows_to_delete.append(row_num)
            
            print(f"   Total rows read: {row_count}")
            print(f"   Unique EVENT_UNIQUE_IDs: {len(event_id_to_rows)}")
            print(f"   EVENT_UNIQUE_IDs with EVENT_TYPE but no CSI_CATEGORY: {len(event_ids_with_type_no_csi)}")
            print(f"   EVENT_UNIQUE_IDs mapped to EVENT_TYPE: {len(event_id_to_event_type)}")
            print(f"   Rows with EVENT_TYPE but no CSI_CATEGORY (to be deleted): {len(rows_to_delete)}")
            
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
    
    # Step 4: Identify rows that need updating
    print("\n" + "=" * 80)
    print("PASS 3: Identifying rows that need EVENT_TYPE updates")
    print("=" * 80)
    
    csi_counter = Counter()
    
    if event_ids_multiple_occurrences:
        for event_id in event_ids_multiple_occurrences:
            rows = event_id_to_rows[event_id]
            correct_event_type = event_id_to_event_type.get(event_id, '')
            
            for row_data in rows:
                # Count CSI_CATEGORY (only if not empty)
                if row_data['csi_category']:
                    csi_counter[row_data['csi_category']] += 1
                
                # Check if this row needs updating: has CSI_CATEGORY but no/wrong EVENT_TYPE
                if row_data['csi_category'] and (is_blank(row_data['event_type']) or row_data['event_type'] != correct_event_type):
                    if correct_event_type:  # Only update if we know the correct EVENT_TYPE
                        rows_to_update.append({
                            'row_num': row_data['row_num'],
                            'event_id': event_id,
                            'old_event_type': row_data['event_type'],
                            'new_event_type': correct_event_type,
                            'csi_category': row_data['csi_category']
                        })
        
        print(f"   Rows identified for EVENT_TYPE update: {len(rows_to_update)}")
    else:
        print("\n   No EVENT_UNIQUE_IDs found that appear more than once.")
    
    # Step 5: Print details for each EVENT_UNIQUE_ID that appears multiple times
    print("\n" + "=" * 80)
    print("PASS 4: Detailed view of EVENT_UNIQUE_IDs appearing multiple times")
    print("=" * 80)
    
    if event_ids_multiple_occurrences:
        for event_id in event_ids_multiple_occurrences:
            rows = event_id_to_rows[event_id]
            correct_event_type = event_id_to_event_type.get(event_id, '')
            
            print(f"\n{'EVENT_UNIQUE_ID:':<20} {event_id}")
            print(f"{'Correct EVENT_TYPE:':<20} {correct_event_type}")
            print(f"{'Total occurrences:':<20} {len(rows)}")
            print("-" * 80)
            print(f"{'Row #':<10} | {'EVENT_TYPE':<25} | {'CSI_CATEGORY':<40} | {'Action':<10}")
            print("-" * 80)
            
            for row_data in rows:
                event_type = row_data['event_type'] if row_data['event_type'] else '(empty)'
                csi_category = row_data['csi_category'] if row_data['csi_category'] else '(empty)'
                
                # Determine action for this row
                if is_blank(csi_category) and not is_blank(event_type):
                    action = 'DELETE'
                elif row_data['csi_category'] and (is_blank(row_data['event_type']) or row_data['event_type'] != correct_event_type) and correct_event_type:
                    action = 'UPDATE'
                else:
                    action = 'KEEP'
                
                print(f"{row_data['row_num']:<10} | {event_type:<25} | {csi_category:<40} | {action:<10}")
            
            print("-" * 80)
    else:
        print("\n   No EVENT_UNIQUE_IDs found that appear more than once.")
    
    # Step 6: Show preview of changes
    print("\n" + "=" * 80)
    print("PASS 5: Preview of Changes")
    print("=" * 80)
    
    # Show rows to update
    if rows_to_update:
        print(f"\n--- ROWS TO UPDATE ({len(rows_to_update)} total) ---")
        print(f"{'Row #':<10} | {'EVENT_UNIQUE_ID':<20} | {'CSI_CATEGORY':<25} | {'Old EVENT_TYPE':<20} | {'New EVENT_TYPE':<20}")
        print("-" * 105)
        
        for update in rows_to_update[:10]:  # Show first 10
            old_type = update['old_event_type'] if update['old_event_type'] else '(empty)'
            print(f"{update['row_num']:<10} | {update['event_id']:<20} | {update['csi_category']:<25} | {old_type:<20} | {update['new_event_type']:<20}")
        
        if len(rows_to_update) > 10:
            print(f"... and {len(rows_to_update) - 10} more rows")
    else:
        print("\n   No rows need updating.")
    
    # Show rows to delete
    if rows_to_delete:
        print(f"\n--- ROWS TO DELETE ({len(rows_to_delete)} total) ---")
        print(f"Rows with EVENT_TYPE but no CSI_CATEGORY that will be removed")
        print(f"{'Row #':<10} | {'EVENT_UNIQUE_ID':<20} | {'EVENT_TYPE':<25}")
        print("-" * 60)
        
        # Show first 10 rows to delete
        delete_count = 0
        for row_info in all_rows:
            if row_info['row_num'] in rows_to_delete and delete_count < 10:
                print(f"{row_info['row_num']:<10} | {row_info['event_id']:<20} | {row_info['event_type']:<25}")
                delete_count += 1
        
        if len(rows_to_delete) > 10:
            print(f"... and {len(rows_to_delete) - 10} more rows")
    else:
        print("\n   No rows marked for deletion.")
    
    # Step 7: Ask for confirmation before writing
    print("\n" + "=" * 80)
    total_changes = len(rows_to_update) + len(rows_to_delete)
    if total_changes > 0:
        response = input(f"\nDo you want to apply {len(rows_to_update)} updates and delete {len(rows_to_delete)} rows? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("\nUpdate cancelled. No changes made.")
            return
    else:
        print("\nNo updates or deletions needed.")
        response = 'no'
    
    # Step 8: Write updated file (NO BACKUP NEEDED SINCE ORIGINAL IS UNTOUCHED)
    if response in ['yes', 'y'] and total_changes > 0:
        print("\n" + "=" * 80)
        print("PASS 6: Writing updated file")
        print("=" * 80)
        
        # Create output path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = filepath.replace('.csv', f'_updated_{timestamp}.csv')
        
        try:
            # Create set of rows to update and delete for fast lookup
            rows_to_update_set = {u['row_num']: u['new_event_type'] for u in rows_to_update}
            rows_to_delete_set = set(rows_to_delete)
            
            # Write updated file
            with open(filepath, mode='r', encoding='utf-8-sig') as infile, \
                 open(output_path, mode='w', encoding='utf-8', newline='') as outfile:
                
                reader = csv.DictReader(infile)
                writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
                writer.writeheader()
                
                update_count = 0
                delete_count = 0
                keep_count = 0
                
                for row_num, row in enumerate(reader, start=2):
                    if row_num in rows_to_delete_set:
                        # Skip this row (delete it)
                        delete_count += 1
                    else:
                        if row_num in rows_to_update_set:
                            # Update EVENT_TYPE
                            row['EVENT_TYPE'] = rows_to_update_set[row_num]
                            update_count += 1
                        keep_count += 1
                        writer.writerow(row)
            
            print(f"   ✓ Updated file created: {output_path}")
            print(f"   ✓ Rows updated: {update_count}")
            print(f"   ✓ Rows deleted: {delete_count}")
            print(f"   ✓ Rows kept: {keep_count}")
            print(f"   ✓ Total rows in new file: {keep_count}")
            print(f"   ✓ Original file preserved: {filepath}")
            
        except Exception as e:
            print(f"   ✗ Error during file update: {e}")
            return
    else:
        update_count = 0
        delete_count = 0
        keep_count = len(all_rows)
    
    # Step 9: Output Final Analysis Results
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
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"   Original total rows: {len(all_rows)}")
    print(f"   Rows with EVENT_TYPE but no CSI_CATEGORY (at start): {len(rows_to_delete)}")
    print(f"   Rows updated with correct EVENT_TYPE: {update_count if response in ['yes', 'y'] else 0}")
    print(f"   Rows deleted (EVENT_TYPE without CSI_CATEGORY): {delete_count if response in ['yes', 'y'] else 0}")
    print(f"   Rows remaining after update: {keep_count if response in ['yes', 'y'] else len(all_rows)}")
    print(f"   Unique CSI_CATEGORY values found: {len(csi_counter)}")
    print("=" * 80)
    
    # Show before/after comparison
    print("\n" + "=" * 80)
    print("BEFORE vs AFTER COMPARISON")
    print("=" * 80)
    if response in ['yes', 'y']:
        print(f"   BEFORE: {len(all_rows)} total rows ({len(rows_to_delete)} with EVENT_TYPE but no CSI_CATEGORY)")
        print(f"   AFTER:  {keep_count} total rows (0 with EVENT_TYPE but no CSI_CATEGORY)")
        print(f"   Reduction: {len(rows_to_delete)} rows removed ({len(rows_to_delete)/len(all_rows)*100:.2f}%)")
        print(f"   Note: Original file remains unchanged and acts as backup.")
    else:
        print(f"   No changes applied (update cancelled)")
    print("=" * 80)

if __name__ == "__main__":
    analyze_and_update_event_csi_relationship(file_path)