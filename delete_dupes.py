import csv
import os
import hashlib
from datetime import datetime

# Try to import tqdm, but make it optional
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    tqdm = lambda iterable, **kwargs: iterable

# Define the file paths
input_file_path = r"project\DB_csv\Open_Consolidated_Data_updated.csv"
output_txt_path = r"project\DB_csv\deduplication_summary.txt"

def is_blank(value):
    """Helper function to determine if a value is effectively empty."""
    if value is None:
        return True
    return str(value).strip() == ""

def row_to_hash(row, fieldnames):
    """Create a hash of the entire row for duplicate detection."""
    row_string = '|'.join(str(row.get(field, '')) for field in fieldnames)
    return hashlib.md5(row_string.encode('utf-8')).hexdigest()

def remove_row_duplicates(input_filepath, output_txt=None):
    # Check if file exists
    if not os.path.exists(input_filepath):
        print(f"Error: File not found at {input_filepath}")
        return

    # Data structures
    seen_hashes = {}              # Map row hash to first row number where it appeared
    duplicate_info = []           # Store info about duplicate rows
    rows_to_keep = []             # Store rows that will be kept (with row data)
    total_rows = 0
    duplicate_rows = 0
    fieldnames = None
    
    print("=" * 80)
    print("PASS 1: READING CSV AND IDENTIFYING DUPLICATE ROWS")
    print("=" * 80)
    
    try:
        with open(input_filepath, mode='r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            
            if not fieldnames:
                print(f"Error: CSV file appears to be empty or has no headers")
                return
            
            print(f"   Columns found: {len(fieldnames)}")
            print(f"   Column names: {', '.join(fieldnames[:5])}{'...' if len(fieldnames) > 5 else ''}")
            print("")
            
            # Read all rows and identify duplicates
            for row_num, row in enumerate(tqdm(reader, desc="Scanning rows", unit="rows"), start=2):
                total_rows += 1
                row_hash = row_to_hash(row, fieldnames)
                
                if row_hash in seen_hashes:
                    # This is a duplicate row
                    duplicate_rows += 1
                    duplicate_info.append({
                        'duplicate_row_num': row_num,
                        'original_row_num': seen_hashes[row_hash],
                        'row_hash': row_hash
                    })
                else:
                    # This is a unique row (first occurrence)
                    seen_hashes[row_hash] = row_num
                    rows_to_keep.append(row)
            
            print(f"\n   Total rows scanned: {total_rows}")
            print(f"   Unique rows: {len(rows_to_keep)}")
            print(f"   Duplicate rows found: {duplicate_rows}")
            
    except Exception as e:
        print(f"An error occurred during Pass 1: {e}")
        return

    # Calculate statistics
    unique_rows = len(rows_to_keep)
    reduction_percentage = (duplicate_rows / total_rows * 100) if total_rows > 0 else 0
    
    print("\n" + "=" * 80)
    print("DEDUPLICATION SUMMARY")
    print("=" * 80)
    print(f"   Original total rows: {total_rows}")
    print(f"   Unique rows to keep: {unique_rows}")
    print(f"   Duplicate rows to remove: {duplicate_rows}")
    print(f"   Reduction: {reduction_percentage:.2f}%")
    print("=" * 80)
    
    # Show sample of duplicates found
    if duplicate_info:
        print("\n" + "=" * 80)
        print("SAMPLE OF DUPLICATE ROWS FOUND (First 10)")
        print("=" * 80)
        print(f"{'Duplicate Row #':<20} | {'Original Row #':<20} | {'Status'}")
        print("-" * 65)
        
        for dup in duplicate_info[:10]:
            print(f"{dup['duplicate_row_num']:<20} | {dup['original_row_num']:<20} | REMOVE")
        
        if len(duplicate_info) > 10:
            print(f"... and {len(duplicate_info) - 10} more duplicate rows")
        print("-" * 65)
    
    # Ask for confirmation before writing
    print("\n" + "=" * 80)
    if duplicate_rows > 0:
        response = input(f"\nDo you want to remove {duplicate_rows} duplicate rows? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("\nDeduplication cancelled. No changes made.")
            return
    else:
        print("\nNo duplicates found. No action needed.")
        response = 'no'
    
    # Write deduplicated file
    if response in ['yes', 'y'] and duplicate_rows > 0:
        print("\n" + "=" * 80)
        print("PASS 2: WRITING DEDUPLICATED FILE")
        print("=" * 80)
        
        # Create output path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = input_filepath.replace('.csv', f'_deduplicated_{timestamp}.csv')
        
        try:
            with open(output_path, mode='w', encoding='utf-8', newline='') as outfile:
                writer = csv.DictWriter(outfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for row in tqdm(rows_to_keep, desc="Writing rows", unit="rows"):
                    writer.writerow(row)
            
            print(f"\n   ✓ Deduplicated file created: {output_path}")
            print(f"   ✓ Original file preserved: {input_filepath}")
            print(f"   ✓ Rows removed: {duplicate_rows}")
            print(f"   ✓ Rows in new file: {unique_rows}")
            
        except Exception as e:
            print(f"   ✗ Error during file write: {e}")
            return
    else:
        output_path = None
        unique_rows = total_rows
    
    # Prepare summary for TXT file
    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append("ROW DEDUPLICATION SUMMARY")
    output_lines.append("=" * 80)
    output_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output_lines.append(f"Source File: {input_filepath}")
    output_lines.append(f"Output File: {output_path if output_path else 'N/A (no changes made)'}")
    output_lines.append("")
    output_lines.append("STATISTICS")
    output_lines.append("-" * 80)
    output_lines.append(f"Original total rows: {total_rows}")
    output_lines.append(f"Unique rows kept: {unique_rows}")
    output_lines.append(f"Duplicate rows removed: {duplicate_rows}")
    output_lines.append(f"Reduction percentage: {reduction_percentage:.2f}%")
    output_lines.append("")
    output_lines.append("DUPLICATE ROW DETAILS")
    output_lines.append("-" * 80)
    
    if duplicate_info:
        output_lines.append(f"{'Duplicate Row #':<20} | {'Original Row #':<20} | {'Action'}")
        output_lines.append("-" * 65)
        
        for dup in duplicate_info:
            output_lines.append(f"{dup['duplicate_row_num']:<20} | {dup['original_row_num']:<20} | REMOVED")
    else:
        output_lines.append("No duplicate rows found.")
    
    output_lines.append("")
    output_lines.append("=" * 80)
    
    # Write to TXT file
    if output_txt:
        try:
            with open(output_txt, mode='w', encoding='utf-8') as f:
                f.write('\n'.join(output_lines))
            print(f"\n✓ Summary saved to: {output_txt}")
        except Exception as e:
            print(f"\n✗ Error saving to TXT file: {e}")
    
    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"   BEFORE: {total_rows} rows")
    print(f"   AFTER:  {unique_rows} rows")
    print(f"   REMOVED: {duplicate_rows} duplicate rows ({reduction_percentage:.2f}%)")
    print(f"   Original file unchanged: {input_filepath}")
    if output_path:
        print(f"   New file created: {output_path}")
    print("=" * 80)
    
    return {
        'total_rows': total_rows,
        'unique_rows': unique_rows,
        'duplicate_rows': duplicate_rows,
        'reduction_percentage': reduction_percentage,
        'output_file': output_path
    }

if __name__ == "__main__":
    remove_row_duplicates(input_file_path, output_txt_path)