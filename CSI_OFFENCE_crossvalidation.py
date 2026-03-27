import pandas as pd
import numpy as np
from collections import defaultdict
import os
from tqdm import tqdm

# For visualizations
from plotnine import ggplot, aes, geom_bar, geom_tile, geom_text, theme, element_text, labs, coord_flip, scale_fill_gradient

def read_csv_file(filepath):
    """Read the CSV file and return a pandas DataFrame"""
    try:
        # Get file size for progress indication
        file_size = os.path.getsize(filepath)
        print(f"File size: {file_size / (1024*1024):.2f} MB")
        
        # Read the CSV file with tqdm progress
        with tqdm(total=100, desc="Reading CSV file", unit="%") as pbar:
            df = pd.read_csv(filepath)
            pbar.update(100)
        
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def get_unique_values_and_counts(df):
    """Get all unique CSI_CATEGORY, unique OFFENCE, and their counts"""
    
    print("=" * 60)
    print("UNIQUE CSI_CATEGORY VALUES:")
    print("=" * 60)
    
    # Handle CSI_CATEGORY (may have NaN values)
    with tqdm(total=2, desc="Processing CSI_CATEGORY", unit="step") as pbar:
        csi_categories = df['CSI_CATEGORY'].dropna()
        pbar.update(1)
        
        if len(csi_categories) > 0:
            unique_csi = csi_categories.unique()
            csi_counts = csi_categories.value_counts()
            pbar.update(1)
            
            for category in unique_csi:
                print(f"  {category}: {csi_counts[category]} occurrences")
        else:
            print("  No CSI_CATEGORY data found")
            unique_csi = []
            csi_counts = pd.Series(dtype=int)
            pbar.update(1)
    
    print("\n" + "=" * 60)
    print("UNIQUE OFFENCE VALUES:")
    print("=" * 60)
    
    # Handle OFFENCE (may have NaN values)
    with tqdm(total=2, desc="Processing OFFENCE", unit="step") as pbar:
        offences = df['OFFENCE'].dropna()
        pbar.update(1)
        
        if len(offences) > 0:
            unique_offence = offences.unique()
            offence_counts = offences.value_counts()
            pbar.update(1)
            
            for offence in unique_offence:
                print(f"  {offence}: {offence_counts[offence]} occurrences")
        else:
            print("  No OFFENCE data found")
            unique_offence = []
            offence_counts = pd.Series(dtype=int)
            pbar.update(1)
    
    return unique_csi, unique_offence, csi_counts, offence_counts

def get_intersection_counts(df, min_threshold=15):
    """Get all unique pairs of CSI_CATEGORY and OFFENCE with their counts"""
    
    print("\n" + "=" * 60)
    print("CSI_CATEGORY AND OFFENCE INTERSECTIONS:")
    print("=" * 60)
    
    # Drop rows where either CSI_CATEGORY or OFFENCE is NaN
    with tqdm(total=3, desc="Preparing data", unit="step") as pbar:
        valid_rows = df.dropna(subset=['CSI_CATEGORY', 'OFFENCE'])
        pbar.update(1)
        
        if len(valid_rows) == 0:
            print("No valid pairs found (both CSI_CATEGORY and OFFENCE need to be present)")
            return None, None, None
        
        # Get all unique values
        unique_csi = valid_rows['CSI_CATEGORY'].unique()
        unique_offence = valid_rows['OFFENCE'].unique()
        pbar.update(1)
        
        # Count occurrences of each pair with progress bar
        pair_counts = defaultdict(int)
        pbar.update(1)
    
    # Count occurrences with progress bar
    print("\nCounting pair occurrences...")
    for _, row in tqdm(valid_rows.iterrows(), total=len(valid_rows), desc="Processing rows", unit="rows"):
        pair = (row['CSI_CATEGORY'], row['OFFENCE'])
        pair_counts[pair] += 1
    
    # Filter pairs with count >= min_threshold
    print("\nFiltering pairs...")
    filtered_counts = {}
    for pair, count in tqdm(pair_counts.items(), desc="Applying threshold", unit="pairs"):
        if count >= min_threshold:
            filtered_counts[pair] = count
    
    print(f"\nFiltering pairs with less than {min_threshold} instances...")
    print(f"Original number of unique pairs: {len(pair_counts)}")
    print(f"Pairs after filtering (>= {min_threshold}): {len(filtered_counts)}")
    
    # Convert to DataFrame for easier plotting
    print("\nCreating DataFrame...")
    pair_data = []
    for (csi, offence), count in tqdm(filtered_counts.items(), desc="Building pair data", unit="pairs"):
        pair_data.append({
            'CSI_CATEGORY': csi,
            'OFFENCE': offence,
            'COUNT': count
        })
    
    pair_df = pd.DataFrame(pair_data)
    
    # Display all pairs that meet the threshold
    print(f"\nPairs with at least {min_threshold} occurrences:")
    print("-" * 50)
    
    # Sort by count descending
    sorted_pairs = sorted(filtered_counts.items(), key=lambda x: x[1], reverse=True)
    
    for (csi, offence), count in sorted_pairs:
        print(f"  ({csi}, {offence}): {count} occurrences")
    
    return pair_counts, pair_df, (unique_csi, unique_offence)

def create_ggplot_visualizations(csi_counts, offence_counts, pair_df, min_threshold=15):
    """Create visualizations using plotnine (ggplot for Python)"""
    
    print("\n" + "=" * 60)
    print("CREATING GGPLOT VISUALIZATIONS...")
    print("=" * 60)
    
    # Plot 1: CSI_CATEGORY bar chart
    if len(csi_counts) > 0:
        with tqdm(total=1, desc="Creating CSI categories plot", unit="plot") as pbar:
            csi_df = csi_counts.reset_index()
            csi_df.columns = ['CSI_CATEGORY', 'COUNT']
            csi_df = csi_df.sort_values('COUNT', ascending=False)
            
            # Convert COUNT to string for display
            csi_df['COUNT_LABEL'] = csi_df['COUNT'].astype(str)
            
            p1 = (ggplot(csi_df, aes(x='reorder(CSI_CATEGORY, -COUNT)', y='COUNT', fill='CSI_CATEGORY'))
                  + geom_bar(stat='identity')
                  + geom_text(aes(label='COUNT_LABEL'), size=10, va='bottom', ha='center', nudge_y=0.5)
                  + theme(axis_text_x=element_text(rotation=45, ha='right'), figure_size=(12, 6))
                  + labs(title='CSI Categories Distribution',
                         x='CSI Category',
                         y='Count')
                  + theme(legend_position='none')
                  )
            p1.save('ggplot_csi_categories.png', dpi=300)
            pbar.update(1)
            print("✓ Saved: ggplot_csi_categories.png")
            print(p1)
    else:
        print("⚠ No CSI_CATEGORY data to plot")
    
    # Plot 2: OFFENCE bar chart (horizontal for better readability)
    if len(offence_counts) > 0:
        with tqdm(total=1, desc="Creating offences plot", unit="plot") as pbar:
            offence_df = offence_counts.reset_index()
            offence_df.columns = ['OFFENCE', 'COUNT']
            offence_df = offence_df.sort_values('COUNT', ascending=False)
            
            # Convert COUNT to string for display
            offence_df['COUNT_LABEL'] = offence_df['COUNT'].astype(str)
            
            # For horizontal bars, add text labels to the bars
            p2 = (ggplot(offence_df, aes(x='reorder(OFFENCE, COUNT)', y='COUNT', fill='OFFENCE'))
                  + geom_bar(stat='identity')
                  + geom_text(aes(label='COUNT_LABEL'), size=10, ha='left', va='center', nudge_x=0.5)
                  + coord_flip()
                  + theme(figure_size=(10, max(6, len(offence_df) * 0.3)))
                  + labs(title='Offence Distribution',
                         x='Offence',
                         y='Count')
                  + theme(legend_position='none')
                  )
            p2.save('ggplot_offences.png', dpi=300)
            pbar.update(1)
            print("✓ Saved: ggplot_offences.png")
            print(p2)
    else:
        print("⚠ No OFFENCE data to plot")
    
    # Plot 3: Pair analysis - Bar chart showing filtered pairs with frequencies
    if pair_df is not None and len(pair_df) > 0:
        with tqdm(total=4, desc="Creating pair visualizations", unit="plot") as pbar:
            # Create a combined label for pairs
            pair_df['PAIR'] = pair_df['CSI_CATEGORY'] + " - " + pair_df['OFFENCE']
            pair_df = pair_df.sort_values('COUNT', ascending=False)
            
            # Convert COUNT to string for display
            pair_df['COUNT_LABEL'] = pair_df['COUNT'].astype(str)
            
            # Bar chart of filtered pairs with count labels
            p3 = (ggplot(pair_df, aes(x='reorder(PAIR, COUNT)', y='COUNT', fill='PAIR'))
                  + geom_bar(stat='identity')
                  + geom_text(aes(label='COUNT_LABEL'), size=10, ha='left', va='center', nudge_x=0.5)
                  + coord_flip()
                  + theme(figure_size=(12, max(8, len(pair_df) * 0.5)), 
                          axis_text_y=element_text(size=10))
                  + labs(title=f'CSI Category - Offence Pairs (≥ {min_threshold} instances)',
                         x='Pair (CSI Category - Offence)',
                         y='Count')
                  + theme(legend_position='none')
                  )
            p3.save('ggplot_pair_frequencies.png', dpi=300)
            pbar.update(1)
            print("✓ Saved: ggplot_pair_frequencies.png")
            print(p3)
            
            # Plot 4: Heatmap of intersections (only filtered pairs)
            # Create a pivot table for the heatmap
            heatmap_data = pair_df.pivot_table(
                values='COUNT', 
                index='CSI_CATEGORY', 
                columns='OFFENCE', 
                fill_value=0,
                aggfunc='sum'
            ).reset_index()
            
            # Melt for ggplot
            heatmap_melted = pd.melt(
                heatmap_data, 
                id_vars=['CSI_CATEGORY'], 
                var_name='OFFENCE', 
                value_name='COUNT'
            )
            
            # Filter out zero counts for cleaner heatmap
            heatmap_melted = heatmap_melted[heatmap_melted['COUNT'] > 0]
            
            # Convert COUNT to string for display in heatmap
            heatmap_melted['COUNT_LABEL'] = heatmap_melted['COUNT'].astype(str)
            
            p4 = (ggplot(heatmap_melted, aes(x='OFFENCE', y='CSI_CATEGORY', fill='COUNT'))
                  + geom_tile()
                  + geom_text(aes(label='COUNT_LABEL'), size=10, color='white')
                  + scale_fill_gradient(low='lightblue', high='darkred')
                  + theme(axis_text_x=element_text(rotation=45, ha='right'), 
                          figure_size=(12, 8),
                          panel_grid_major=element_blank(),
                          panel_grid_minor=element_blank())
                  + labs(title=f'CSI Category vs Offence Intersection Heatmap (≥ {min_threshold} instances)',
                         x='Offence',
                         y='CSI Category',
                         fill='Count')
                  )
            p4.save('ggplot_intersection_heatmap.png', dpi=300)
            pbar.update(1)
            print("✓ Saved: ggplot_intersection_heatmap.png")
            print(p4)
            
            # Plot 5: Additional - Top 15 pairs bar chart for better readability
            top_pairs = pair_df.nlargest(min(15, len(pair_df)), 'COUNT')
            top_pairs['COUNT_LABEL'] = top_pairs['COUNT'].astype(str)
            
            p5 = (ggplot(top_pairs, aes(x='reorder(PAIR, COUNT)', y='COUNT', fill='PAIR'))
                  + geom_bar(stat='identity')
                  + geom_text(aes(label='COUNT_LABEL'), size=10, ha='left', va='center', nudge_x=0.5)
                  + coord_flip()
                  + theme(figure_size=(12, max(6, len(top_pairs) * 0.5)), 
                          axis_text_y=element_text(size=10))
                  + labs(title=f'Top {len(top_pairs)} Most Common CSI Category - Offence Pairs',
                         x='Pair (CSI Category - Offence)',
                         y='Count')
                  + theme(legend_position='none')
                  )
            p5.save('ggplot_top_pairs.png', dpi=300)
            pbar.update(1)
            print("✓ Saved: ggplot_top_pairs.png")
            print(p5)
            
            pbar.update(1)  # Complete the progress bar
            
    else:
        print("⚠ No pair data to plot (or no pairs meet the minimum threshold)")

def main():
    # File path
    filepath = r"project\DB_csv\Open_Consolidated_Data.csv"
    
    # Set minimum threshold for pairs
    MIN_PAIR_THRESHOLD = 15
    
    print("=" * 60)
    print("STARTING DATA ANALYSIS")
    print("=" * 60)
    
    # Check if file exists
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        print("Please ensure the file path is correct.")
        return
    
    # Read the CSV file
    print(f"\nReading file: {filepath}")
    df = read_csv_file(filepath)
    
    if df is None:
        return
    
    print(f"\nTotal rows in dataset: {len(df):,}")
    print(f"Columns: {list(df.columns)}")
    
    # Function 1: Get unique values and counts
    unique_csi, unique_offence, csi_counts, offence_counts = get_unique_values_and_counts(df)
    
    # Function 2: Get intersection counts with threshold
    pair_counts, pair_df, unique_values = get_intersection_counts(df, MIN_PAIR_THRESHOLD)
    
    # Additional summary
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print("=" * 60)
    print(f"Total unique CSI_CATEGORY: {len(unique_csi):,}")
    print(f"Total unique OFFENCE: {len(unique_offence):,}")
    print(f"Total unique pairs (all): {len(pair_counts) if pair_counts else 0:,}")
    print(f"Total unique pairs (≥ {MIN_PAIR_THRESHOLD}): {len(pair_df) if pair_df is not None else 0:,}")
    
    # Create visualizations
    if len(csi_counts) > 0 or len(offence_counts) > 0 or (pair_df is not None and len(pair_df) > 0):
        print("\n" + "=" * 60)
        print("VISUALIZATIONS:")
        print("=" * 60)
        
        # Create ggplot visualizations
        try:
            create_ggplot_visualizations(csi_counts, offence_counts, pair_df, MIN_PAIR_THRESHOLD)
        except Exception as e:
            print(f"Error creating ggplot visualizations: {e}")
            print("Make sure plotnine is installed: pip install plotnine")
    else:
        print("\nNo data available for visualizations.")
    
    # Print all unique pairs that meet the threshold
    if pair_df is not None and len(pair_df) > 0:
        print("\n" + "=" * 60)
        print(f"COMPLETE PAIR ANALYSIS (≥ {MIN_PAIR_THRESHOLD} instances):")
        print("=" * 60)
        print("\nAll unique pairs (CSI_CATEGORY, OFFENCE) with their frequencies:")
        print("-" * 50)
        pair_df_sorted = pair_df.sort_values('COUNT', ascending=False)
        for idx, row in pair_df_sorted.iterrows():
            print(f"  ({row['CSI_CATEGORY']}, {row['OFFENCE']}): {row['COUNT']:,} occurrences")
        
        # Show summary statistics
        print(f"\nSummary Statistics for Filtered Pairs:")
        print("-" * 50)
        print(f"  Minimum count: {pair_df['COUNT'].min():,}")
        print(f"  Maximum count: {pair_df['COUNT'].max():,}")
        print(f"  Average count: {pair_df['COUNT'].mean():.1f}")
        print(f"  Median count: {pair_df['COUNT'].median():,.0f}")
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()