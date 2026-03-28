#!/usr/bin/env python3
"""
db_query.py - Interactive query and visualization tool for Toronto Crime Database
"""
import os
import warnings
import psycopg2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm, Normalize, LinearSegmentedColormap
from dotenv import load_dotenv
from datetime import datetime
from textwrap import wrap

warnings.filterwarnings('ignore', category=UserWarning, 
                       message='pandas only supports SQLAlchemy')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 120)

load_dotenv()

DB_CONFIG = {
    'dbname': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'host': os.getenv('DB_HOST'),
    'port': os.getenv('DB_PORT')
}

RECORD_SEPARATOR = "*-" * 25
HEADER_SEPARATOR = "-" * 70

STANDARD_HEAT_CMAP = LinearSegmentedColormap.from_list(
    'standard_heat',
    ['#FFFFFF', '#FFFFCC', '#FFEB99', '#FFCC33', '#FF9900', '#FF6600', '#FF3300', '#CC0000'],
    N=256
)


def get_db_connection():
    try:
        return psycopg2.connect(**DB_CONFIG)
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None


def format_value(value):
    if pd.isna(value) or value is None:
        return "-"
    if isinstance(value, datetime):
        return value.strftime('%Y-%m-%d %H:%M:%S')
    if isinstance(value, bool):
        return "Yes" if value else "No"
    return str(value)


def print_search_results(df, search_term, search_type="exact"):
    print(f"\n{'='*80}")
    print(f"  SEARCH RESULTS | {search_type.upper()} MATCH for: '{search_term}'")
    print(f"{'='*80}")
    
    if df is None or df.empty:
        print(f"\n  No records found matching your search.\n{'='*80}\n")
        return
    
    print(f"\n  Total records found: {len(df):,}\n{'-'*40}")
    
    for _, row in df.iterrows():
        for col in df.columns:
            col_name = col.replace('_', ' ').title()
            print(f"  {col_name}: {format_value(row[col])}")
        print(RECORD_SEPARATOR)
    
    if search_type == "partial" and len(df) > 0:
        print(f"\n  Tip: Use more specific search terms to narrow results.")
    print(f"{'='*80}\n")


def search_by_event_id(conn, table_name, event_id, exact=True):
    try:
        cursor = conn.cursor()
        query = f"SELECT * FROM {table_name} WHERE event_unique_id {'=' if exact else 'LIKE'} %s"
        cursor.execute(query, (event_id if exact else f'%{event_id}%',))
        
        columns = [desc[0] for desc in cursor.description]
        results = cursor.fetchall()
        cursor.close()
        
        if results:
            print_search_results(pd.DataFrame(results, columns=columns), event_id, 
                               "exact" if exact else "partial")
        else:
            print_search_results(None, event_id, "exact" if exact else "partial")
    except Exception as e:
        print(f"Error searching: {e}")


def plot_neighbourhood_crime_histogram(conn, show_all=True):
    try:
        df = pd.read_sql_query("""
            SELECT neighbourhood_158, COUNT(*) as crime_count
            FROM open_consolidated_data
            WHERE neighbourhood_158 IS NOT NULL AND neighbourhood_158 != ''
            GROUP BY neighbourhood_158 ORDER BY crime_count DESC
        """, conn)
        
        if df.empty:
            print("No neighbourhood data available for plotting")
            return
        
        plot_df = df if show_all else df.head(40)
        mid = len(plot_df) // 2
        left, right = plot_df.iloc[:mid], plot_df.iloc[mid:]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(8, len(plot_df)*0.2)))
        fig.suptitle(f'Neighbourhoods by Crime Count {"(All)" if show_all else "(Top 40)"}', 
                    fontsize=14, fontweight='bold', y=0.99)
        
        for ax, data, title, palette in [(ax1, left, 'Higher Crime Areas', 'viridis'),
                                          (ax2, right, 'Lower Crime Areas', 'rocket')]:
            ax.barh(range(len(data)), data['crime_count'], 
                   color=sns.color_palette(palette, len(data)))
            ax.set_yticks(range(len(data)))
            ax.set_yticklabels([str(x)[:30]+'...' if len(str(x))>30 else str(x) 
                              for x in data['neighbourhood_158']], fontsize=6, ha='right')
            ax.set_xlabel('Crime Count', fontsize=9, fontweight='bold')
            ax.set_title(title, fontsize=10, fontweight='bold', pad=5)
            ax.grid(axis='x', alpha=0.3, linestyle='--')
            ax.invert_yaxis()
            if len(data) <= 40:
                for i, (_, row) in enumerate(data.iterrows()):
                    ax.text(row['crime_count']+1, i, f'{row["crime_count"]:,}', 
                           va='center', fontsize=5)
        
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()
        print(f"\nDisplayed {'all' if show_all else f'top {len(plot_df)}'} neighbourhoods (split view)")
        
    except Exception as e:
        print(f"Error plotting neighbourhood histogram: {e}")


def plot_offence_histogram(conn, show_all=True):
    try:
        df = pd.read_sql_query("""
            SELECT csi_category, COUNT(*) as count FROM open_consolidated_data
            WHERE csi_category IS NOT NULL AND csi_category != ''
            GROUP BY csi_category ORDER BY count DESC
        """, conn)
        
        if df.empty:
            print("No csi_category data available for plotting")
            return
        
        plot_df = df if show_all else df.head(20)
        plt.figure(figsize=(14, max(8, len(plot_df)*0.35)))
        
        plt.barh(range(len(plot_df)), plot_df['count'], 
                color=sns.color_palette("flare", len(plot_df)))
        labels = ['\n'.join(wrap(str(o), 45)) if len(str(o))>45 else str(o) 
                 for o in plot_df['csi_category']]
        plt.yticks(range(len(plot_df)), labels, fontsize=7, ha='right')
        
        plt.xlabel('Number of Occurrences', fontsize=10, fontweight='bold')
        plt.ylabel('csi_category Type', fontsize=10, fontweight='bold')
        plt.title(f'Crime csi_category by Frequency {"(All)" if show_all else "(Top 20)"}', 
                 fontsize=12, fontweight='bold', pad=10)
        
        if len(plot_df) <= 30:
            for i, v in enumerate(plot_df['count']):
                plt.text(v+1, i, f'{v:,}', va='center', fontsize=6)
        
        plt.tight_layout()
        plt.show()
        print(f"\nDisplayed {'all' if show_all else f'top {len(plot_df)}'} csi_category categories")
        
    except Exception as e:
        print(f"Error plotting csi_category histogram: {e}")


def plot_event_type_histogram(conn, show_all=True):
    try:
        df = pd.read_sql_query("""
            SELECT event_type, COUNT(*) as count FROM open_consolidated_data
            WHERE event_type IS NOT NULL AND event_type != ''
            GROUP BY event_type ORDER BY count DESC
        """, conn)
        
        if df.empty:
            print("No event_type data available for plotting")
            return
        
        plot_df = df if show_all else df.head(15)
        plt.figure(figsize=(12, max(6, len(plot_df)*0.4)))
        
        plt.barh(range(len(plot_df)), plot_df['count'], 
                color=sns.color_palette("crest", len(plot_df)))
        plt.yticks(range(len(plot_df)), plot_df['event_type'], fontsize=8, ha='right')
        
        plt.xlabel('Number of Occurrences', fontsize=10, fontweight='bold')
        plt.ylabel('Event Type', fontsize=10, fontweight='bold')
        plt.title(f'Event Types by Frequency {"(All)" if show_all else "(Top 15)"}', 
                 fontsize=12, fontweight='bold', pad=10)
        plt.gca().invert_yaxis()
        
        if len(plot_df) <= 25:
            for i, v in enumerate(plot_df['count']):
                plt.text(v+1, i, f'{v:,}', va='center', fontsize=7)
        
        plt.tight_layout()
        plt.show()
        print(f"\nDisplayed {'all' if show_all else f'top {len(plot_df)}'} event types")
        
    except Exception as e:
        print(f"Error plotting event_type histogram: {e}")


def plot_neighbourhood_year_heatmap(conn, show_all=True, use_log_scale=True):
    try:
        df = pd.read_sql_query("""
            SELECT neighbourhood_158, EXTRACT(YEAR FROM occ_date) as year, COUNT(*) as crime_count
            FROM open_consolidated_data
            WHERE neighbourhood_158 IS NOT NULL AND neighbourhood_158 != '' AND occ_date IS NOT NULL
            GROUP BY neighbourhood_158, EXTRACT(YEAR FROM occ_date)
            ORDER BY neighbourhood_158, year
        """, conn)
        
        if df.empty:
            print("No temporal neighbourhood data available for heatmap")
            return
        
        pivot_df = df.pivot_table(index='neighbourhood_158', columns='year', 
                                 values='crime_count', fill_value=0)
        pivot_df = pivot_df.loc[pivot_df.sum(axis=1).sort_values(ascending=False).index]
        
        if not show_all:
            pivot_df = pivot_df.head(25)
        
        mid = len(pivot_df) // 2
        high_df, low_df = pivot_df.iloc[:mid], pivot_df.iloc[mid:]
        
        all_values = pivot_df.values[pivot_df.values > 0]
        if len(all_values) == 0:
            print("No crime data available for heatmap")
            return
        
        norm = LogNorm(vmin=float(all_values.min()), vmax=float(all_values.max())) if use_log_scale else Normalize(vmin=0, vmax=float(all_values.max()))
        
        for data, title in [(high_df, 'Higher Crime Areas'), (low_df, 'Lower Crime Areas')]:
            print(f"\n  Opening Figure: {title} ({len(data)} neighbourhoods)...")
            fig, ax = plt.subplots(1, 1, figsize=(14, max(12, len(data)*0.45)))
            
            sns.heatmap(data, ax=ax, cmap=STANDARD_HEAT_CMAP, norm=norm, annot=False,
                       linewidths=0.5, linecolor='white', cbar=True,
                       cbar_kws={'label': 'Crime Count', 'shrink': 0.9, 'pad': 0.05},
                       xticklabels=True, yticklabels=True, rasterized=True)
            
            ax.set_xlabel('Year', fontsize=11, fontweight='bold')
            ax.set_ylabel('Neighbourhood', fontsize=11, fontweight='bold')
            ax.set_title(f'{title} - Crime Trends Over Time', fontsize=13, fontweight='bold', pad=15)
            ax.tick_params(axis='x', rotation=45, labelsize=9, pad=4)
            ax.tick_params(axis='y', labelsize=9, pad=10, length=0)
            ax.invert_yaxis()
            
            scale_note = "LOG" if use_log_scale else "LINEAR"
            fig.text(0.5, 0.01, f'Scale: White->Yellow->Orange->Red | {scale_note} | White=0, Red=max', 
                    ha='center', fontsize=9, style='italic', 
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.96])
            plt.show()
            plt.close(fig)
        
        years = f"{int(pivot_df.columns.min())}-{int(pivot_df.columns.max())}"
        print(f"\nDisplayed TWO separate heatmaps for {'all' if show_all else 'top 25'} neighbourhoods")
        print(f"  Years: {years}{' (log scale)' if use_log_scale else ''}")
        
    except Exception as e:
        print(f"Error plotting heatmap: {e}")
        import traceback
        traceback.print_exc()


def plot_traffic_type_histogram(conn):
    try:
        df = pd.read_sql_query("""
            SELECT SUM(CASE WHEN automobile THEN 1 ELSE 0 END) as automobile,
                   SUM(CASE WHEN motorcycle THEN 1 ELSE 0 END) as motorcycle,
                   SUM(CASE WHEN passenger THEN 1 ELSE 0 END) as passenger,
                   SUM(CASE WHEN bicycle THEN 1 ELSE 0 END) as bicycle,
                   SUM(CASE WHEN pedestrian THEN 1 ELSE 0 END) as pedestrian
            FROM traffic_collisions_data
        """, conn)
        
        if df.empty or df.iloc[0].isnull().all():
            print("No traffic type data available for plotting")
            return
        
        plot_data = df.melt(var_name='accident_type', value_name='count')
        plot_data = plot_data[plot_data['count'] > 0]
        
        if plot_data.empty:
            print("No accident type data to display")
            return
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(plot_data['accident_type'].str.title(), plot_data['count'], 
                      color=sns.color_palette("coolwarm", len(plot_data)), edgecolor='black', linewidth=0.5)
        
        plt.xlabel('Accident Type', fontsize=10, fontweight='bold')
        plt.ylabel('Number of Occurrences', fontsize=10, fontweight='bold')
        plt.title('Traffic Collisions by Vehicle/Pedestrian Type', fontsize=12, fontweight='bold', pad=10)
        
        for bar in bars:
            h = bar.get_height()
            plt.text(bar.get_x()+bar.get_width()/2, h+max(plot_data['count'])*0.015, 
                    f'{int(h):,}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.xticks(fontsize=9, rotation=15, ha='right')
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.show()
        print("\nDisplayed histogram of traffic accident types")
        
    except Exception as e:
        print(f"Error plotting traffic type histogram: {e}")


def plot_traffic_neighbourhood_histogram(conn, show_all=True):
    try:
        df = pd.read_sql_query("""
            SELECT neighbourhood_158, COUNT(*) as accident_count
            FROM traffic_collisions_data
            WHERE neighbourhood_158 IS NOT NULL AND neighbourhood_158 != ''
            GROUP BY neighbourhood_158 ORDER BY accident_count DESC
        """, conn)
        
        if df.empty:
            print("No neighbourhood data available for traffic accidents")
            return
        
        plot_df = df if show_all else df.head(40)
        mid = len(plot_df) // 2
        left, right = plot_df.iloc[:mid], plot_df.iloc[mid:]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(8, len(plot_df)*0.2)))
        fig.suptitle(f'Neighbourhoods by Traffic Accident Count {"(All)" if show_all else "(Top 40)"}', 
                    fontsize=14, fontweight='bold', y=0.99)
        
        for ax, data, title, palette in [(ax1, left, 'Higher Accident Areas', 'Reds'),
                                          (ax2, right, 'Lower Accident Areas', 'Blues')]:
            ax.barh(range(len(data)), data['accident_count'], 
                   color=sns.color_palette(palette, len(data)))
            ax.set_yticks(range(len(data)))
            ax.set_yticklabels([str(x)[:30]+'...' if len(str(x))>30 else str(x) 
                              for x in data['neighbourhood_158']], fontsize=6, ha='right')
            ax.set_xlabel('Accident Count', fontsize=9, fontweight='bold')
            ax.set_title(title, fontsize=10, fontweight='bold', pad=5)
            ax.grid(axis='x', alpha=0.3, linestyle='--')
            ax.invert_yaxis()
            if len(data) <= 40:
                for i, (_, row) in enumerate(data.iterrows()):
                    ax.text(row['accident_count']+1, i, f'{row["accident_count"]:,}', 
                           va='center', fontsize=5)
        
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()
        print(f"\nDisplayed {'all' if show_all else f'top {len(plot_df)}'} neighbourhoods (split view)")
        
    except Exception as e:
        print(f"Error plotting traffic neighbourhood histogram: {e}")


def plot_traffic_year_linegraph(conn):
    try:
        df = pd.read_sql_query("""
            SELECT EXTRACT(YEAR FROM occ_date) as year, COUNT(*) as accident_count
            FROM traffic_collisions_data WHERE occ_date IS NOT NULL
            GROUP BY EXTRACT(YEAR FROM occ_date) ORDER BY year
        """, conn)
        
        if df.empty:
            print("No temporal data available for line graph")
            return
        
        plt.figure(figsize=(12, 6))
        plt.plot(df['year'], df['accident_count'], marker='o', linewidth=2, markersize=6, 
                color='#2E86AB', markerfacecolor='#A23B72', markeredgecolor='white', markeredgewidth=1)
        plt.fill_between(df['year'], df['accident_count'], alpha=0.15, color='#2E86AB')
        
        plt.xlabel('Year', fontsize=10, fontweight='bold')
        plt.ylabel('Number of Traffic Accidents', fontsize=10, fontweight='bold')
        plt.title('Traffic Accidents Over Time', fontsize=12, fontweight='bold', pad=10)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.xticks(df['year'].astype(int), rotation=45, ha='right', fontsize=9)
        
        if len(df) <= 20:
            for _, row in df.iterrows():
                plt.text(row['year'], row['accident_count']+max(df['accident_count'])*0.025, 
                        f'{int(row["accident_count"]):,}', ha='center', fontsize=8, fontweight='bold')
        
        if len(df) >= 2:
            first, last = df['accident_count'].iloc[0], df['accident_count'].iloc[-1]
            change = ((last-first)/first*100) if first != 0 else 0
            trend = "+" if change > 0 else "-" if change < 0 else ""
            plt.figtext(0.5, 0.01, f'{trend} {abs(change):.1f}% change from {int(df["year"].min())} to {int(df["year"].max())}', 
                       ha='center', fontsize=9, style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
        print(f"\nDisplayed line graph for years {int(df['year'].min())} to {int(df['year'].max())}")
        
    except Exception as e:
        print(f"Error plotting year line graph: {e}")


def ask_show_all(prompt="Show all data (may be crowded)?"):
    while True:
        choice = input(f"\n  {prompt} (y/n): ").strip().lower()
        if choice in ['y', 'yes']: return True
        elif choice in ['n', 'no']: return False
        print("  Please enter 'y' or 'n'")


def ask_heatmap_sensitivity():
    while True:
        print("\n  Heatmap Color Scale Options:")
        print("  1. Log scale (better for seeing small differences)")
        print("  2. Linear scale (standard proportional view)")
        choice = input("  Select (1-2, default=1): ").strip()
        if choice in ['1', '']: return True
        elif choice == '2': return False
        print("  Please enter 1 or 2")


def consolidated_crime_menu(conn):
    while True:
        print(f"\n{HEADER_SEPARATOR}")
        print("  CONSOLIDATED CRIMES - Query Options")
        print(HEADER_SEPARATOR)
        print("  1. Search by exact event_unique_id")
        print("  2. Search by partial event_unique_id")
        print("  3. Plot: Neighbourhoods by crime count (split view)")
        print("  4. Plot: csi_category types by frequency")
        print("  5. Plot: Event types by frequency")
        print("  6. Plot: Neighbourhood vs Year heatmap (separate pages)")
        print("  0. Back to main menu")
        print(HEADER_SEPARATOR)
        
        choice = input("\n  Select option (0-6): ").strip()
        
        if choice == '1':
            search_by_event_id(conn, 'open_consolidated_data', input("  Enter exact event_unique_id: ").strip(), True)
        elif choice == '2':
            search_by_event_id(conn, 'open_consolidated_data', input("  Enter partial event_unique_id: ").strip(), False)
        elif choice == '3':
            plot_neighbourhood_crime_histogram(conn, ask_show_all("Show ALL neighbourhoods in histogram?"))
        elif choice == '4':
            plot_offence_histogram(conn, ask_show_all("Show ALL csi_category types?"))
        elif choice == '5':
            plot_event_type_histogram(conn, ask_show_all("Show ALL event types?"))
        elif choice == '6':
            plot_neighbourhood_year_heatmap(conn, ask_show_all("Show ALL neighbourhoods in heatmap?"), 
                                          ask_heatmap_sensitivity())
        elif choice == '0': break
        else: print("  Invalid option. Please select 0-6.")


def traffic_accidents_menu(conn):
    while True:
        print(f"\n{HEADER_SEPARATOR}")
        print("  TRAFFIC ACCIDENTS - Query Options")
        print(HEADER_SEPARATOR)
        print("  1. Search by exact event_unique_id")
        print("  2. Search by partial event_unique_id")
        print("  3. Plot: Accident types frequency")
        print("  4. Plot: Neighbourhoods by accident count (split view)")
        print("  5. Plot: Accidents per year (line graph)")
        print("  0. Back to main menu")
        print(HEADER_SEPARATOR)
        
        choice = input("\n  Select option (0-5): ").strip()
        
        if choice == '1':
            search_by_event_id(conn, 'traffic_collisions_data', input("  Enter exact event_unique_id: ").strip(), True)
        elif choice == '2':
            search_by_event_id(conn, 'traffic_collisions_data', input("  Enter partial event_unique_id: ").strip(), False)
        elif choice == '3':
            plot_traffic_type_histogram(conn)
        elif choice == '4':
            plot_traffic_neighbourhood_histogram(conn, ask_show_all("Show ALL neighbourhoods in histogram?"))
        elif choice == '5':
            plot_traffic_year_linegraph(conn)
        elif choice == '0': break
        else: print("  Invalid option. Please select 0-5.")


def main_menu():
    print(f"\n{'#'*70}")
    print("  TORONTO CRIME DATABASE - Interactive Query Tool")
    print(f"{'#'*70}")
    
    conn = get_db_connection()
    if not conn:
        print("Could not connect to database. Please check your .env configuration.")
        return
    
    try:
        while True:
            print(f"\n{HEADER_SEPARATOR}")
            print("  Select Database:")
            print(HEADER_SEPARATOR)
            print("  1. Consolidated Crimes Database")
            print("  2. Traffic Accidents Database")
            print("  0. Exit")
            print(HEADER_SEPARATOR)
            
            choice = input("\n  Select option (0-2): ").strip()
            
            if choice == '1': consolidated_crime_menu(conn)
            elif choice == '2': traffic_accidents_menu(conn)
            elif choice == '0':
                print("\n  Thank you for using Toronto Crime Query Tool. Goodbye!\n")
                break
            else: print("  Invalid option. Please select 0-2.")
    finally:
        conn.close()


if __name__ == "__main__":
    main_menu()