#!/usr/bin/env python
# coding: utf-8

"""
Check Track Forecasts Results

This script verifies that forecast data was properly stored in the database
for a specific track ID.
"""

import os
import sys
import pandas as pd
import atexit
import argparse

# Add project root to path to ensure imports work correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import database utilities
from utils.database import query_to_dataframe, close_all_connections

# Always register the cleanup function to ensure proper shutdown
atexit.register(close_all_connections)

def check_forecasts(track_id):
    """
    Check forecasts for a specific track ID.
    
    Args:
        track_id: ChartMetric track ID to check
    
    Returns:
        DataFrame with forecast data
    """
    # Replace '%s' with direct parameterization for SQLAlchemy
    query = f"""
    SELECT 
        month, 
        forecasted_value, 
        segment_used, 
        time_used, 
        cm_track_id, 
        cm_artist_id,
        created_at
    FROM backtest_track_streams_forecast
    WHERE cm_track_id = {track_id}
    ORDER BY month
    """
    
    # Get forecast data
    forecast_df = query_to_dataframe(query)
    
    if forecast_df.empty:
        print(f"No forecasts found for track ID {track_id}")
        return None
    
    # Print summary statistics
    print(f"Found {len(forecast_df)} forecast periods for track ID {track_id}")
    print(f"Artist ID: {forecast_df['cm_artist_id'].iloc[0]}")
    print(f"Total forecasted streams: {forecast_df['forecasted_value'].sum():.2f}")
    print(f"Created at: {forecast_df['created_at'].iloc[0]}")
    
    # Print first 5 forecast periods
    print("\nFirst 5 forecast periods:")
    for _, row in forecast_df.head(5).iterrows():
        print(f"Month {row['month']}: {row['forecasted_value']:.2f} streams (Segment {row['segment_used']})")
    
    # Save results to CSV
    output_file = f"track_{track_id}_forecast_check.csv"
    forecast_df.to_csv(output_file, index=False)
    print(f"\nSaved results to {output_file}")
    
    return forecast_df

def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Check track stream forecasts in the database')
    parser.add_argument('--track_id', type=int, required=True, help='ChartMetric track ID to check')
    
    args = parser.parse_args()
    
    # Check forecasts for the specified track
    check_forecasts(args.track_id)

if __name__ == "__main__":
    main() 