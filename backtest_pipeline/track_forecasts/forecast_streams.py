#!/usr/bin/env python
# coding: utf-8

"""
Track Streams Forecasting Script

This script connects to the database, retrieves track streaming data,
calculates forecasts, and stores the results back in the database.

It uses the following tables:
- backtest_track_daily_training_data: Source of track streaming data
- backtest_artist_mldr: Source of artist MLDR values
- backtest_track_streams_forecast: Destination for forecast results
"""

import os
import sys
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
import atexit
import argparse

# Add project root to path to ensure imports work correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import database utilities
from utils.database import (
    fetch_all, 
    fetch_one, 
    query_to_dataframe,
    dataframe_to_table,
    execute_batch,
    close_all_connections
)

# Import track stream forecasting utilities
from utils.track_stream_forecasting import (
    extract_track_metrics,
    build_complete_track_forecast,
    calculate_track_decay_rates_by_segment,
    track_lifecycle_segment_boundaries
)
from utils.data_processing import (
    calculate_monthly_stream_averages
)

# Always register the cleanup function to ensure proper shutdown
atexit.register(close_all_connections)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('forecast_streams')

def get_artist_mldr(cm_artist_id):
    """
    Retrieve the artist MLDR (Monthly Listener Decay Rate) from the database.
    
    Args:
        cm_artist_id: ChartMetric artist ID used to find the corresponding MLDR
        
    Returns:
        float: The artist's MLDR value, or None if not found
    """
    # Convert numpy types to native Python types
    if isinstance(cm_artist_id, (np.integer, np.floating)):
        cm_artist_id = int(cm_artist_id)
        
    query = """
    SELECT mldr FROM backtest_artist_mldr 
    WHERE cm_artist_id = %s 
    ORDER BY created_at DESC 
    LIMIT 1
    """
    
    # Use a tuple for parameters with psycopg2
    result = fetch_one(query, params=(cm_artist_id,))
    
    if result:
        # Convert Decimal to float if needed
        mldr_value = result[0]
        if hasattr(mldr_value, 'as_tuple'):  # Check if it's a Decimal
            mldr_value = float(mldr_value)
        return mldr_value
    else:
        logger.warning(f"No MLDR found for artist ID {cm_artist_id}")
        return None

def get_training_data(cm_track_id, cm_artist_id):
    """
    Retrieve and process training data for a specific track.
    
    Args:
        cm_track_id: ChartMetric track ID
        cm_artist_id: ChartMetric artist ID
        
    Returns:
        tuple: (training_data_id, track_streams_last_30days, months_since_release)
               or (None, None, None) if insufficient data
    """
    # Convert numpy types to native Python types
    if isinstance(cm_track_id, (np.integer, np.floating)):
        cm_track_id = int(cm_track_id)
    if isinstance(cm_artist_id, (np.integer, np.floating)):
        cm_artist_id = int(cm_artist_id)
        
    # Let's use a different approach - fetch the data using fetch_all first
    query = """
    SELECT id, date, daily_streams, days_from_release 
    FROM backtest_track_daily_training_data 
    WHERE cm_track_id = %s AND cm_artist_id = %s 
    ORDER BY date ASC
    """
    
    # Fetch the data using fetch_all
    results = fetch_all(query, params=(cm_track_id, cm_artist_id))
    
    if not results:
        logger.warning(f"No training data found for track ID {cm_track_id}")
        return None, None, None
    
    # Convert to DataFrame and ensure numeric columns are proper floats
    track_data = pd.DataFrame(results, columns=['id', 'date', 'daily_streams', 'days_from_release'])
    
    # Convert Decimal types to float if needed
    if 'daily_streams' in track_data.columns:
        track_data['daily_streams'] = track_data['daily_streams'].astype(float)
    if 'days_from_release' in track_data.columns:
        track_data['days_from_release'] = track_data['days_from_release'].astype(float)
    
    # Get the ID of the first training data entry for reference
    training_data_id = track_data.iloc[0]['id']
    
    # Calculate months since release from days_from_release
    months_since_release = track_data['days_from_release'].max() / 30.44  # Average days per month
    
    # Calculate streams in the last 30 days
    recent_data = track_data.sort_values('date', ascending=False).head(30)
    track_streams_last_30days = float(recent_data['daily_streams'].sum())
    
    return training_data_id, track_streams_last_30days, months_since_release, track_data

def get_default_decay_parameters():
    """
    Get default decay parameters for track forecasting.
    
    Returns:
        tuple: (fitted_params_df, sp_range, sp_reach) with default values
    """
    # Create default fitted parameters DataFrame with decay rates for each segment
    segments = list(range(1, len(track_lifecycle_segment_boundaries)))
    # These are placeholder decay rates, should be adjusted based on actual data analysis
    decay_rates = [0.15, 0.08, 0.04, 0.02, 0.01]  # Higher rates for early segments, lower for later
    
    fitted_params_df = pd.DataFrame({
        'segment': segments,
        'k': decay_rates
    })
    
    # Create default stream influence factor ranges
    sp_range = pd.DataFrame({
        'RangeStart': [0, 100000, 500000, 1000000, 5000000],
        'RangeEnd': [100000, 500000, 1000000, 5000000, float('inf')],
        'Column 1': ['low', 'medium_low', 'medium', 'medium_high', 'high']
    })
    
    # Create default stream influence adjustment factors
    sp_reach = pd.DataFrame({
        'low': [0.18, 0.10, 0.05, 0.03, 0.015],
        'medium_low': [0.17, 0.09, 0.045, 0.025, 0.012],
        'medium': [0.16, 0.08, 0.04, 0.02, 0.01],
        'medium_high': [0.15, 0.07, 0.035, 0.018, 0.009],
        'high': [0.14, 0.06, 0.03, 0.015, 0.008]
    })
    
    return fitted_params_df, sp_range, sp_reach

def generate_track_forecasts(cm_track_id, cm_artist_id, mldr=None):
    """
    Generate stream forecasts for a specific track.
    
    Args:
        cm_track_id: ChartMetric track ID
        cm_artist_id: ChartMetric artist ID
        mldr: Monthly Listener Decay Rate (optional, will be fetched if not provided)
        
    Returns:
        list: List of forecast dictionaries or None if forecasting failed
    """
    # Step 1: Get track training data
    training_data_result = get_training_data(cm_track_id, cm_artist_id)
    if training_data_result[0] is None:
        logger.warning(f"Cannot generate forecasts: insufficient training data for track {cm_track_id}")
        return None
    
    training_data_id, track_streams_last_30days, months_since_release_total, track_data = training_data_result
    
    # Step 2: Get MLDR if not provided
    if mldr is None:
        mldr = get_artist_mldr(cm_artist_id)
        if mldr is None:
            logger.warning(f"No MLDR found for artist {cm_artist_id}, using default value")
            mldr = 0.05  # Default MLDR value if none found
    
    # Step 3: Extract track metrics from training data
    # Convert days_from_release to months
    track_data['months_from_release'] = track_data['days_from_release'] / 30.44
    months_since_release = track_data['months_from_release'].to_list()
    
    # Calculate monthly stream averages
    monthly_averages = calculate_monthly_stream_averages(
        months=months_since_release,
        streams=track_data['daily_streams'].to_list()
    )
    
    # Create track metrics dictionary
    track_metrics = {
        'months_since_release': months_since_release,
        'monthly_averages': monthly_averages,
        'track_streams_last_30days': track_streams_last_30days,
        'months_since_release_total': months_since_release_total
    }
    
    # Step 4: Get default decay parameters
    fitted_params_df, sp_range, sp_reach = get_default_decay_parameters()
    
    # Step 5: Set forecast parameters
    forecast_periods = 60  # 5 years of monthly forecasts
    
    # Simple estimation for stream influence factor based on last 30 days streams
    # This should be improved with actual data from the track
    stream_influence_factor = min(track_streams_last_30days * 10, 5000000)
    
    # Step 6: Generate forecasts
    try:
        forecast_result = build_complete_track_forecast(
            track_metrics=track_metrics,
            mldr=mldr,
            fitted_params_df=fitted_params_df,
            stream_influence_factor=stream_influence_factor,
            sp_range=sp_range,
            sp_reach=sp_reach,
            track_lifecycle_segment_boundaries=track_lifecycle_segment_boundaries,
            forecast_periods=forecast_periods
        )
        
        # Step 7: Format forecasts for database storage
        forecasts = []
        for _, row in forecast_result['forecast_df'].iterrows():
            forecast = {
                'month': int(row['month']),
                'predicted_streams_for_month': float(row['predicted_streams_for_month']),
                'segment_used': int(row['segment_used']),
                'time_used': int(row['time_used']),
                'cm_track_id': cm_track_id,
                'cm_artist_id': cm_artist_id,
                'training_data_id': training_data_id
            }
            forecasts.append(forecast)
        
        logger.info(f"Generated {len(forecasts)} forecast periods for track {cm_track_id}")
        return forecasts
    
    except Exception as e:
        logger.error(f"Error generating forecasts for track {cm_track_id}: {str(e)}")
        return None

def store_forecasts(forecasts):
    """
    Store the generated forecasts in the database.
    
    Args:
        forecasts: List of forecast dictionaries
        
    Returns:
        int: Number of forecasts stored
    """
    if not forecasts:
        return 0
    
    # Prepare data for batch insert
    forecast_values = []
    for forecast in forecasts:
        # Convert any numpy types to native Python types
        month = int(forecast['month']) if isinstance(forecast['month'], np.integer) else forecast['month']
        forecasted_value = float(forecast['predicted_streams_for_month']) if isinstance(forecast['predicted_streams_for_month'], np.number) else forecast['predicted_streams_for_month']
        segment_used = int(forecast['segment_used']) if isinstance(forecast['segment_used'], np.integer) else forecast['segment_used']
        time_used = int(forecast['time_used']) if isinstance(forecast['time_used'], np.integer) else forecast['time_used']
        cm_track_id = int(forecast['cm_track_id']) if isinstance(forecast['cm_track_id'], np.integer) else forecast['cm_track_id']
        cm_artist_id = int(forecast['cm_artist_id']) if isinstance(forecast['cm_artist_id'], np.integer) else forecast['cm_artist_id']
        training_data_id = int(forecast['training_data_id']) if isinstance(forecast['training_data_id'], np.integer) else forecast['training_data_id']
        
        forecast_values.append((
            month,
            forecasted_value,
            segment_used,
            time_used,
            cm_track_id,
            cm_artist_id,
            training_data_id,
            datetime.now()
        ))
    
    # Insert into database
    insert_query = """
    INSERT INTO backtest_track_streams_forecast
    (month, forecasted_value, segment_used, time_used, cm_track_id, cm_artist_id, training_data_id, created_at)
    VALUES %s
    """
    
    rows_affected = execute_batch(insert_query, forecast_values)
    return rows_affected

def get_tracks_to_forecast():
    """
    Get a list of tracks that need forecasting.
    
    Returns:
        list: List of (cm_track_id, cm_artist_id) tuples
    """
    query = """
    SELECT DISTINCT cm_track_id, cm_artist_id 
    FROM backtest_track_daily_training_data 
    WHERE cm_track_id NOT IN (
        SELECT DISTINCT cm_track_id FROM backtest_track_streams_forecast
    )
    """
    
    tracks = fetch_all(query)
    return tracks

def get_track_artist_id(cm_track_id):
    """
    Get the artist ID for a given track ID.
    
    Args:
        cm_track_id: ChartMetric track ID
        
    Returns:
        int: ChartMetric artist ID or None if not found
    """
    query = """
    SELECT DISTINCT cm_artist_id 
    FROM backtest_track_daily_training_data 
    WHERE cm_track_id = %s
    LIMIT 1
    """
    
    result = fetch_one(query, params=(cm_track_id,))
    
    if result:
        # Convert to int if needed
        cm_artist_id = result[0]
        if isinstance(cm_artist_id, (np.integer, np.floating)):
            cm_artist_id = int(cm_artist_id)
        return cm_artist_id
    else:
        logger.warning(f"No artist ID found for track ID {cm_track_id}")
        return None

def main(single_track_id=None):
    """
    Main function to orchestrate the forecasting process.
    
    Args:
        single_track_id: Optional ChartMetric track ID to process just one track
    """
    logger.info("Starting track streams forecasting process")
    
    if single_track_id:
        # Process just one specific track
        logger.info(f"Processing single track ID: {single_track_id}")
        
        # Get the artist ID for this track
        cm_artist_id = get_track_artist_id(single_track_id)
        
        if not cm_artist_id:
            logger.error(f"Could not find artist ID for track {single_track_id}")
            return
            
        # Generate forecasts for the single track
        forecasts = generate_track_forecasts(single_track_id, cm_artist_id)
        
        if forecasts:
            # Store forecasts in database
            rows_stored = store_forecasts(forecasts)
            logger.info(f"Stored {rows_stored} forecast periods for track {single_track_id}")
        else:
            logger.warning(f"Failed to generate forecasts for track {single_track_id}")
    else:
        # Process all tracks that need forecasting
        tracks = get_tracks_to_forecast()
        logger.info(f"Found {len(tracks)} tracks to forecast")
        
        # Process each track
        for track_num, (cm_track_id, cm_artist_id) in enumerate(tracks, 1):
            logger.info(f"Processing track {track_num}/{len(tracks)}: {cm_track_id}")
            
            # Generate forecasts
            forecasts = generate_track_forecasts(cm_track_id, cm_artist_id)
            
            if forecasts:
                # Store forecasts in database
                rows_stored = store_forecasts(forecasts)
                logger.info(f"Stored {rows_stored} forecast periods for track {cm_track_id}")
            else:
                logger.warning(f"Failed to generate forecasts for track {cm_track_id}")
    
    logger.info("Track streams forecasting process completed")

if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Generate track stream forecasts')
    parser.add_argument('--track_id', type=int, help='Process a single track with the given ChartMetric track ID')
    
    args = parser.parse_args()
    
    # Run with the provided track ID if specified
    main(single_track_id=args.track_id) 