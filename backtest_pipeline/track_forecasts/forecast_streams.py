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

# Import decay model utilities
from utils.decay_models import (
    adjust_track_decay_rates,
    calculate_track_decay_rates_by_segment,
    forecast_track_streams
)

from utils.decay_models.parameter_updates import (
    generate_track_decay_rates_by_month,
    create_decay_rate_dataframe
)

# Import decay rate parameters
from utils.decay_rates import (
    track_lifecycle_segment_boundaries,
    DEFAULT_FORECAST_PERIODS,
    DEFAULT_STREAM_INFLUENCE_FACTOR,
    fitted_params_df,
    sp_range,
    SP_REACH
)

# Always register the cleanup function to ensure proper shutdown
atexit.register(close_all_connections)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('forecast_streams')

def get_decay_parameters():
    """
    Get the decay parameters for forecasting.
    
    Returns:
        DataFrame: DataFrame containing decay parameters for different segments.
    """
    # Get decay rates based on fitted parameters from the decay rates module
    decay_rates_df = fitted_params_df.copy()
    return decay_rates_df

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
    
    return training_data_id, track_streams_last_30days, months_since_release

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
    # Get track training data
    training_data_id, track_streams_last_30days, months_since_release = get_training_data(cm_track_id, cm_artist_id)
    
    if not training_data_id:
        return None
    
    # Get artist MLDR if not provided
    if mldr is None:
        mldr = get_artist_mldr(cm_artist_id)
        if mldr is None:
            logger.warning(f"Unable to forecast without MLDR for artist {cm_artist_id}")
            return None
    
    # Get decay parameters
    decay_rates_df = get_decay_parameters()
    
    # Generate track-specific decay rates for all forecast months
    track_monthly_decay_rates = generate_track_decay_rates_by_month(decay_rates_df, track_lifecycle_segment_boundaries)
    
    # Determine the observed time range from the track's streaming data
    track_data_start_month = 1  # Assuming we start from month 1
    track_data_end_month = int(months_since_release)
    
    # Create a structured DataFrame that combines model-derived decay rates with observed data
    track_decay_rate_df = create_decay_rate_dataframe(
        track_months_since_release=list(range(1, 501)),  # Forecast for 500 months
        track_monthly_decay_rates=track_monthly_decay_rates,
        mldr=mldr,
        track_data_start_month=track_data_start_month,
        track_data_end_month=track_data_end_month
    )
    
    # Fit the track's decay pattern based on a placeholder k value (we don't have actual track_decay_k)
    # This is a simplification - in a real scenario, we'd calculate this from actual data
    adjusted_track_decay_df, _ = adjust_track_decay_rates(
        track_decay_rate_df,
        track_decay_k=0.05  # Using a default value as we don't have the actual fitted value
    )
    
    # Calculate average decay rates for each segment
    segmented_track_decay_rates_df = calculate_track_decay_rates_by_segment(
        adjusted_track_decay_df, 
        track_lifecycle_segment_boundaries
    )
    
    # Generate stream forecasts
    forecasts = forecast_track_streams(
        segmented_track_decay_rates_df,
        track_streams_last_30days,
        int(months_since_release),
        DEFAULT_FORECAST_PERIODS
    )
    
    # Add metadata to each forecast
    for forecast in forecasts:
        forecast['cm_track_id'] = int(cm_track_id) if isinstance(cm_track_id, (np.integer, np.floating)) else cm_track_id
        forecast['cm_artist_id'] = int(cm_artist_id) if isinstance(cm_artist_id, (np.integer, np.floating)) else cm_artist_id
        forecast['training_data_id'] = int(training_data_id) if isinstance(training_data_id, (np.integer, np.floating)) else training_data_id
    
    return forecasts

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