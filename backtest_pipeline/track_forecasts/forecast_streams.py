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
import time
import concurrent.futures

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
from utils.decay_rates.fitted_params import fitted_params_df
from utils.decay_rates.sp_reach import SP_REACH
from utils.decay_rates.volume_ranges import sp_range

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
    SELECT id, "Date", "CumulativeStreams", days_from_release 
    FROM backtest_track_daily_training_data 
    WHERE cm_track_id = %s AND cm_artist_id = %s 
    ORDER BY "Date" ASC
    """
    
    # Fetch the data using fetch_all
    results = fetch_all(query, params=(cm_track_id, cm_artist_id))
    
    if not results:
        logger.warning(f"No training data found for track ID {cm_track_id}")
        return None, None, None
    
    # Convert to DataFrame and ensure numeric columns are proper floats
    track_data = pd.DataFrame(results, columns=['id', 'Date', 'CumulativeStreams', 'days_from_release'])
    
    # Convert Decimal types to float if needed
    if 'CumulativeStreams' in track_data.columns:
        track_data['CumulativeStreams'] = track_data['CumulativeStreams'].astype(float)
    if 'days_from_release' in track_data.columns:
        track_data['days_from_release'] = track_data['days_from_release'].astype(float)
    
    # Get the ID of the first training data entry for reference
    training_data_id = track_data.iloc[0]['id']
    
    # Calculate months since release from days_from_release
    months_since_release = track_data['days_from_release'].max() / 30.44  # Average days per month
    
    # We need to derive daily_streams from the CumulativeStreams
    # Calculate daily_streams by differencing the CumulativeStreams
    track_data = track_data.sort_values('Date')
    track_data['daily_streams'] = track_data['CumulativeStreams'].diff().fillna(track_data['CumulativeStreams'])
    
    # Calculate streams in the last 30 days
    recent_data = track_data.sort_values('Date', ascending=False).head(30)
    track_streams_last_30days = float(recent_data['daily_streams'].sum())
    
    return training_data_id, track_streams_last_30days, months_since_release, track_data

def get_default_decay_parameters():
    """
    Get default decay parameters for track forecasting.
    
    Returns:
        tuple: (fitted_params_df, sp_range, sp_reach) with default values
    """
    # Use parameters imported from utils.decay_rates modules
    # - fitted_params_df from fitted_params.py
    # - sp_range from volume_ranges.py
    # - SP_REACH from sp_reach.py
    
    return fitted_params_df, sp_range, SP_REACH

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
    # We'll use the existing extract_track_metrics function instead of manual calculations
    track_metrics = extract_track_metrics(track_data)
    
    # Step 4: Get default decay parameters
    fitted_params_df, sp_range, sp_reach = get_default_decay_parameters()
    
    # Step 5: Set forecast parameters
    forecast_periods = 24  # 2 years of monthly forecasts
    
    # Simple estimation for stream influence factor based on last 30 days streams
    # This should be improved with actual data from the track
    stream_influence_factor = min(track_metrics['track_streams_last_30days'] * 10, 5000000)
    
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
                'predicted_streams_for_month': int(round(row['predicted_streams_for_month'])),  # Round to integer
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
    now = datetime.now()  # Get current time once for all forecasts
    
    for forecast in forecasts:
        # Use integers directly, cast only when needed
        forecast_values.append((
            forecast['month'],
            forecast['predicted_streams_for_month'],  # Already converted to int above
            forecast['segment_used'],
            forecast['time_used'],
            forecast['cm_track_id'],
            forecast['cm_artist_id'],
            forecast['training_data_id'],
            now
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

def get_batch_track_artist_ids(track_ids):
    """
    Get artist IDs for multiple tracks in a single database query.
    
    Args:
        track_ids: List of ChartMetric track IDs
        
    Returns:
        dict: Dictionary mapping track IDs to artist IDs
    """
    if not track_ids:
        return {}
    
    # Build the SQL placeholders for the IN clause
    placeholders = ','.join(['%s'] * len(track_ids))
    
    query = f"""
    SELECT DISTINCT cm_track_id, cm_artist_id 
    FROM backtest_track_daily_training_data 
    WHERE cm_track_id IN ({placeholders})
    """
    
    results = fetch_all(query, params=track_ids)
    
    # Convert to dictionary
    track_artist_map = {int(row[0]): int(row[1]) for row in results}
    
    return track_artist_map

def get_batch_training_data(track_artist_pairs):
    """
    Get training data for multiple tracks in a single database query.
    
    Args:
        track_artist_pairs: List of (track_id, artist_id) tuples
        
    Returns:
        dict: Dictionary mapping track IDs to their training data
    """
    if not track_artist_pairs:
        return {}
    
    # Extract just the track IDs for the query
    track_ids = [pair[0] for pair in track_artist_pairs]
    
    # Build the SQL placeholders for the IN clause
    placeholders = ','.join(['%s'] * len(track_ids))
    
    query = f"""
    SELECT id, "Date", "CumulativeStreams", days_from_release, cm_track_id, cm_artist_id 
    FROM backtest_track_daily_training_data 
    WHERE cm_track_id IN ({placeholders})
    ORDER BY cm_track_id, "Date" ASC
    """
    
    results = fetch_all(query, params=track_ids)
    
    # Group by track ID
    training_data_by_track = {}
    for row in results:
        track_id = int(row[4])  # cm_track_id index
        if track_id not in training_data_by_track:
            training_data_by_track[track_id] = []
        training_data_by_track[track_id].append(row)
    
    # Convert each track's data to DataFrame
    for track_id, rows in training_data_by_track.items():
        df = pd.DataFrame(rows, columns=['id', 'Date', 'CumulativeStreams', 'days_from_release', 'cm_track_id', 'cm_artist_id'])
        
        # Convert types
        if 'CumulativeStreams' in df.columns:
            df['CumulativeStreams'] = df['CumulativeStreams'].astype(float)
        if 'days_from_release' in df.columns:
            df['days_from_release'] = df['days_from_release'].astype(float)
        
        # Calculate daily_streams by differencing CumulativeStreams
        df = df.sort_values('Date')
        df['daily_streams'] = df['CumulativeStreams'].diff().fillna(df['CumulativeStreams'])
        
        # Store prepared DataFrame
        training_data_by_track[track_id] = df
    
    return training_data_by_track

def get_batch_artist_mldrs(artist_ids):
    """
    Retrieve MLDRs for multiple artists in a single database query.
    
    Args:
        artist_ids: List of ChartMetric artist IDs
        
    Returns:
        dict: Dictionary mapping artist IDs to their MLDR values
    """
    if not artist_ids:
        return {}
    
    # Build the SQL placeholders for the IN clause
    placeholders = ','.join(['%s'] * len(artist_ids))
    
    query = f"""
    SELECT cm_artist_id, mldr 
    FROM backtest_artist_mldr
    WHERE cm_artist_id IN ({placeholders})
    ORDER BY created_at DESC
    """
    
    results = fetch_all(query, params=artist_ids)
    
    # Process results - keep only the most recent MLDR for each artist
    mldr_map = {}
    for row in results:
        artist_id = int(row[0])
        mldr_value = float(row[1]) if row[1] is not None else None
        
        # Only add if we haven't seen this artist yet (since results are ordered by created_at DESC)
        if artist_id not in mldr_map:
            mldr_map[artist_id] = mldr_value
    
    return mldr_map

def process_track_batch(track_ids, batch_num, total_batches):
    """
    Process a batch of tracks efficiently.
    
    Args:
        track_ids: List of track IDs to process
        batch_num: Current batch number (for logging)
        total_batches: Total number of batches (for logging)
    
    Returns:
        tuple: (success_count, failure_count, all_forecasts)
    """
    logger.info(f"Processing batch {batch_num}/{total_batches} with {len(track_ids)} tracks")
    batch_start_time = time.time()
    
    # Get all artist IDs in a single query
    track_artist_map = get_batch_track_artist_ids(track_ids)
    logger.info(f"Retrieved {len(track_artist_map)} artist IDs in {time.time() - batch_start_time:.2f} seconds")
    
    # Create list of valid (track_id, artist_id) pairs
    valid_pairs = [(track_id, artist_id) for track_id, artist_id in track_artist_map.items()]
    
    # Get all artist MLDRs in a single query
    artist_ids = list(set([artist_id for _, artist_id in valid_pairs]))
    mldr_retrieval_start = time.time()
    artist_mldr_map = get_batch_artist_mldrs(artist_ids)
    logger.info(f"Retrieved {len(artist_mldr_map)} artist MLDRs in {time.time() - mldr_retrieval_start:.2f} seconds")
    
    # Get all training data in a single query
    training_data_retrieval_start = time.time()
    training_data_map = get_batch_training_data(valid_pairs)
    logger.info(f"Retrieved training data for {len(training_data_map)} tracks in {time.time() - training_data_retrieval_start:.2f} seconds")
    
    # Process each track and collect forecasts
    all_forecasts = []
    success_count = 0
    failure_count = 0
    
    forecast_generation_start = time.time()
    for track_id, artist_id in valid_pairs:
        # Skip if we don't have training data
        if track_id not in training_data_map:
            logger.warning(f"No training data found for track ID {track_id}")
            failure_count += 1
            continue
        
        track_data = training_data_map[track_id]
        
        # Extract first training data ID for reference
        training_data_id = track_data.iloc[0]['id']
        
        # Get MLDR for artist from our batch-fetched map
        mldr = artist_mldr_map.get(artist_id, None)
        if mldr is None:
            logger.warning(f"No MLDR found for artist {artist_id}, using default value")
            mldr = 0.05  # Default value
        
        # Extract track metrics
        try:
            track_metrics = extract_track_metrics(track_data)
            
            # Get default decay parameters
            fitted_params_df, sp_range, sp_reach = get_default_decay_parameters()
            
            # Set forecast parameters
            forecast_periods = 24  # 2 years of monthly forecasts
            
            # Estimate stream influence factor
            stream_influence_factor = min(track_metrics['track_streams_last_30days'] * 10, 5000000)
            
            # Generate forecasts
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
            
            # Format forecasts for database storage
            for _, row in forecast_result['forecast_df'].iterrows():
                forecast = {
                    'month': int(row['month']),
                    'predicted_streams_for_month': int(round(row['predicted_streams_for_month'])),  # Round to integer
                    'segment_used': int(row['segment_used']),
                    'time_used': int(row['time_used']),
                    'cm_track_id': track_id,
                    'cm_artist_id': artist_id,
                    'training_data_id': training_data_id
                }
                all_forecasts.append(forecast)
            
            success_count += 1
        except Exception as e:
            logger.error(f"Error generating forecasts for track {track_id}: {str(e)}")
            failure_count += 1
    
    logger.info(f"Generated forecasts for {success_count} tracks in {time.time() - forecast_generation_start:.2f} seconds")
    
    # Return results
    batch_time = time.time() - batch_start_time
    logger.info(f"Batch {batch_num}/{total_batches} completed in {batch_time:.2f} seconds")
    
    return success_count, failure_count, all_forecasts

def process_batch_wrapper(args):
    """
    Wrapper function for parallel batch processing.
    
    Args:
        args: Tuple of (batch_track_ids, batch_num, total_batches)
        
    Returns:
        tuple: Result of process_track_batch function
    """
    return process_track_batch(*args)

def main(single_track_id=None):
    """
    Main function to orchestrate the forecasting process.
    
    Args:
        single_track_id: Optional ChartMetric track ID to process just one track
    """
    start_time = time.time()
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
        # Process all tracks that need forecasting in batches
        all_tracks = get_tracks_to_forecast()
        logger.info(f"Found {len(all_tracks)} tracks to forecast")
        
        # Process in batches of 100 tracks
        batch_size = 100
        total_batches = (len(all_tracks) + batch_size - 1) // batch_size  # Ceiling division
        
        # Track overall stats
        total_success = 0
        total_failure = 0
        
        # Prepare batch arguments for parallel processing
        batch_args = []
        for batch_num in range(1, total_batches + 1):
            # Get batch of track IDs
            start_idx = (batch_num - 1) * batch_size
            end_idx = min(start_idx + batch_size, len(all_tracks))
            batch_tracks = all_tracks[start_idx:end_idx]
            batch_track_ids = [track[0] for track in batch_tracks]
            
            # Add batch arguments to list
            batch_args.append((batch_track_ids, batch_num, total_batches))
        
        # Use parallel processing to process batches simultaneously
        logger.info(f"Starting parallel processing of {total_batches} batches with {os.cpu_count()} CPU cores")
        parallel_start_time = time.time()
        
        # Determine optimal number of workers based on CPU cores
        max_workers = min(os.cpu_count(), total_batches)
        logger.info(f"Using {max_workers} parallel workers")
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all batch jobs
            futures = [executor.submit(process_batch_wrapper, args) for args in batch_args]
            
            # Process results as they complete
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                try:
                    success_count, failure_count, batch_forecasts = future.result()
                    
                    # Store all forecasts for this batch at once
                    if batch_forecasts:
                        storage_start = time.time()
                        rows_stored = store_forecasts(batch_forecasts)
                        logger.info(f"Stored {rows_stored} forecast periods in {time.time() - storage_start:.2f} seconds")
                    
                    # Update stats
                    total_success += success_count
                    total_failure += failure_count
                    completed += 1
                    
                    # Log progress percentage
                    progress_pct = (completed / total_batches) * 100
                    logger.info(f"Progress: {completed}/{total_batches} batches ({progress_pct:.1f}%) - Success: {total_success}, Failure: {total_failure}")
                    
                except Exception as e:
                    logger.error(f"Batch processing failed: {str(e)}")
        
        parallel_time = time.time() - parallel_start_time
        logger.info(f"Parallel processing completed in {parallel_time:.2f} seconds")
        logger.info(f"Completed {total_batches} batches with parallel processing. Total success: {total_success}, Total failure: {total_failure}")
    
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    
    logger.info(f"Track streams forecasting process completed in {hours}h {minutes}m {seconds}s ({total_time:.2f} seconds)")
    
    if not single_track_id:
        logger.info(f"Summary: {total_success} tracks succeeded, {total_failure} tracks failed")
        if total_success > 0:
            logger.info(f"Average time per successful track: {total_time/total_success:.2f} seconds")

if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Generate track stream forecasts')
    parser.add_argument('--track_id', type=int, help='Process a single track with the given ChartMetric track ID')
    
    args = parser.parse_args()
    
    # Run with the provided track ID if specified
    main(single_track_id=args.track_id) 