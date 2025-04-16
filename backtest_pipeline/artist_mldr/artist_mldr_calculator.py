#!/usr/bin/env python3
# coding: utf-8

"""
Artist MLDR Calculator

This script calculates the Monthly Listener Decay Rate (MLDR) for all artists
in the backtest_artist_daily_training_data table and stores the results in 
the backtest_artist_mldr table.

The MLDR represents how quickly an artist loses listeners over time and is a 
key input for track-level streaming forecasts.

## Database Schema

### backtest_artist_daily_training_data

This table contains the historical monthly listener data for artists used to calculate the MLDR.

| Column | Type | Description |
|--------|------|-------------|
| id | UUID | Primary key |
| cm_artist_id | INTEGER | Chartmetric artist ID (unique identifier for the artist) |
| date | DATE | Date of the listener data point |
| monthly_listeners | INTEGER | Number of monthly listeners for the artist on this date |
| created_at | TIMESTAMP WITH TIME ZONE | When the record was created in the database |

### backtest_artist_mldr

This table stores the calculated MLDR values for each artist, derived from their historical monthly listener data.

| Column | Type | Description |
|--------|------|-------------|
| id | UUID | Primary key |
| cm_artist_id | INTEGER | Chartmetric artist ID (foreign key to backtest_artist_daily_training_data) |
| mldr | DECIMAL(12,8) | Monthly Listener Decay Rate - the rate at which an artist's listeners decline monthly |
| created_at | TIMESTAMP WITH TIME ZONE | When the MLDR was first calculated |
| backtest_artist_daily_training_data_id | UUID | Reference to the training data used (foreign key) |

## Database Relationship

The tables are related through the `cm_artist_id` field:

backtest_artist_mldr.cm_artist_id ────→ backtest_artist_daily_training_data.cm_artist_id


This allows the system to:
1. Retrieve an artist's historical listener data from `backtest_artist_daily_training_data`
2. Calculate their MLDR using the `analyze_listener_decay()` function 
3. Store the result in `backtest_artist_mldr`
4. Later look up an artist's pre-calculated MLDR when forecasting track streams

## What is MLDR?

The Monthly Listener Decay Rate (MLDR) represents how quickly an artist loses listeners over time. Mathematically, it's the 'k' value in the exponential decay function:

Where:
- S(t) is the number of listeners at time t
- S₀ is the initial number of listeners
- k is the decay rate (the MLDR)
- t is time in months

A lower MLDR indicates better listener retention (the artist maintains their audience over time), while a higher MLDR indicates faster audience decline.

## Calculator Usage

The `artist_mldr_calculator.py` script:
1. Connects to the database
2. Retrieves all unique artist IDs from the training data
3. For each artist, loads their listener data and calculates the MLDR
4. Stores the results in the `backtest_artist_mldr` table

## Run Examples:

# Test with just one artist:
PYTHONPATH=/Users/mitchdeforest/Documents/valuationmodel-mitch-refector python3 backtest_pipeline/artist_mldr/artist_mldr_calculator.py --artist-limit 1

# Process 10 artists with default settings:
PYTHONPATH=/Users/mitchdeforest/Documents/valuationmodel-mitch-refector python3 backtest_pipeline/artist_mldr/artist_mldr_calculator.py --artist-limit 10

# Process all artists with 8 workers:
PYTHONPATH=/Users/mitchdeforest/Documents/valuationmodel-mitch-refector python3 backtest_pipeline/artist_mldr/artist_mldr_calculator.py --workers 8 --max-connections 25

# Process a specific artist by Chartmetric ID (e.g., 3604555):
PYTHONPATH=/Users/mitchdeforest/Documents/valuationmodel-mitch-refector python3 backtest_pipeline/artist_mldr/artist_mldr_calculator.py --artist-id 3604555
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional
import atexit
import concurrent.futures
import multiprocessing
import argparse
import time
import threading

# Import the MLDR calculation function from our existing codebase
from utils.decay_models import analyze_listener_decay

# Import database utilities
from utils.database import (
    setup_ssh_tunnel,
    get_db_connection,
    release_db_connection,
    get_sqlalchemy_engine,
    close_all_connections,
    fetch_all,
    execute_batch,
    query_to_dataframe,
    DEFAULT_WORKERS,
    MAX_CONNECTIONS
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('artist_mldr_calculator')

# Process-safe logging
def safe_log(level, message):
    """Thread/process-safe logging function."""
    if level == 'INFO':
        logger.info(f"[Process {os.getpid()}] {message}")
    elif level == 'WARNING':
        logger.warning(f"[Process {os.getpid()}] {message}")
    elif level == 'ERROR':
        logger.error(f"[Process {os.getpid()}] {message}")
    elif level == 'DEBUG':
        logger.debug(f"[Process {os.getpid()}] {message}")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Calculate MLDR for artists in parallel')
    parser.add_argument('--workers', type=int, default=DEFAULT_WORKERS,
                        help=f'Number of worker processes (default: {DEFAULT_WORKERS})')
    parser.add_argument('--max-connections', type=int, default=MAX_CONNECTIONS,
                        help=f'Maximum database connections (default: {MAX_CONNECTIONS})')
    parser.add_argument('--artist-limit', type=int, default=None,
                        help='Limit the number of artists to process (for testing)')
    parser.add_argument('--artist-id', type=int, default=None,
                        help='Process a specific Chartmetric artist ID only')
    return parser.parse_args()

def get_all_artists():
    """Get a list of all unique artist IDs from the training data table."""
    query = """
    SELECT DISTINCT cm_artist_id 
    FROM backtest_artist_daily_training_data
    """
    
    try:
        artists = [row[0] for row in fetch_all(query)]
        safe_log('INFO', f"Found {len(artists)} unique artists")
        return artists
    except Exception as e:
        safe_log('ERROR', f"Error fetching artists: {e}")
        raise

def get_artists_data_batch(artist_ids):
    """
    Get monthly listener data for a batch of artists at once.
    
    Args:
        artist_ids: List of Chartmetric artist IDs
        
    Returns:
        Dict mapping artist_id to its DataFrame with Date and Monthly Listeners columns
    """
    if not artist_ids:
        return {}
    
    # Format the artist IDs for the IN clause
    ids_str = ','.join(str(id) for id in artist_ids)
    
    query = f"""
    SELECT cm_artist_id, date, monthly_listeners
    FROM backtest_artist_daily_training_data
    WHERE cm_artist_id IN ({ids_str})
    ORDER BY cm_artist_id, date
    """
    
    try:
        # Get all data for the batch of artists in one query
        df = query_to_dataframe(query)
        
        if len(df) == 0:
            safe_log('WARNING', f"No data found for any artists in batch")
            return {}
        
        # Split the data by artist_id
        result = {}
        for artist_id in artist_ids:
            artist_df = df[df['cm_artist_id'] == artist_id].copy()
            
            if len(artist_df) == 0:
                safe_log('WARNING', f"No data found for artist {artist_id}")
                continue
                
            # Drop the artist_id column as it's no longer needed
            artist_df = artist_df.drop('cm_artist_id', axis=1)
            
            # Rename columns to match what analyze_listener_decay expects
            artist_df.rename(columns={
                'date': 'Date',
                'monthly_listeners': 'Monthly Listeners'
            }, inplace=True)
            
            # Ensure Date is datetime
            artist_df['Date'] = pd.to_datetime(artist_df['Date'])
            
            safe_log('INFO', f"Retrieved {len(artist_df)} data points for artist {artist_id}")
            result[artist_id] = artist_df
            
        return result
    except Exception as e:
        safe_log('ERROR', f"Error retrieving data for artist batch: {e}")
        return {}

def store_mldr_batch(mldr_data):
    """
    Store multiple calculated MLDRs in a batch.
    
    Args:
        mldr_data: List of tuples (artist_id, mldr)
    
    Returns:
        Dict mapping artist_id to success status
    """
    if not mldr_data:
        return {}
    
    # Query to check which artists already exist
    check_query = """
    SELECT cm_artist_id FROM backtest_artist_mldr
    WHERE cm_artist_id = ANY(%s)
    """
    
    # Query for batch insert of new records
    insert_query = """
    INSERT INTO backtest_artist_mldr 
    (cm_artist_id, mldr, created_at)
    VALUES %s
    """
    
    # Query for batch update of existing records
    update_query = """
    UPDATE backtest_artist_mldr
    SET mldr = data_table.mldr, created_at = NOW()
    FROM (VALUES %s) AS data_table(cm_artist_id, mldr)
    WHERE backtest_artist_mldr.cm_artist_id = data_table.cm_artist_id
    """
    
    connection = get_db_connection()
    result = {artist_id: False for artist_id, _ in mldr_data}
    
    try:
        with connection.cursor() as cursor:
            # Check which artists already exist in the database
            artist_ids = [artist_id for artist_id, _ in mldr_data]
            cursor.execute(check_query, (artist_ids,))
            existing_artists = {row[0] for row in cursor.fetchall()}
            
            # Separate new and existing artists
            new_data = [(artist_id, mldr) for artist_id, mldr in mldr_data 
                        if artist_id not in existing_artists]
            update_data = [(artist_id, mldr) for artist_id, mldr in mldr_data 
                          if artist_id in existing_artists]
            
            # Insert new records in batch
            if new_data:
                # Add timestamp to each record
                insert_values = [(artist_id, mldr, 'NOW()') for artist_id, mldr in new_data]
                execute_batch(cursor, insert_query, insert_values)
                safe_log('INFO', f"Inserted {len(new_data)} new MLDR records")
                
                # Mark as success
                for artist_id, _ in new_data:
                    result[artist_id] = True
            
            # Update existing records in batch
            if update_data:
                execute_batch(cursor, update_query, update_data)
                safe_log('INFO', f"Updated {len(update_data)} existing MLDR records")
                
                # Mark as success
                for artist_id, _ in update_data:
                    result[artist_id] = True
                
        connection.commit()
        return result
    except Exception as e:
        connection.rollback()
        safe_log('ERROR', f"Error in batch MLDR storage: {e}")
        return result
    finally:
        release_db_connection(connection)

def process_artist_batch(artist_ids):
    """
    Process a batch of artists - gets data, calculates MLDR, and stores results in batch.
    
    Args:
        artist_ids: List of Chartmetric artist IDs
        
    Returns:
        Dict mapping artist_id to result dict
    """
    results = {artist_id: {'artist_id': artist_id, 'success': False, 'reason': None} 
              for artist_id in artist_ids}
    
    # Get data for all artists in the batch at once
    batch_data = get_artists_data_batch(artist_ids)
    
    # Process each artist's data and collect MLDRs
    mldr_data = []
    for artist_id in artist_ids:
        # Skip if no data was found
        if artist_id not in batch_data:
            results[artist_id]['reason'] = "No data found"
            continue
            
        artist_data = batch_data[artist_id]
        
        # Check if we have enough data points
        if len(artist_data) < 3:
            results[artist_id]['reason'] = "Insufficient data"
            continue
            
        # Calculate MLDR
        try:
            # Run the analysis and capture important details for logging
            min_date = artist_data['Date'].min().strftime('%Y-%m-%d')
            max_date = artist_data['Date'].max().strftime('%Y-%m-%d')
            date_range_days = (artist_data['Date'].max() - artist_data['Date'].min()).days
            num_data_points = len(artist_data)
            
            safe_log('INFO', f"Raw date range for artist {artist_id}: {min_date} to {max_date} ({date_range_days} days, {num_data_points} data points)")
            
            # Calculate MLDR
            mldr_result = analyze_listener_decay(artist_data)
            mldr = mldr_result['mldr']
            
            # Log results
            normalized_start = mldr_result['normalized_start_date'].strftime('%Y-%m-%d') if 'normalized_start_date' in mldr_result else "Unknown"
            normalized_end = mldr_result['normalized_end_date'].strftime('%Y-%m-%d') if 'normalized_end_date' in mldr_result else "Unknown"
            safe_log('INFO', f"Calculated MLDR for artist {artist_id}: {mldr} (normalized range: {normalized_start} to {normalized_end})")
            
            # Add to batch for storage
            mldr_data.append((artist_id, mldr))
        except Exception as e:
            safe_log('ERROR', f"Error calculating MLDR for artist {artist_id}: {e}")
            results[artist_id]['reason'] = "Error in calculation"
            continue
    
    # Store all MLDRs in a batch
    if mldr_data:
        batch_results = store_mldr_batch(mldr_data)
        
        # Update results based on storage success
        for artist_id, success in batch_results.items():
            if success:
                results[artist_id]['success'] = True
            else:
                results[artist_id]['reason'] = "Failed to store MLDR"
    
    return results

def process_all_artists_parallel(num_workers: int = DEFAULT_WORKERS, artist_limit: Optional[int] = None, specific_artist_id: Optional[int] = None):
    """
    Process all artists in parallel using a thread pool.
    
    Args:
        num_workers: Number of worker threads to use
        artist_limit: Optional limit on the number of artists to process (for testing)
        specific_artist_id: Optional specific artist ID to process
    """
    # Get list of all artists or use specific artist
    if specific_artist_id is not None:
        artists = [specific_artist_id]
        safe_log('INFO', f"Processing only artist ID: {specific_artist_id}")
    else:
        artists = get_all_artists()
        
        # Apply limit if specified
        if artist_limit is not None and artist_limit > 0:
            safe_log('INFO', f"Limiting to {artist_limit} artists for testing")
            artists = artists[:artist_limit]
            
    total_artists = len(artists)
    
    safe_log('INFO', f"Processing {total_artists} artists using {num_workers} workers")
    
    # Initialize counters
    successful = 0
    failed = 0
    
    # Define batch size for processing
    # Use smaller batches for fewer artists to ensure parallelism
    artist_batch_size = min(5, max(1, total_artists // (num_workers * 2)))
    
    # Process artists in batches
    for i in range(0, total_artists, artist_batch_size):
        # Get next batch of artists
        batch = artists[i:i+artist_batch_size]
        
        # Process sub-batches in parallel
        sub_batches = []
        for j in range(0, len(batch), num_workers):
            sub_batch = batch[j:j+num_workers]
            if sub_batch:
                sub_batches.append(sub_batch)
        
        # Process each sub-batch in parallel
        for sub_batch in sub_batches:
            batch_results = process_artist_batch(sub_batch)
            
            # Update counters
            for artist_id, result in batch_results.items():
                if result['success']:
                    successful += 1
                else:
                    failed += 1
                    safe_log('WARNING', f"Failed to process artist {artist_id}: {result['reason']}")
            
        # Log progress after each batch
        completed = min(i + artist_batch_size, total_artists)
        safe_log('INFO', f"Progress: {completed}/{total_artists} artists processed ({successful} successful, {failed} failed)")
            
    # Log summary
    safe_log('INFO', f"Processing complete: {successful} successful, {failed} failed, {total_artists} total")

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Use a conservative worker count to avoid overwhelming the SSH tunnel
    worker_count = min(args.workers, 4)  # Cap at 4 workers
    if worker_count != args.workers and args.workers != DEFAULT_WORKERS:
        safe_log('WARNING', f"Reducing worker count from {args.workers} to {worker_count} to prevent SSH tunnel overload")
    
    # Register the cleanup function to ensure proper shutdown
    atexit.register(close_all_connections)
    
    # Setup the SSH tunnel and connection pools before starting work
    setup_ssh_tunnel()
    
    # Process one artist at a time for the most reliable operation
    if args.artist_id is not None:
        # For a single artist, don't use parallelism
        safe_log('INFO', f"Processing single artist (ID: {args.artist_id}) without parallelism")
        try:
            # Process the artist
            result = process_artist_batch([args.artist_id])
            if result[args.artist_id]['success']:
                safe_log('INFO', f"Successfully processed artist {args.artist_id}")
            else:
                safe_log('WARNING', f"Failed to process artist {args.artist_id}: {result[args.artist_id]['reason']}")
        except Exception as e:
            safe_log('ERROR', f"Error processing artist {args.artist_id}: {e}")
    else:
        # For multiple artists, use parallelism with careful connection management
        try:
            # Process all artists in parallel with batch approach
            process_all_artists_parallel(worker_count, args.artist_limit, args.artist_id)
        except Exception as e:
            safe_log('ERROR', f"Fatal error: {e}")
    
    safe_log('INFO', "Artist MLDR calculation complete")