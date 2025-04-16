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
import psycopg2
from psycopg2 import pool
from psycopg2.extras import execute_values
from sqlalchemy import create_engine, pool as sa_pool
import logging
from typing import Dict, List, Tuple, Optional
import sshtunnel
import atexit
import concurrent.futures
import multiprocessing
import argparse

# Import the MLDR calculation function from our existing codebase
from utils.decay_models import analyze_listener_decay

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('artist_mldr_calculator')

# SSH connection parameters
SSH_PARAMS = {
    'ssh_host': '135.181.17.84',
    'ssh_username': 'root',
    'ssh_password': 'E7Kigxn3UtQiTx',
    'remote_bind_address': ('localhost', 5432)
}

# Database connection parameters
DB_PARAMS = {
    'dbname': 'postgres',
    'user': 'postgres',
    'password': 'Zafe5Ph353',
    'host': 'localhost',  # Connect via SSH tunnel
    'port': '5432'
}

# Default parallelism and connection pool settings
DEFAULT_WORKERS = max(1, multiprocessing.cpu_count() - 1)  # Default to CPU count - 1
MIN_CONNECTIONS = 2
MAX_CONNECTIONS = 20  # Increased for parallel processing

# Global SSH tunnel and connection pools
ssh_tunnel = None
db_connection_pool = None
sqlalchemy_engine = None

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

def setup_ssh_tunnel():
    """Set up SSH tunnel for database connections."""
    global ssh_tunnel
    if ssh_tunnel is None or not ssh_tunnel.is_active:
        try:
            # Create SSH tunnel
            tunnel = sshtunnel.SSHTunnelForwarder(
                (SSH_PARAMS['ssh_host']),
                ssh_username=SSH_PARAMS['ssh_username'],
                ssh_password=SSH_PARAMS['ssh_password'],
                remote_bind_address=SSH_PARAMS['remote_bind_address'],
                local_bind_address=('localhost', 0)  # Use random local port
            )
            tunnel.start()
            ssh_tunnel = tunnel
            safe_log('INFO', f"SSH tunnel established on local port {tunnel.local_bind_port}")
            return tunnel
        except Exception as e:
            safe_log('ERROR', f"Error setting up SSH tunnel: {e}")
            if 'tunnel' in locals() and tunnel.is_active:
                tunnel.close()
            raise
    return ssh_tunnel

def initialize_connection_pool():
    """Initialize the database connection pool using SSH tunnel."""
    global db_connection_pool, ssh_tunnel
    
    if db_connection_pool is not None:
        return db_connection_pool
        
    # Make sure we have an SSH tunnel
    tunnel = setup_ssh_tunnel()
    
    try:
        # Create connection parameters with tunnel's local port
        conn_params = DB_PARAMS.copy()
        conn_params['port'] = tunnel.local_bind_port
        
        # Create connection pool
        connection_pool = pool.ThreadedConnectionPool(
            minconn=MIN_CONNECTIONS,
            maxconn=MAX_CONNECTIONS,
            **conn_params
        )
        db_connection_pool = connection_pool
        safe_log('INFO', f"Database connection pool initialized with {MIN_CONNECTIONS}-{MAX_CONNECTIONS} connections")
        return connection_pool
    except Exception as e:
        safe_log('ERROR', f"Error initializing connection pool: {e}")
        raise

def initialize_sqlalchemy_engine():
    """Initialize SQLAlchemy engine with connection pooling."""
    global sqlalchemy_engine, ssh_tunnel
    
    if sqlalchemy_engine is not None:
        return sqlalchemy_engine
        
    # Make sure we have an SSH tunnel
    tunnel = setup_ssh_tunnel()
    
    try:
        # Create connection string using tunnel's local port
        conn_string = f"postgresql://{DB_PARAMS['user']}:{DB_PARAMS['password']}@localhost:{tunnel.local_bind_port}/{DB_PARAMS['dbname']}"
        
        # Create engine with connection pooling
        engine = create_engine(
            conn_string,
            poolclass=sa_pool.QueuePool,
            pool_size=MAX_CONNECTIONS,
            max_overflow=5,  # Increased for parallel processing
            pool_timeout=30,
            pool_recycle=1800  # Recycle connections after 30 minutes
        )
        
        sqlalchemy_engine = engine
        safe_log('INFO', "SQLAlchemy engine initialized with connection pooling")
        return engine
    except Exception as e:
        safe_log('ERROR', f"Error creating SQLAlchemy engine: {e}")
        raise

def get_db_connection():
    """Get a connection from the connection pool."""
    global db_connection_pool
    
    # Initialize pool if not already done
    if db_connection_pool is None:
        initialize_connection_pool()
        
    try:
        # Get connection from pool
        connection = db_connection_pool.getconn()
        return connection
    except Exception as e:
        safe_log('ERROR', f"Error getting connection from pool: {e}")
        raise

def release_db_connection(connection):
    """Release a connection back to the pool."""
    global db_connection_pool
    
    if db_connection_pool is not None and connection is not None:
        db_connection_pool.putconn(connection)

def get_sqlalchemy_engine():
    """Get the SQLAlchemy engine with connection pooling."""
    global sqlalchemy_engine
    
    # Initialize engine if not already done
    if sqlalchemy_engine is None:
        initialize_sqlalchemy_engine()
        
    return sqlalchemy_engine

def close_all_connections():
    """Close all database connections and SSH tunnel when the script exits."""
    global db_connection_pool, sqlalchemy_engine, ssh_tunnel
    
    safe_log('INFO', "Closing all database connections and SSH tunnel")
    
    # Close psycopg2 connection pool
    if db_connection_pool is not None:
        db_connection_pool.closeall()
        db_connection_pool = None
    
    # Close SQLAlchemy engine
    if sqlalchemy_engine is not None:
        sqlalchemy_engine.dispose()
        sqlalchemy_engine = None
    
    # Close SSH tunnel
    if ssh_tunnel is not None and ssh_tunnel.is_active:
        ssh_tunnel.close()
        ssh_tunnel = None

# Register cleanup function to be called when the script exits
atexit.register(close_all_connections)

def get_all_artists():
    """Get a list of all unique artist IDs from the training data table."""
    query = """
    SELECT DISTINCT cm_artist_id 
    FROM backtest_artist_daily_training_data
    """
    
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(query)
            artists = [row[0] for row in cursor.fetchall()]
            safe_log('INFO', f"Found {len(artists)} unique artists")
            return artists
    except Exception as e:
        safe_log('ERROR', f"Error fetching artists: {e}")
        raise
    finally:
        release_db_connection(connection)

def get_artist_data(artist_id: int) -> pd.DataFrame:
    """
    Get monthly listener data for a specific artist.
    
    Args:
        artist_id: The Chartmetric artist ID
        
    Returns:
        DataFrame with Date and Monthly Listeners columns
    """
    query = """
    SELECT date, monthly_listeners
    FROM backtest_artist_daily_training_data
    WHERE cm_artist_id = %s
    ORDER BY date
    """
    
    engine = get_sqlalchemy_engine()
    try:
        df = pd.read_sql(query, engine, params=(artist_id,))
        
        if len(df) == 0:
            safe_log('WARNING', f"No data found for artist {artist_id}")
            return None
            
        # Rename columns to match what analyze_listener_decay expects
        df.rename(columns={
            'date': 'Date',
            'monthly_listeners': 'Monthly Listeners'
        }, inplace=True)
        
        # Ensure Date is datetime
        df['Date'] = pd.to_datetime(df['Date'])
        
        safe_log('INFO', f"Retrieved {len(df)} data points for artist {artist_id}")
        return df
    except Exception as e:
        safe_log('ERROR', f"Error retrieving data for artist {artist_id}: {e}")
        return None

def calculate_artist_mldr(artist_data: pd.DataFrame) -> float:
    """
    Calculate MLDR for an artist using the analyze_listener_decay function.
    
    Args:
        artist_data: DataFrame with Date and Monthly Listeners columns
        
    Returns:
        MLDR value (float)
    """
    try:
        # Log the date range we're analyzing
        min_date = artist_data['Date'].min().strftime('%Y-%m-%d')
        max_date = artist_data['Date'].max().strftime('%Y-%m-%d')
        date_range_days = (artist_data['Date'].max() - artist_data['Date'].min()).days
        num_data_points = len(artist_data)
        
        safe_log('INFO', f"Raw date range: {min_date} to {max_date} ({date_range_days} days, {num_data_points} data points)")
        
        # Log a few sample data points to verify data content
        safe_log('INFO', f"Data sample (first 3 points): {artist_data.head(3)[['Date', 'Monthly Listeners']].to_dict('records')}")
        safe_log('INFO', f"Data sample (last 3 points): {artist_data.tail(3)[['Date', 'Monthly Listeners']].to_dict('records')}")
        
        # Run the analysis - this handles data cleaning, anomaly detection, and curve fitting
        results = analyze_listener_decay(artist_data)
        
        # Extract MLDR from results
        mldr = results['mldr']
        
        # Log MLDR and normalized date range information
        safe_log('INFO', f"Calculated MLDR: {mldr}")
        
        # Log normalized date range if available
        if 'normalized_start_date' in results and 'normalized_end_date' in results:
            normalized_start = results['normalized_start_date'].strftime('%Y-%m-%d') if hasattr(results['normalized_start_date'], 'strftime') else results['normalized_start_date']
            normalized_end = results['normalized_end_date'].strftime('%Y-%m-%d') if hasattr(results['normalized_end_date'], 'strftime') else results['normalized_end_date']
            safe_log('INFO', f"Normalized date range used for MLDR calculation: {normalized_start} to {normalized_end}")
        
        if 'date_filtered_listener_data' in results:
            filtered_min_date = results['date_filtered_listener_data']['Date'].min().strftime('%Y-%m-%d')
            filtered_max_date = results['date_filtered_listener_data']['Date'].max().strftime('%Y-%m-%d')
            safe_log('INFO', f"Filtered data date range: {filtered_min_date} to {filtered_max_date}")
        
        if 'used_date_range' in results:
            safe_log('INFO', f"Actual dates used in calculation: {results['used_date_range']}")
        if 'outliers_removed' in results and results['outliers_removed']:
            safe_log('INFO', f"Outliers were removed: {results['outliers_removed']}")
        
        return mldr
    except Exception as e:
        safe_log('ERROR', f"Error calculating MLDR: {e}")
        return None

def store_artist_mldr(artist_id: int, mldr: float):
    """
    Store the calculated MLDR in the database.
    
    Args:
        artist_id: The Chartmetric artist ID
        mldr: The calculated MLDR value
    """
    # First check if a record already exists for this artist
    check_query = """
    SELECT id FROM backtest_artist_mldr 
    WHERE cm_artist_id = %s
    """
    
    # Insert or update query based on whether the record exists
    insert_query = """
    INSERT INTO backtest_artist_mldr 
    (cm_artist_id, mldr, created_at)
    VALUES (%s, %s, NOW())
    """
    
    update_query = """
    UPDATE backtest_artist_mldr
    SET mldr = %s, created_at = NOW()
    WHERE cm_artist_id = %s
    """
    
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            # Check if record exists
            cursor.execute(check_query, (artist_id,))
            record = cursor.fetchone()
            
            if record:
                # Update existing record
                cursor.execute(update_query, (mldr, artist_id))
                safe_log('INFO', f"Updated MLDR {mldr} for artist {artist_id}")
            else:
                # Insert new record
                cursor.execute(insert_query, (artist_id, mldr))
                safe_log('INFO', f"Inserted new MLDR {mldr} for artist {artist_id}")
                
        connection.commit()
        return True
    except Exception as e:
        connection.rollback()
        safe_log('ERROR', f"Error storing MLDR for artist {artist_id}: {e}")
        return False
    finally:
        release_db_connection(connection)

def process_artist(artist_id: int) -> Dict:
    """
    Process a single artist - gets data, calculates MLDR, and stores it.
    This function is designed to be used with parallel processing.
    
    Args:
        artist_id: The Chartmetric artist ID
        
    Returns:
        Dict with processing results
    """
    result = {
        'artist_id': artist_id,
        'success': False,
        'reason': None
    }
    
    # Get artist data
    artist_data = get_artist_data(artist_id)
    if artist_data is None or len(artist_data) < 3:  # Need at least 3 data points for fitting
        result['reason'] = "Insufficient data"
        return result
        
    # Calculate MLDR
    mldr = calculate_artist_mldr(artist_data)
    if mldr is None:
        result['reason'] = "Failed to calculate MLDR"
        return result
        
    # Store MLDR in database
    if store_artist_mldr(artist_id, mldr):
        result['success'] = True
    else:
        result['reason'] = "Failed to store MLDR"
    
    return result

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
    
    # Create thread pool and process artists in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_artist = {executor.submit(process_artist, artist_id): artist_id for artist_id in artists}
        
        # Process results as they complete
        for i, future in enumerate(concurrent.futures.as_completed(future_to_artist)):
            artist_id = future_to_artist[future]
            try:
                result = future.result()
                if result['success']:
                    successful += 1
                else:
                    failed += 1
                    safe_log('WARNING', f"Failed to process artist {artist_id}: {result['reason']}")
                
                # Log progress
                if (i+1) % 10 == 0 or (i+1) == total_artists:
                    safe_log('INFO', f"Progress: {i+1}/{total_artists} artists processed ({successful} successful, {failed} failed)")
                    
            except Exception as e:
                failed += 1
                safe_log('ERROR', f"Exception processing artist {artist_id}: {e}")
    
    # Log summary
    safe_log('INFO', f"Processing complete: {successful} successful, {failed} failed, {total_artists} total")

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Update connection pool size if needed
    if args.max_connections > MAX_CONNECTIONS:
        MAX_CONNECTIONS = args.max_connections
    
    safe_log('INFO', f"Starting Artist MLDR calculation with {args.workers} workers and {MAX_CONNECTIONS} max connections")
    
    # Initialize connection pools at the beginning
    initialize_connection_pool()
    initialize_sqlalchemy_engine()
    
    try:
        # Process all artists in parallel
        process_all_artists_parallel(args.workers, args.artist_limit, args.artist_id)
    finally:
        # Ensure all connections are closed
        close_all_connections()
        
    safe_log('INFO', "Artist MLDR calculation complete")