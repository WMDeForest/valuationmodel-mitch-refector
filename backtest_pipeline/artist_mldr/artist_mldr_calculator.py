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
import time
import threading

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
# Use a more conservative worker count to avoid overloading the SSH tunnel
DEFAULT_WORKERS = min(4, max(1, multiprocessing.cpu_count() // 2))  # Lower worker count
MIN_CONNECTIONS = 2
MAX_CONNECTIONS = 10  # Reduced max connections to avoid overwhelming the tunnel

# Global SSH tunnel and connection pools
ssh_tunnel = None
db_connection_pool = None
sqlalchemy_engine = None
ssh_tunnel_lock = threading.Lock()  # Add a lock for thread-safe tunnel access

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
    
    # Use a lock to prevent multiple threads from creating tunnels simultaneously
    with ssh_tunnel_lock:
        if ssh_tunnel is None or not ssh_tunnel.is_active:
            try:
                # Close any existing tunnel
                if ssh_tunnel is not None:
                    try:
                        ssh_tunnel.close()
                    except:
                        pass
                
                safe_log('INFO', "Setting up new SSH tunnel...")
                
                # Create SSH tunnel with conservative connection settings
                tunnel = sshtunnel.SSHTunnelForwarder(
                    (SSH_PARAMS['ssh_host']),
                    ssh_username=SSH_PARAMS['ssh_username'],
                    ssh_password=SSH_PARAMS['ssh_password'],
                    remote_bind_address=SSH_PARAMS['remote_bind_address'],
                    local_bind_address=('localhost', 0),  # Use random local port
                    set_keepalive=5,                     # Keep tunnel alive with packets every 5 seconds
                    compression=True,                    # Enable compression for better performance
                    allow_agent=False                    # Don't use SSH agent
                )
                
                # Start the tunnel
                tunnel.start()
                
                # Wait a moment to ensure it's fully established
                time.sleep(1)
                
                if not tunnel.is_active:
                    raise Exception("Failed to establish SSH tunnel")
                
                ssh_tunnel = tunnel
                safe_log('INFO', f"SSH tunnel established on local port {tunnel.local_bind_port}")
                
                # Add a heartbeat thread to keep the tunnel alive
                def heartbeat():
                    while ssh_tunnel and ssh_tunnel.is_active:
                        try:
                            # Send a keep-alive packet
                            if hasattr(ssh_tunnel.ssh_transport, 'send_ignore'):
                                ssh_tunnel.ssh_transport.send_ignore()
                            time.sleep(10)  # Send heartbeat every 10 seconds
                        except:
                            # If there's an error, break out of the loop
                            break
                
                heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
                heartbeat_thread.start()
                
                return tunnel
            except Exception as e:
                safe_log('ERROR', f"Error setting up SSH tunnel: {e}")
                if 'tunnel' in locals() and tunnel.is_active:
                    try:
                        tunnel.close()
                    except:
                        pass
                raise
    
    return ssh_tunnel

def initialize_connection_pool():
    """Initialize the database connection pool using SSH tunnel."""
    global db_connection_pool, ssh_tunnel
    
    if db_connection_pool is not None:
        return db_connection_pool
    
    # Ensure we have a working SSH tunnel
    retry_count = 0
    while retry_count < 3:
        try:
            # Make sure we have an SSH tunnel
            tunnel = setup_ssh_tunnel()
            
            # Create connection parameters with tunnel's local port
            conn_params = DB_PARAMS.copy()
            conn_params['port'] = tunnel.local_bind_port
            
            # Create connection pool
            connection_pool = pool.ThreadedConnectionPool(
                minconn=MIN_CONNECTIONS,
                maxconn=MAX_CONNECTIONS,
                **conn_params
            )
            
            # Test a connection to make sure it works
            test_conn = connection_pool.getconn()
            connection_pool.putconn(test_conn)
            
            db_connection_pool = connection_pool
            safe_log('INFO', f"Database connection pool initialized with {MIN_CONNECTIONS}-{MAX_CONNECTIONS} connections")
            return connection_pool
        except Exception as e:
            retry_count += 1
            safe_log('ERROR', f"Error initializing connection pool (attempt {retry_count}/3): {e}")
            
            # If we have a tunnel issue, reset it and retry
            if ssh_tunnel is not None:
                try:
                    ssh_tunnel.close()
                except:
                    pass
                ssh_tunnel = None
                
            # Wait before retrying
            time.sleep(2)
    
    # If we exhaust all retries, raise the error
    raise Exception("Failed to initialize connection pool after multiple attempts")

def initialize_sqlalchemy_engine():
    """Initialize SQLAlchemy engine with connection pooling."""
    global sqlalchemy_engine, ssh_tunnel
    
    if sqlalchemy_engine is not None:
        return sqlalchemy_engine
    
    # Ensure we have a working SSH tunnel
    retry_count = 0
    while retry_count < 3:
        try:
            # Make sure we have an SSH tunnel
            tunnel = setup_ssh_tunnel()
            
            # Create connection string using tunnel's local port
            conn_string = f"postgresql://{DB_PARAMS['user']}:{DB_PARAMS['password']}@localhost:{tunnel.local_bind_port}/{DB_PARAMS['dbname']}"
            
            # Create engine with connection pooling
            engine = create_engine(
                conn_string,
                poolclass=sa_pool.QueuePool,
                pool_size=MAX_CONNECTIONS,
                max_overflow=2,
                pool_timeout=30,
                pool_recycle=1800  # Recycle connections after 30 minutes
            )
            
            # Test the connection without using execute
            with engine.connect() as conn:
                pass  # Just open and close the connection to test it
            
            sqlalchemy_engine = engine
            safe_log('INFO', "SQLAlchemy engine initialized with connection pooling")
            return engine
        except Exception as e:
            retry_count += 1
            safe_log('ERROR', f"Error creating SQLAlchemy engine (attempt {retry_count}/3): {e}")
            
            # If we have a tunnel issue, reset it and retry
            if ssh_tunnel is not None:
                try:
                    ssh_tunnel.close()
                except:
                    pass
                ssh_tunnel = None
                
            # Wait before retrying
            time.sleep(2)
    
    # If we exhaust all retries, raise the error
    raise Exception("Failed to initialize SQLAlchemy engine after multiple attempts")

def get_db_connection():
    """Get a connection from the connection pool."""
    global db_connection_pool
    
    # Initialize pool if not already done
    if db_connection_pool is None:
        initialize_connection_pool()
    
    # Retry logic for getting a connection
    retry_count = 0
    while retry_count < 3:
        try:
            # Get connection from pool
            connection = db_connection_pool.getconn()
            
            # Test the connection with a simple query
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                
            return connection
        except Exception as e:
            retry_count += 1
            safe_log('ERROR', f"Error getting connection from pool (attempt {retry_count}/3): {e}")
            
            # If the connection failed, reinitialize everything
            if retry_count >= 2:
                safe_log('WARNING', "Attempting to rebuild connection pool...")
                try:
                    if db_connection_pool is not None:
                        db_connection_pool.closeall()
                except:
                    pass
                    
                db_connection_pool = None
                
                # Reset SSH tunnel
                if ssh_tunnel is not None:
                    try:
                        ssh_tunnel.close()
                    except:
                        pass
                ssh_tunnel = None
                
                # Reinitialize
                initialize_connection_pool()
            
            time.sleep(1)  # Wait before retrying
    
    raise Exception("Failed to get a database connection after multiple attempts")

def release_db_connection(connection):
    """Release a connection back to the pool."""
    global db_connection_pool
    
    if db_connection_pool is not None and connection is not None:
        try:
            db_connection_pool.putconn(connection)
        except Exception as e:
            safe_log('ERROR', f"Error returning connection to pool: {e}")

def get_sqlalchemy_engine():
    """Get the SQLAlchemy engine with connection pooling."""
    global sqlalchemy_engine
    
    # Initialize engine if not already done
    if sqlalchemy_engine is None:
        initialize_sqlalchemy_engine()
    
    # Return the engine - no test query needed
    return sqlalchemy_engine

def close_all_connections():
    """Close all database connections and SSH tunnel when the script exits."""
    global db_connection_pool, sqlalchemy_engine, ssh_tunnel
    
    safe_log('INFO', "Closing all database connections and SSH tunnel")
    
    # Close psycopg2 connection pool
    if db_connection_pool is not None:
        try:
            db_connection_pool.closeall()
        except Exception as e:
            safe_log('ERROR', f"Error closing connection pool: {e}")
        db_connection_pool = None
    
    # Close SQLAlchemy engine
    if sqlalchemy_engine is not None:
        try:
            sqlalchemy_engine.dispose()
        except Exception as e:
            safe_log('ERROR', f"Error disposing SQLAlchemy engine: {e}")
        sqlalchemy_engine = None
    
    # Close SSH tunnel
    if ssh_tunnel is not None and ssh_tunnel.is_active:
        try:
            ssh_tunnel.close()
        except Exception as e:
            safe_log('ERROR', f"Error closing SSH tunnel: {e}")
        ssh_tunnel = None

# Register cleanup function to be called when the script exits
atexit.register(close_all_connections)

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
    
    engine = get_sqlalchemy_engine()
    try:
        # Get all data for the batch of artists in one query
        df = pd.read_sql(query, engine)
        
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
                execute_values(cursor, insert_query, insert_values)
                safe_log('INFO', f"Inserted {len(new_data)} new MLDR records")
                
                # Mark as success
                for artist_id, _ in new_data:
                    result[artist_id] = True
            
            # Update existing records in batch
            if update_data:
                execute_values(cursor, update_query, update_data)
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
    
    # Update connection pool size if needed
    if args.max_connections > MAX_CONNECTIONS:
        MAX_CONNECTIONS = args.max_connections
    
    safe_log('INFO', f"Starting Artist MLDR calculation with {worker_count} workers and {MAX_CONNECTIONS} max connections")
    
    # Process one artist at a time for the most reliable operation
    if args.artist_id is not None:
        # For a single artist, don't use parallelism
        safe_log('INFO', f"Processing single artist (ID: {args.artist_id}) without parallelism")
        try:
            # Initialize SSH tunnel and connection once
            setup_ssh_tunnel()
            
            # Process the artist
            result = process_artist_batch([args.artist_id])
            if result[args.artist_id]['success']:
                safe_log('INFO', f"Successfully processed artist {args.artist_id}")
            else:
                safe_log('WARNING', f"Failed to process artist {args.artist_id}: {result[args.artist_id]['reason']}")
        except Exception as e:
            safe_log('ERROR', f"Error processing artist {args.artist_id}: {e}")
        finally:
            close_all_connections()
    else:
        # For multiple artists, use parallelism with careful connection management
        try:
            # Initialize connection pools before starting parallel processing
            setup_ssh_tunnel()  # Ensure tunnel is established first
            
            # Process all artists in parallel with batch approach
            process_all_artists_parallel(worker_count, args.artist_limit, args.artist_id)
        except Exception as e:
            safe_log('ERROR', f"Fatal error: {e}")
        finally:
            close_all_connections()
    
    safe_log('INFO', "Artist MLDR calculation complete")