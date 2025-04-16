#!/usr/bin/env python
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
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import psycopg2
from psycopg2.extras import execute_values
from sqlalchemy import create_engine
import logging
from typing import Dict, List, Tuple

# Import the MLDR calculation function from our existing codebase
from utils.decay_models import analyze_listener_decay

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('artist_mldr_calculator')

# Database connection parameters - replace with your actual values
DB_PARAMS = {
    'dbname': 'your_db_name',
    'user': 'your_username',
    'password': 'your_password',
    'host': 'localhost',
    'port': '5432'
}

def get_db_connection():
    """Create and return a database connection."""
    try:
        connection = psycopg2.connect(**DB_PARAMS)
        return connection
    except Exception as e:
        logger.error(f"Error connecting to database: {e}")
        raise

def get_sqlalchemy_engine():
    """Create and return a SQLAlchemy engine for pandas operations."""
    conn_string = f"postgresql://{DB_PARAMS['user']}:{DB_PARAMS['password']}@{DB_PARAMS['host']}:{DB_PARAMS['port']}/{DB_PARAMS['dbname']}"
    return create_engine(conn_string)

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
            logger.info(f"Found {len(artists)} unique artists")
            return artists
    except Exception as e:
        logger.error(f"Error fetching artists: {e}")
        raise
    finally:
        connection.close()

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
        df = pd.read_sql(query, engine, params=[artist_id])
        
        if len(df) == 0:
            logger.warning(f"No data found for artist {artist_id}")
            return None
            
        # Rename columns to match what analyze_listener_decay expects
        df.rename(columns={
            'date': 'Date',
            'monthly_listeners': 'Monthly Listeners'
        }, inplace=True)
        
        # Ensure Date is datetime
        df['Date'] = pd.to_datetime(df['Date'])
        
        logger.info(f"Retrieved {len(df)} data points for artist {artist_id}")
        return df
    except Exception as e:
        logger.error(f"Error retrieving data for artist {artist_id}: {e}")
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
        # Run the analysis - this handles data cleaning, anomaly detection, and curve fitting
        results = analyze_listener_decay(artist_data)
        
        # Extract MLDR from results
        mldr = results['mldr']
        
        return mldr
    except Exception as e:
        logger.error(f"Error calculating MLDR: {e}")
        return None

def store_artist_mldr(artist_id: int, mldr: float):
    """
    Store the calculated MLDR in the database.
    
    Args:
        artist_id: The Chartmetric artist ID
        mldr: The calculated MLDR value
    """
    query = """
    INSERT INTO backtest_artist_mldr 
    (cm_artist_id, mldr, created_at, updated_at)
    VALUES (%s, %s, NOW(), NOW())
    ON CONFLICT (cm_artist_id) 
    DO UPDATE SET 
        mldr = EXCLUDED.mldr,
        updated_at = NOW()
    """
    
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(query, (artist_id, mldr))
        connection.commit()
        logger.info(f"Stored MLDR {mldr} for artist {artist_id}")
    except Exception as e:
        connection.rollback()
        logger.error(f"Error storing MLDR for artist {artist_id}: {e}")
    finally:
        connection.close()

def process_all_artists():
    """
    Main function to process all artists in the database.
    Gets artist data, calculates MLDR, and stores the results.
    """
    # Get list of all artists
    artists = get_all_artists()
    
    # Track statistics
    total_artists = len(artists)
    successful = 0
    failed = 0
    
    # Process each artist
    for i, artist_id in enumerate(artists):
        logger.info(f"Processing artist {artist_id} ({i+1}/{total_artists})")
        
        # Get artist data
        artist_data = get_artist_data(artist_id)
        if artist_data is None or len(artist_data) < 3:  # Need at least 3 data points for fitting
            logger.warning(f"Insufficient data for artist {artist_id}, skipping")
            failed += 1
            continue
            
        # Calculate MLDR
        mldr = calculate_artist_mldr(artist_data)
        if mldr is None:
            logger.warning(f"Failed to calculate MLDR for artist {artist_id}")
            failed += 1
            continue
            
        # Store MLDR in database
        store_artist_mldr(artist_id, mldr)
        successful += 1
        
    # Log summary
    logger.info(f"Processing complete: {successful} successful, {failed} failed, {total_artists} total")

if __name__ == "__main__":
    logger.info("Starting Artist MLDR calculation")
    process_all_artists()
    logger.info("Artist MLDR calculation complete")