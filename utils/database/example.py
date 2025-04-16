#!/usr/bin/env python3
# coding: utf-8

"""
Database Utilities Example Script

This script demonstrates how to use the database utility functions
for common database operations.

Example usage:
PYTHONPATH=/path/to/project python3 utils/database/example.py
"""

import os
import pandas as pd
import numpy as np
import atexit
import logging
from datetime import datetime

# Import database utilities
from utils.database import (
    setup_ssh_tunnel,
    get_db_connection,
    release_db_connection,
    get_sqlalchemy_engine,
    close_all_connections,
    fetch_all,
    fetch_one,
    execute_query,
    execute_batch,
    query_to_dataframe,
    dataframe_to_table
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('database_example')

def example_basic_query():
    """Example of a basic query."""
    logger.info("Running basic query example")
    
    # Simple query to get some data
    query = "SELECT * FROM backtest_artist_daily_training_data LIMIT 10"
    
    try:
        # Fetch results
        results = fetch_all(query)
        logger.info(f"Query returned {len(results)} rows")
        
        # Display the first row
        if results:
            logger.info(f"First row: {results[0]}")
    except Exception as e:
        logger.error(f"Error in basic query: {e}")

def example_parameterized_query():
    """Example of a parameterized query."""
    logger.info("Running parameterized query example")
    
    # Parameterized query
    query = """
    SELECT * FROM backtest_artist_daily_training_data 
    WHERE cm_artist_id = %s
    LIMIT %s
    """
    
    try:
        # Define parameters
        artist_id = 3604555  # Example artist ID
        limit = 5
        
        # Fetch results with parameters
        results = fetch_all(query, params=(artist_id, limit))
        logger.info(f"Query returned {len(results)} rows for artist {artist_id}")
        
        # Display all rows
        for i, row in enumerate(results):
            logger.info(f"Row {i+1}: {row}")
    except Exception as e:
        logger.error(f"Error in parameterized query: {e}")

def example_pandas_dataframe():
    """Example of using pandas DataFrames."""
    logger.info("Running pandas DataFrame example")
    
    # Query to get data for pandas
    query = """
    SELECT cm_artist_id, date, monthly_listeners 
    FROM backtest_artist_daily_training_data
    WHERE cm_artist_id = 3604555
    ORDER BY date
    LIMIT 100
    """
    
    try:
        # Get data as a DataFrame
        df = query_to_dataframe(query)
        logger.info(f"DataFrame has {len(df)} rows and {len(df.columns)} columns")
        
        # Show DataFrame info
        logger.info(f"DataFrame columns: {df.columns.tolist()}")
        logger.info(f"DataFrame dtypes: {df.dtypes}")
        
        # Example data processing
        if not df.empty:
            # Convert date to datetime if needed
            if df['date'].dtype != 'datetime64[ns]':
                df['date'] = pd.to_datetime(df['date'])
            
            # Calculate daily percentage change
            df = df.sort_values('date')
            df['prev_listeners'] = df['monthly_listeners'].shift(1)
            df['pct_change'] = (df['monthly_listeners'] - df['prev_listeners']) / df['prev_listeners'] * 100
            
            # Show some stats
            logger.info(f"Mean monthly listeners: {df['monthly_listeners'].mean():.2f}")
            logger.info(f"Mean daily change: {df['pct_change'].mean():.2f}%")
            
            # Example of writing back to the database (commented out for safety)
            """
            # Create a temporary table name
            temp_table = f"temp_example_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            # Write DataFrame to table
            rows_written = dataframe_to_table(df, temp_table, if_exists='replace')
            logger.info(f"Wrote {rows_written} rows to {temp_table}")
            
            # Clean up - drop the temporary table
            execute_query(f"DROP TABLE {temp_table}", commit=True)
            logger.info(f"Dropped temporary table {temp_table}")
            """
    except Exception as e:
        logger.error(f"Error in pandas DataFrame example: {e}")

def example_batch_operation():
    """Example of a batch database operation."""
    logger.info("Running batch operation example")
    
    # This example shows how to do a batch insert/update
    # We'll just prepare the data but not execute for safety
    
    # Prepare sample data
    data = [
        (1, 'Artist 1', 50000),
        (2, 'Artist 2', 75000),
        (3, 'Artist 3', 120000)
    ]
    
    # Show how to format the query and data
    query = """
    INSERT INTO some_table (id, name, monthly_listeners) 
    VALUES %s
    """
    
    logger.info(f"Batch operation would insert {len(data)} rows")
    logger.info(f"Sample data: {data[0]}")
    logger.info(f"Query template: {query}")
    
    # In a real scenario, you would execute:
    # rows_affected = execute_batch(query, data)
    # logger.info(f"Inserted {rows_affected} rows")

def main():
    """Main function to run examples."""
    logger.info("Starting database utilities example")
    
    # Register cleanup with atexit
    atexit.register(close_all_connections)
    
    try:
        # Initialize SSH tunnel and connection pool
        setup_ssh_tunnel()
        
        # Run examples
        example_basic_query()
        example_parameterized_query()
        example_pandas_dataframe()
        example_batch_operation()
        
        logger.info("Examples completed successfully")
    except Exception as e:
        logger.error(f"Error running examples: {e}")
    finally:
        # Clean up connections (also called by atexit)
        close_all_connections()

if __name__ == "__main__":
    main() 