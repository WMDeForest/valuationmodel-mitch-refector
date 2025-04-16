#!/usr/bin/env python3
# coding: utf-8

"""
Simple Database Connection Test

This script performs a minimal test of the database utilities:
1. Sets up the SSH tunnel
2. Connects to the database
3. Runs a simple SELECT query
4. Prints the results
5. Closes the connection
"""

import logging
import atexit

# Import only what we need from database utilities
from utils.database import setup_ssh_tunnel, fetch_all, close_all_connections

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('db_test')

def main():
    """Run a simple database connection test."""
    logger.info("Starting simple database connection test")
    
    # Register cleanup
    atexit.register(close_all_connections)
    
    try:
        # 1. Set up the SSH tunnel
        logger.info("Setting up SSH tunnel...")
        setup_ssh_tunnel()
        logger.info("SSH tunnel established successfully")
        
        # 2. Run a simple query
        logger.info("Running test query...")
        query = "SELECT COUNT(*) FROM backtest_artist_daily_training_data"
        result = fetch_all(query)
        
        # 3. Print the result
        count = result[0][0] if result else 0
        logger.info(f"Query result: {count} rows in backtest_artist_daily_training_data")
        
        logger.info("Test completed successfully!")
        return True
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False
    finally:
        # 4. Clean up (also handled by atexit)
        close_all_connections()

if __name__ == "__main__":
    main() 