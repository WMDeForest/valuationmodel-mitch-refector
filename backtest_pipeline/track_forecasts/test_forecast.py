#!/usr/bin/env python
# coding: utf-8

"""
Test Script for Track Streams Forecasting

This script tests the track forecasting functionality on a single track ID.
It's useful for debugging and validating the forecasting process.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import atexit

# Add project root to path to ensure imports work correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import database utilities
from utils.database import close_all_connections, query_to_dataframe

# Import the forecast module
from forecast_streams import generate_track_forecasts, get_artist_mldr, get_training_data

# Always register the cleanup function to ensure proper shutdown
atexit.register(close_all_connections)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_forecast')

def test_single_track(cm_track_id, cm_artist_id):
    """
    Test forecasting for a single track.
    
    Args:
        cm_track_id: ChartMetric track ID
        cm_artist_id: ChartMetric artist ID
    """
    # Convert numpy types to native Python types
    if isinstance(cm_track_id, np.integer):
        cm_track_id = int(cm_track_id)
    if isinstance(cm_artist_id, np.integer):
        cm_artist_id = int(cm_artist_id)
        
    logger.info(f"Testing forecast for track ID {cm_track_id}, artist ID {cm_artist_id}")
    
    # Get artist MLDR
    mldr = get_artist_mldr(cm_artist_id)
    if mldr:
        logger.info(f"Artist MLDR: {mldr}")
    else:
        logger.error("Could not retrieve artist MLDR")
        return
    
    # Get training data metrics
    training_data_id, track_streams_last_30days, months_since_release = get_training_data(cm_track_id, cm_artist_id)
    if not training_data_id:
        logger.error("Could not retrieve training data")
        return
    
    logger.info(f"Training data ID: {training_data_id}")
    logger.info(f"Track streams (last 30 days): {track_streams_last_30days}")
    logger.info(f"Months since release: {months_since_release}")
    
    # Generate forecasts
    forecasts = generate_track_forecasts(cm_track_id, cm_artist_id, mldr)
    if not forecasts:
        logger.error("Failed to generate forecasts")
        return
    
    # Convert forecasts to DataFrame for easier inspection
    forecast_df = pd.DataFrame(forecasts)
    
    # Summary statistics
    logger.info(f"Generated {len(forecast_df)} forecast periods")
    logger.info(f"Total predicted streams: {forecast_df['predicted_streams_for_month'].sum()}")
    
    # Show first few forecast periods
    logger.info("First 10 forecast periods:")
    for i, forecast in forecast_df.head(10).iterrows():
        logger.info(f"Month {forecast['month']}: {forecast['predicted_streams_for_month']:.2f} streams (Segment {forecast['segment_used']})")
    
    # Save the forecast to a CSV file for inspection
    output_file = f"track_{cm_track_id}_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    forecast_df.to_csv(output_file, index=False)
    logger.info(f"Saved forecast to {output_file}")

def find_track_to_test():
    """
    Find a suitable track for testing that has both training data and MLDR.
    
    Returns:
        tuple: (cm_track_id, cm_artist_id) or (None, None) if not found
    """
    query = """
    SELECT t.cm_track_id, t.cm_artist_id
    FROM backtest_track_daily_training_data t
    JOIN backtest_artist_mldr m ON t.cm_artist_id = m.cm_artist_id
    WHERE t.cm_track_id NOT IN (
        SELECT DISTINCT cm_track_id FROM backtest_track_streams_forecast
    )
    GROUP BY t.cm_track_id, t.cm_artist_id
    LIMIT 1
    """
    
    result = query_to_dataframe(query)
    
    if result.empty:
        return None, None
    
    # Convert numpy types to Python native types
    cm_track_id = int(result.iloc[0]['cm_track_id']) if isinstance(result.iloc[0]['cm_track_id'], np.integer) else result.iloc[0]['cm_track_id']
    cm_artist_id = int(result.iloc[0]['cm_artist_id']) if isinstance(result.iloc[0]['cm_artist_id'], np.integer) else result.iloc[0]['cm_artist_id']
    
    return cm_track_id, cm_artist_id

def main():
    """
    Main function to run the test.
    """
    # Find a track to test
    cm_track_id, cm_artist_id = find_track_to_test()
    
    if not cm_track_id:
        logger.error("Could not find a suitable track for testing")
        return
    
    # Test the track
    test_single_track(cm_track_id, cm_artist_id)

if __name__ == "__main__":
    main() 