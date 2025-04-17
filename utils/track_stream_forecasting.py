"""
Track stream forecasting utilities for extracting metrics and building stream predictions.

This module provides functions for analyzing track streaming data and generating
forecasts of future streams based on historical patterns and decay models.
"""

import pandas as pd
import numpy as np
import streamlit as st

from utils.data_processing import (
    extract_earliest_date,
    calculate_period_streams,
    calculate_months_since_release,
    calculate_monthly_stream_averages
)
from utils.decay_models.fitting import prepare_decay_rate_fitting_data
from utils.decay_models import (
    fit_segment,
    calculate_monthly_stream_projections
)
from utils.decay_models.parameter_updates import (
    generate_track_decay_rates_by_month,
    create_decay_rate_dataframe,
    adjust_track_decay_rates,
    calculate_track_decay_rates_by_segment
)

def update_fitted_params(fitted_params_df, value, sp_range, SP_REACH):
    """
    Update fitted decay parameters based on stream influence factor.
    
    This function adjusts the base decay rates based on a track's stream influence factor
    (formerly called Spotify playlist reach). Tracks with higher influence factors typically 
    have slower decay rates (more sustainability) as they continue to receive streams over time.
    
    The adjustment uses a weighted average approach:
    - 67% weight on the original decay rates
    - 33% weight on the adjustment factors
    
    This balances the intrinsic track decay patterns with the boost from external factors.
    
    Args:
        fitted_params_df: DataFrame with initial fitted parameters
                         Contains segment numbers and corresponding 'k' values
        value: Stream influence factor (numeric measure of external streaming influence)
               [Formerly called Spotify playlist reach]
        sp_range: DataFrame with ranges for segmentation
                 Maps influence values to appropriate adjustment columns
        SP_REACH: DataFrame with adjustment factors for different influence levels
                 Contains modifier columns for different segments
        
    Returns:
        DataFrame: Updated fitted parameters with adjusted 'k' values
                  Returns None if the segment is not found in SP_REACH
    """
    updated_fitted_params_df = fitted_params_df.copy()
    
    # Find the appropriate segment based on the influence factor value
    # (formerly called playlist reach value)
    segment = sp_range.loc[(sp_range['RangeStart'] <= value) & 
                          (sp_range['RangeEnd'] > value), 'Column 1'].iloc[0]
    
    # Validate the segment exists in SP_REACH
    if segment not in SP_REACH.columns:
        st.error(f"Error: Column '{segment}' not found in SP_REACH.")
        st.write("Available columns:", SP_REACH.columns)
        return None
    
    # Get the adjustment column and update the decay rate
    # Uses a weighted average: 67% original value, 33% influence-based adjustment
    # (formerly called playlist-based adjustment)
    column_to_append = SP_REACH[segment]
    updated_fitted_params_df['k'] = updated_fitted_params_df['k'] * 0.67 + column_to_append * 0.33

    return updated_fitted_params_df 

def get_decay_parameters(fitted_params_df, stream_influence_factor, sp_range, sp_reach):
    """
    Get decay parameters in both DataFrame and dictionary formats.
    
    This is a comprehensive function that returns both the updated parameters DataFrame
    and the dictionary (records) representation in a single call. It can be used with
    data from any source (API, CSV, database) as long as the fitted_params_df is provided
    in the expected format.
    
    Args:
        fitted_params_df: DataFrame with initial fitted parameters
        stream_influence_factor: Numeric measure of external streaming influence
        sp_range: DataFrame with ranges for segmentation
        sp_reach: DataFrame with adjustment factors for different influence levels
        
    Returns:
        tuple: (updated_params_df, updated_params_dict) where:
               - updated_params_df is the DataFrame with updated parameters
               - updated_params_dict is the list of dictionaries (records format)
               
               Returns (None, None) if parameters could not be updated
    """
    updated_params_df = update_fitted_params(
        fitted_params_df, 
        stream_influence_factor, 
        sp_range, 
        sp_reach
    )
    
    if updated_params_df is not None:
        updated_params_dict = updated_params_df.to_dict(orient='records')
        return updated_params_df, updated_params_dict
    
    return None, None

def extract_track_metrics(track_data_df, track_name=None):
    """
    Extract basic track metrics from streaming data.
    
    Parameters:
    -----------
    track_data_df : pandas.DataFrame
        DataFrame containing the track's streaming data with 'Date' and 'CumulativeStreams' columns
    track_name : str, optional
        Name of the track
        
    Returns:
    --------
    dict
        Dictionary containing basic track metrics
    """
    # Extract base metrics from the track data
    earliest_track_date = extract_earliest_date(track_data_df, 'Date')
    total_historical_track_streams = track_data_df['CumulativeStreams'].iloc[-1]
    
    # Calculate period-specific stream counts
    track_streams_last_30days = calculate_period_streams(track_data_df, 'CumulativeStreams', 30)
    track_streams_last_90days = calculate_period_streams(track_data_df, 'CumulativeStreams', 90)
    track_streams_last_365days = calculate_period_streams(track_data_df, 'CumulativeStreams', 365)
    
    # Calculate time-based metrics
    months_since_release_total = calculate_months_since_release(earliest_track_date)
    
    # Calculate monthly averages for different time periods
    avg_monthly_streams_months_4to12, avg_monthly_streams_months_2to3 = calculate_monthly_stream_averages(
        track_streams_last_30days,
        track_streams_last_90days,
        track_streams_last_365days,
        months_since_release_total
    )
    
    # Prepare arrays for decay rate fitting
    months_since_release, monthly_averages = prepare_decay_rate_fitting_data(
        months_since_release_total,
        avg_monthly_streams_months_4to12,
        avg_monthly_streams_months_2to3,
        track_streams_last_30days
    )
    
    # Return a dictionary with all calculated metrics
    metrics = {
        'track_name': track_name,
        'earliest_track_date': earliest_track_date,
        'total_historical_track_streams': total_historical_track_streams,
        'track_streams_last_30days': track_streams_last_30days,
        'track_streams_last_90days': track_streams_last_90days,
        'track_streams_last_365days': track_streams_last_365days,
        'months_since_release_total': months_since_release_total,
        'avg_monthly_streams_months_4to12': avg_monthly_streams_months_4to12,
        'avg_monthly_streams_months_2to3': avg_monthly_streams_months_2to3,
        'months_since_release': months_since_release,
        'monthly_averages': monthly_averages
    }
    
    return metrics

def build_complete_track_forecast(
    track_metrics,
    mldr,
    fitted_params_df,
    stream_influence_factor,
    sp_range,
    sp_reach,
    track_lifecycle_segment_boundaries,
    forecast_periods
):
    """
    Build a complete track forecast by coordinating the end-to-end forecasting process.
    
    This high-level function orchestrates the entire forecasting pipeline:
    1. Prepares decay parameters and rates
    2. Fits the decay model to track data
    3. Adjusts rates based on observed patterns
    4. Generates the final stream projections
    
    Parameters:
    -----------
    track_metrics : dict
        Dictionary of track metrics from extract_track_metrics function
    mldr : float
        Monthly Listener Decay Rate from artist-level analysis
    fitted_params_df : pandas.DataFrame
        DataFrame containing pre-fitted decay parameters
    stream_influence_factor : float
        Factor to adjust stream projections
    sp_range : list
        Range list for stream projections
    sp_reach : list
        Reach data for stream projections
    track_lifecycle_segment_boundaries : dict
        Dictionary defining track lifecycle segment boundaries
    forecast_periods : int
        Number of periods to forecast
        
    Returns:
    --------
    dict
        Dictionary containing forecast data and parameters
    """
    # Extract needed metrics
    months_since_release = track_metrics['months_since_release']
    monthly_averages = track_metrics['monthly_averages']
    track_streams_last_30days = track_metrics['track_streams_last_30days']
    months_since_release_total = track_metrics['months_since_release_total']
    
    # 1. Get decay parameters
    decay_rates_df, updated_fitted_params = get_decay_parameters(
        fitted_params_df, 
        stream_influence_factor, 
        sp_range, 
        sp_reach
    )
    
    # 2. Fit decay model to stream data
    params = fit_segment(months_since_release, monthly_averages)
    S0, track_decay_k = params
    
    # 3. Generate track-specific decay rates for forecast
    track_monthly_decay_rates = generate_track_decay_rates_by_month(
        decay_rates_df, 
        track_lifecycle_segment_boundaries
    )
    
    # 4. Determine observed time range
    track_data_start_month = min(months_since_release)
    track_data_end_month = max(months_since_release)
    
    # 5. Create structured decay rate dataframe
    track_decay_rate_df = create_decay_rate_dataframe(
        track_months_since_release=list(range(1, 501)),  # Forecast for 500 months
        track_monthly_decay_rates=track_monthly_decay_rates,
        mldr=mldr,
        track_data_start_month=track_data_start_month,
        track_data_end_month=track_data_end_month
    )
    
    # 6. Adjust decay rates based on observed data
    adjusted_track_decay_df, track_adjustment_info = adjust_track_decay_rates(
        track_decay_rate_df, 
        track_decay_k=track_decay_k
    )
    
    # 7. Segment decay rates by time period
    segmented_track_decay_rates_df = calculate_track_decay_rates_by_segment(
        adjusted_track_decay_df, 
        track_lifecycle_segment_boundaries
    )
    
    # 8. Generate stream forecasts
    track_streams_forecast = calculate_monthly_stream_projections(
        segmented_track_decay_rates_df, 
        track_streams_last_30days, 
        months_since_release_total, 
        forecast_periods
    )
    
    # 9. Convert forecasts to DataFrame
    track_streams_forecast_df = pd.DataFrame(track_streams_forecast)
    
    # Prepare forecast results
    forecast_result = {
        'track_decay_k': track_decay_k,
        'forecast_df': track_streams_forecast_df,
        'decay_parameters': {
            'S0': S0,
            'track_decay_k': track_decay_k,
            'track_adjustment_info': track_adjustment_info
        }
    }
    
    return forecast_result 