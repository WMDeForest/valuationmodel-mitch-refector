"""
Track stream forecasting utilities for extracting metrics and building stream predictions.

This module provides functions for analyzing track streaming data and generating
forecasts of future streams based on historical patterns and decay models.
"""

import pandas as pd
import numpy as np
import streamlit as st
from scipy.optimize import curve_fit

from utils.data_processing import (
    extract_earliest_date,
    calculate_period_streams,
    calculate_months_since_release,
    calculate_monthly_stream_averages,
    extract_track_metrics
)
from utils.decay_rates import track_lifecycle_segment_boundaries

def piecewise_exp_decay(x, S0, k):
    """
    Core piecewise exponential decay function that models stream decay over time.
    
    This function implements the standard exponential decay formula S(t) = S0 * e^(-kt),
    where S0 is the initial value and k is the decay rate constant. The "piecewise" aspect 
    comes from using different k values for different time segments after release, which is 
    handled in the forecasting module.
    
    Args:
        x: Time variable (typically months since release)
        S0: Initial value (starting number of streams)
        k: Decay rate constant (higher values = faster decay)
        
    Returns:
        Decayed value at time x (predicted streams at month x)
    
    Example:
        >>> piecewise_exp_decay(3, 100000, 0.1)  # After 3 months with decay rate 0.1
        74081.82234034616  # Streams in month 3
    """
    return S0 * np.exp(-k * x)

def calculate_monthly_stream_projections(consolidated_df, initial_value, start_period, forecast_periods):
    """
    Generate month-by-month stream projections using segmented decay rates.
    
    This function is the heart of the valuation model's forecasting capability.
    It generates month-by-month stream predictions by:
    1. Starting with current stream level (initial_value)
    2. Applying appropriate decay rates for each future month
    3. Tracking which time segment applies to each forecast period
    
    The key insight is that music streaming typically follows different decay
    patterns at different stages of a track's lifecycle:
    - Very new tracks often have fast initial decay
    - Mid-life tracks settle into a moderate decay pattern
    - Catalog tracks (older) tend to have slower, more stable decay
    
    Args:
        consolidated_df: DataFrame with segment-specific decay rates
                         Contains 'segment' and 'k' columns
        initial_value: Starting value for forecasting (current month's streams)
        start_period: Starting time period (months since release)
        forecast_periods: Number of periods to forecast (how many months ahead)
        
    Returns:
        list: List of dictionaries with forecast details for each period:
             - 'month': Month number since release
             - 'forecasted_value': Predicted streams for that month
             - 'segment_used': Which time segment was used (1, 2, 3, etc.)
             - 'time_used': Forecasting period (1 = first forecast month, etc.)
    
    Notes:
        The track_lifecycle_segment_boundaries imported from utils.decay_rates define the boundaries
        between different time segments (e.g., 1-3 months, 4-12 months, etc.).
        
        For each forecast month, the function:
        1. Determines which segment applies based on months since release
        2. Uses the decay rate (k) for that segment
        3. Applies exponential decay for one month
        4. Uses each month's output as the input for the next month
        
        This creates a chain of forecasts that can extend years into the future.
    """
    # Convert DataFrame to list of dictionaries for easier iteration
    params = consolidated_df.to_dict(orient='records')
    forecasts = []
    current_value = initial_value
    
    # Generate forecasts for each period
    for i in range(forecast_periods):
        current_segment = 0
        current_month = start_period + i
        
        # Determine which segment applies to the current month
        # This complex calculation finds which segment the current month belongs to
        # based on the segment boundaries defined in the decay_rates module
        while current_month >= sum(len(range(track_lifecycle_segment_boundaries[j] + 1, track_lifecycle_segment_boundaries[j + 1] + 1)) 
                                 for j in range(current_segment + 1)):
            current_segment += 1
            
            # Safety check to avoid index errors if we exceed defined segments
            if current_segment >= len(params):
                current_segment = len(params) - 1
                break
        
        # Get the parameters for the current segment
        current_segment_params = params[current_segment]
        S0 = current_value  # Starting value for this month is previous month's result
        k = current_segment_params['k']  # Decay rate for this segment
        
        # Calculate the forecast for one month using exponential decay
        # Note: The time period is always 1 because we're forecasting just one month at a time
        forecast_value = S0 * np.exp(-k * (1))
        
        # Store the forecast with metadata
        forecasts.append({
            'month': current_month,
            'predicted_streams_for_month': forecast_value,
            'segment_used': current_segment + 1,  # +1 for human-readable segment numbering
            'time_used': current_month - start_period + 1
        })
        
        # Update current value for next iteration
        current_value = forecast_value
    
    return forecasts

def generate_track_decay_rates_by_month(decay_rates_df, segment_boundaries, forecast_horizon=500):
    """
    Generate a list of track-specific decay rates for each month in the forecast horizon.
    
    This function creates a comprehensive mapping of decay rates for each month
    in the forecast period, based on the segmentation defined by segment_boundaries.
    Different segments of a track's lifespan may have different decay rate patterns.
    
    Args:
        decay_rates_df: DataFrame containing decay rates by segment
                       Must have 'k' column with decay rate values
        segment_boundaries: List of month numbers that define the segments
                   For example, [1, 6, 12, 24, 60, 500] defines 5 segments
        forecast_horizon: Maximum number of months to generate rates for
                         Default is 500 months (approx. 41.7 years)
    
    Returns:
        list: List of track decay rates, with one value per month up to forecast_horizon
    
    Notes:
        - The function finds which segment each month belongs to based on segment_boundaries
        - It then assigns the appropriate decay rate from decay_rates_df to that month
        - This creates a month-by-month mapping of decay rates for the entire forecast period
    """
    # Create a list for all months in the forecast horizon
    all_forecast_months = list(range(1, forecast_horizon + 1))
    monthly_decay_rates = []
    
    # For each month, determine its segment and assign the appropriate decay rate
    for month in all_forecast_months:
        for i in range(len(segment_boundaries) - 1):
            # Check if the month falls within this segment's range
            if segment_boundaries[i] <= month < segment_boundaries[i + 1]:
                # Assign the decay rate from the corresponding segment
                segment_decay_rate = decay_rates_df.loc[i, 'k']
                monthly_decay_rates.append(segment_decay_rate)
                break
                
    return monthly_decay_rates

def create_decay_rate_dataframe(track_months_since_release, track_monthly_decay_rates, mldr=None, 
                             track_data_start_month=None, track_data_end_month=None):
    """
    Create a DataFrame with track months since release and corresponding decay rates.
    
    This function creates a structured DataFrame for all decay rate data,
    including both model-derived rates and observed rates from actual data.
    The DataFrame provides a foundation for further adjustments and analysis.
    
    Args:
        track_months_since_release: List of integers representing months since track release
                                  Used for forecasting over the track's lifetime
        track_monthly_decay_rates: List of track-specific decay rates corresponding to each month
        mldr: Monthly Listener Decay Rate observed from artist listener data analysis
              Used to incorporate artist-level decay patterns into track projections
        track_data_start_month: Start month of track's actual streaming data observation period
        track_data_end_month: End month of track's actual streaming data observation period
    
    Returns:
        pandas.DataFrame: DataFrame with months and corresponding decay rates
                         Includes columns for both model and observed rates
    
    Notes:
        - When MLDR data is provided, the function adds the observed decay rate
          to the months that fall within the observation period
        - This allows comparison between model-predicted decay and actual decay
    """
    # Create the basic DataFrame with months and model-derived decay rates
    decay_df = pd.DataFrame({
        'months_since_release': track_months_since_release,
        'decay_rate': track_monthly_decay_rates
    })
    
    # If MLDR is provided, add it to the appropriate months
    if mldr is not None and track_data_start_month is not None and track_data_end_month is not None:
        # Add column for MLDR (Monthly Listener Decay Rate)
        decay_df['mldr'] = None
        
        # Apply MLDR to the period where we have actual track streaming data
        decay_df.loc[(decay_df['months_since_release'] >= track_data_start_month) & 
                    (decay_df['months_since_release'] <= track_data_end_month), 'mldr'] = mldr
    
    return decay_df

def adjust_track_decay_rates(track_decay_df, track_decay_k=None):
    """
    Adjust theoretical track decay rates using observed data for more accurate forecasting.
    
    This function performs a two-step adjustment process:
    1. First adjustment: Analyzes difference between theoretical and artist-observed decay rates (MLDR)
       and applies a weighted adjustment based on the direction of difference
    2. Second adjustment: Compares the first adjustment with fitted decay rates from actual
       track streaming data and applies another weighted adjustment
    
    The weighting approach helps balance theoretical models with real-world behavior.
    
    Args:
        track_decay_df: DataFrame with track months and corresponding decay rates
                      Must contain 'decay_rate' column and optionally 'mldr' column
        track_decay_k: Decay rate parameter from exponential curve fitting of track data
                        Only used for second-level adjustment if provided
    
    Returns:
        tuple: (adjusted_df, adjustment_info) where:
               - adjusted_df is the DataFrame with adjusted track decay rates
               - adjustment_info is a dictionary with metrics about the adjustments made
    
    Notes:
        - Positive differences (observed > theoretical) lead to adjustments
          that slow the decay, extending lifetime value
        - The function applies weighting to prevent extreme adjustments
        - Clean-up is performed to remove intermediate calculation columns
    """
    # Deep copy to avoid modifying the original DataFrame
    adjusted_df = track_decay_df.copy()
    
    # First adjustment: Compare observed vs. theoretical decay rates
    if 'mldr' in adjusted_df.columns:
        # Calculate percentage difference between observed and theoretical decay
        adjusted_df['percent_change'] = ((adjusted_df['mldr'] - adjusted_df['decay_rate']) / 
                                         adjusted_df['decay_rate']) * 100
        
        # Calculate the average percentage change across observed months
        average_percent_change = adjusted_df['percent_change'].dropna().mean()
        
        # Apply weighting based on direction of change
        # Positive changes (slower decay) are given more weight than negative changes
        if average_percent_change > 0:
            adjustment_weight = min(1, max(0, average_percent_change / 100))
        else:
            adjustment_weight = 0
        
        # Apply first level of adjustment to decay rates
        adjusted_df['adjusted_decay_rate'] = (adjusted_df['decay_rate'] * 
                                             (1 + (average_percent_change * adjustment_weight) / 100))
    else:
        # If no observed data, keep original decay rates
        adjusted_df['adjusted_decay_rate'] = adjusted_df['decay_rate']
        adjustment_weight = 0
        average_percent_change = 0
    
    # Second adjustment: If fitted parameter is available, apply another adjustment
    if track_decay_k is not None and 'mldr' in adjusted_df.columns:
        # Get indices of months with observed data
        observed_months_mask = ~adjusted_df['mldr'].isna()
        
        # Add fitted decay rate to observed period
        adjusted_df.loc[observed_months_mask, 'new_decay_rate'] = track_decay_k
        
        # Compare adjusted decay rate with newly fitted rate
        adjusted_df['percent_change_new_vs_adjusted'] = ((adjusted_df['new_decay_rate'] - 
                                                          adjusted_df['adjusted_decay_rate']) / 
                                                         adjusted_df['adjusted_decay_rate']) * 100
        
        # Calculate average change for second adjustment
        average_percent_change_new_vs_adjusted = adjusted_df['percent_change_new_vs_adjusted'].dropna().mean()
        
        # Only apply additional weight for positive changes (slower decay)
        second_adjustment_weight = 1 if average_percent_change_new_vs_adjusted > 0 else 0
        
        # Apply final adjustment
        adjusted_df['final_adjusted_decay_rate'] = (adjusted_df['adjusted_decay_rate'] * 
                                                   (1 + (average_percent_change_new_vs_adjusted * 
                                                        second_adjustment_weight) / 100))
    else:
        # If no fitted parameter, use the first adjustment
        adjusted_df['final_adjusted_decay_rate'] = adjusted_df['adjusted_decay_rate']
    
    # Store adjustment weights and changes for reference (outside the function)
    adjustment_info = {
        'first_adjustment_weight': adjustment_weight,
        'first_average_percent_change': average_percent_change,
        'second_adjustment_weight': second_adjustment_weight if track_decay_k is not None else 0,
        'second_average_percent_change': average_percent_change_new_vs_adjusted if track_decay_k is not None else 0
    }
    
    # Clean up intermediate calculation columns
    columns_to_drop = ['decay_rate', 'percent_change', 'adjusted_decay_rate']
    if 'mldr' in adjusted_df.columns:
        columns_to_drop.append('mldr')
    if 'new_decay_rate' in adjusted_df.columns:
        columns_to_drop.append('new_decay_rate')
        columns_to_drop.append('percent_change_new_vs_adjusted')
    
    adjusted_df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    
    return adjusted_df, adjustment_info

def calculate_track_decay_rates_by_segment(adjusted_df, segment_boundaries):
    """
    Calculate average decay rates for each segment defined by segment_boundaries.
    
    This function divides the forecast period into segments based on the provided
    segment_boundaries, and calculates the average decay rate within each segment.
    This allows for simplified modeling while respecting the different decay
    behaviors at different stages of a track's lifecycle.
    
    Args:
        adjusted_df: DataFrame with months and adjusted decay rates
                    Must contain 'months_since_release' and 'final_adjusted_decay_rate' columns
        segment_boundaries: List of month numbers that define segment boundaries
                    For example, [1, 6, 12, 24, 60, 500] defines 5 segments
    
    Returns:
        pandas.DataFrame: DataFrame with segment numbers and corresponding average decay rates
                         Contains 'segment' and 'k' columns
    
    Notes:
        - Each segment spans from one boundary (inclusive) to the next (exclusive)
        - The function calculates the average decay rate within each segment
        - This consolidated representation is used for efficient forecasting
    """
    segments = []
    avg_decay_rates = []

    # Process each segment defined by the boundaries
    for i in range(len(segment_boundaries) - 1):
        # Define segment boundaries
        start_month = segment_boundaries[i]
        end_month = segment_boundaries[i + 1] - 1
        
        # Extract data for this segment
        segment_data = adjusted_df[(adjusted_df['months_since_release'] >= start_month) & 
                                  (adjusted_df['months_since_release'] <= end_month)]
        
        # Calculate average decay rate for the segment
        avg_decay_rate = segment_data['final_adjusted_decay_rate'].mean()
        
        # Store segment number (1-indexed) and its average decay rate
        segments.append(i + 1)
        avg_decay_rates.append(avg_decay_rate)

    # Create consolidated DataFrame with segment numbers and decay rates
    segmented_track_decay_rates_df = pd.DataFrame({
        'segment': segments,
        'k': avg_decay_rates
    })
    
    return segmented_track_decay_rates_df 

def prepare_decay_rate_fitting_data(months_since_release, avg_monthly_streams_months_4to12, avg_monthly_streams_months_2to3, streams_last_30days):
    """
    Prepare data arrays for decay rate fitting model.
    
    Parameters:
    -----------
    months_since_release : int
        Number of months since the track was released
    avg_monthly_streams_months_4to12 : float
        Average monthly streams for months 4-12
    avg_monthly_streams_months_2to3 : float
        Average monthly streams for months 2-3
    streams_last_30days : float
        Total streams in the last month (last 30 days)
        
    Returns:
    --------
    tuple:
        (months_array, averages_array)
        NumPy arrays containing the months since release and corresponding average stream values
    """
    # ===== CREATE HISTORICAL DATA POINTS FOR DECAY CURVE FITTING =====
    # This creates three data points at different points in the track's history:
    # 1. A point representing month 12 (or earliest available if track is newer)
    # 2. A point representing month 3 (or earliest available if track is newer)
    # 3. A point representing the current month
    # These three points will be used to fit an exponential decay curve
    months_array = np.array([
        max((months_since_release - 11), 0),  # 12 months ago (or 0 if track is newer)
        max((months_since_release - 2), 0),   # 3 months ago (or 0 if track is newer)
        months_since_release - 0              # Current month
    ])
    
    # ===== CREATE CORRESPONDING STREAM VALUES =====
    # For each month in the months_array, we provide the corresponding stream value:
    # 1. The avg monthly streams from months 4-12 for the first point
    # 2. The avg monthly streams from months 2-3 for the second point 
    # 3. The actual streams from the last 30 days for the current month
    # This gives us a time series of points showing how stream volume has changed over time
    averages_array = np.array([avg_monthly_streams_months_4to12, avg_monthly_streams_months_2to3, streams_last_30days])
    
    return months_array, averages_array

def fit_segment(months_since_release, streams):
    """
    Fit exponential decay model to a segment of streaming data.
    
    This function uses SciPy's curve_fit to find the optimal parameters (S0, k)
    that minimize the difference between the actual stream data and the predicted
    values from the exponential decay function.
    
    The optimization is constrained to ensure both S0 (initial streams) and k (decay rate)
    are positive values, which makes physical sense for music streaming patterns.
    
    Args:
        months_since_release: Array or list of months since release (time values)
        streams: Array or list of stream counts corresponding to each month
        
    Returns:
        tuple: (S0, k) parameters for the fitted model where:
               - S0 is the initial streams value
               - k is the decay rate (higher = faster decay)
    
    Notes:
        This function is typically used for fitting decay patterns in specific time segments
        (e.g., months 1-3, 4-12, 13-36, etc.) as decay rates often change over a track's lifecycle.
    """
    # Convert inputs to numpy arrays if they're lists
    if isinstance(months_since_release, list):
        months_since_release = np.array(months_since_release)
    if isinstance(streams, list):
        streams = np.array(streams)
        
    # Set initial parameter guess based on first observed stream count and typical decay rate
    initial_guess = [streams[0], 0.01]  
    
    # Set bounds to ensure physically meaningful parameters (positive values only)
    bounds = ([0, 0], [np.inf, np.inf])  
    
    # Perform curve fitting to find optimal parameters
    params, covariance = curve_fit(piecewise_exp_decay, months_since_release, streams, 
                                   p0=initial_guess, bounds=bounds)
    
    return params

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

def build_complete_track_forecast(
    track_metrics,
    mldr,
    fitted_params_df,
    stream_influence_factor,
    sp_range,
    sp_reach,
    track_lifecycle_segment_boundaries,
    forecast_periods):
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