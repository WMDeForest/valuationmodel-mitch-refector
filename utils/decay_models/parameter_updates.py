"""
Functions for updating decay model parameters based on external factors.

This module contains functions that modify decay rate parameters based on 
external influence factors, such as streaming influence factors (formerly called playlist reach),
which can significantly affect the long-term decay patterns of music streams.
"""
import streamlit as st

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
    
    Notes:
        The adjustment process:
        1. Determine the appropriate segment based on the influence factor value
        2. Look up the corresponding adjustment column in SP_REACH
        3. Apply a weighted average (67% original, 33% adjustment)
        
        A higher influence factor typically results in a lower decay rate,
        extending the track's commercial lifetime in the forecast.
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
    
    Usage:
        # Get both formats:
        df, dict_format = get_decay_parameters(...)
        
        # Get only DataFrame:
        df, _ = get_decay_parameters(...)
        
        # Get only dictionary:
        _, dict_format = get_decay_parameters(...)
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
    import pandas as pd
    
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
    import pandas as pd
    
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