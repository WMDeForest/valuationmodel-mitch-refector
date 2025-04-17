"""
Parameter update utilities for adjusting decay curve parameters based on
external influence factors, such as streaming influence factors (formerly called playlist reach),
which can significantly affect the long-term decay patterns of music streams.
"""
import streamlit as st
import pandas as pd

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