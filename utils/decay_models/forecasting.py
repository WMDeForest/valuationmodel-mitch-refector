"""
Forecasting functions for predicting future streaming values.

This module contains functions that generate stream forecasts based on 
fitted decay parameters. It handles the complex logic of applying different
decay rates to different time segments in a track's lifetime.
"""
import numpy as np
from utils.decay_rates import track_lifecycle_segment_boundaries

def forecast_values(consolidated_df, initial_value, start_period, forecast_periods):
    """
    Generate forecasts for future streaming values using segmented decay rates.
    
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
            'forecasted_value': forecast_value,
            'segment_used': current_segment + 1,  # +1 for human-readable segment numbering
            'time_used': current_month - start_period + 1
        })
        
        # Update current value for next iteration
        current_value = forecast_value
    
    return forecasts 