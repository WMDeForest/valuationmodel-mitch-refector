"""
Forecasting functions for predicting future streaming values.
"""
import numpy as np
from utils.decay_rates import breakpoints

def forecast_values(consolidated_df, initial_value, start_period, forecast_periods):
    """
    Generate forecasts for future streaming values.
    
    Args:
        consolidated_df: DataFrame with segment-specific decay rates
        initial_value: Starting value for forecasting
        start_period: Starting time period (e.g., current month since release)
        forecast_periods: Number of periods to forecast
        
    Returns:
        list: Dictionary of forecasted values and metadata for each period
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
        while current_month >= sum(len(range(breakpoints[j] + 1, breakpoints[j + 1] + 1)) 
                                 for j in range(current_segment + 1)):
            current_segment += 1
        
        # Get the parameters for the current segment
        current_segment_params = params[current_segment]
        S0 = current_value
        k = current_segment_params['k']
        
        # Calculate the forecast for one month
        forecast_value = S0 * np.exp(-k * (1))
        
        # Store the forecast with metadata
        forecasts.append({
            'month': current_month,
            'forecasted_value': forecast_value,
            'segment_used': current_segment + 1,
            'time_used': current_month - start_period + 1
        })
        
        # Update current value for next iteration
        current_value = forecast_value
    
    return forecasts 