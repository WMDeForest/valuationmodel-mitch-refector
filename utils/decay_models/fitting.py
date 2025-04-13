"""
Model fitting functions for decay rate estimation.

This module contains functions for fitting exponential decay models to streaming data
and estimating decay rates. These functions serve as the bridge between the raw data
and the forecasting process, extracting the key parameters that drive predictions.
"""
import numpy as np
from scipy.optimize import curve_fit
from utils.decay_models.core import piecewise_exp_decay, exponential_decay
from utils.decay_models.preprocessing import remove_anomalies
from utils.data_processing import sample_data

def fit_segment(months_since_release, streams):
    """
    Fit exponential decay model to a segment of streaming data.
    
    This function uses SciPy's curve_fit to find the optimal parameters (S0, k)
    that minimize the difference between the actual stream data and the predicted
    values from the exponential decay function.
    
    The optimization is constrained to ensure both S0 (initial streams) and k (decay rate)
    are positive values, which makes physical sense for music streaming patterns.
    
    Args:
        months_since_release: Array of months since release (time values)
        streams: Array of stream counts corresponding to each month
        
    Returns:
        tuple: (S0, k) parameters for the fitted model where:
               - S0 is the initial streams value
               - k is the decay rate (higher = faster decay)
    
    Notes:
        This function is typically used for fitting decay patterns in specific time segments
        (e.g., months 1-3, 4-12, 13-36, etc.) as decay rates often change over a track's lifecycle.
        The initial guess for optimization starts with the first data point as S0 and a small
        decay rate (0.01) that is typical for music streaming.
    """
    # Set initial parameter guess based on first observed stream count and typical decay rate
    initial_guess = [streams[0], 0.01]  
    
    # Set bounds to ensure physically meaningful parameters (positive values only)
    bounds = ([0, 0], [np.inf, np.inf])  
    
    # Perform curve fitting to find optimal parameters
    params, covariance = curve_fit(piecewise_exp_decay, months_since_release, streams, 
                                   p0=initial_guess, bounds=bounds)
    
    return params

def calculate_decay_rate(monthly_data):
    """
    Calculate overall decay rate from monthly streaming data (MLDR - Music Listener Decay Rate).
    
    This function:
    1. Converts date values to months since first date to create a time series
    2. Fits an exponential decay curve to the entire dataset
    3. Extracts the decay rate parameter (b) as the MLDR
    
    The MLDR represents the overall rate at which an artist's streaming numbers
    decline over time. This is a key metric used to adjust individual track
    decay rates based on the artist's typical retention patterns.
    
    Args:
        monthly_data: DataFrame with 'Date' and '4_Week_MA' columns
                      Pre-processed data with anomalies removed
        
    Returns:
        tuple: (decay_rate, fitted_parameters) where:
               - decay_rate is the b parameter from exponential_decay
               - fitted_parameters is the (a, b) tuple containing both parameters
    
    Notes:
        The resulting decay rate is used in combination with other factors like 
        playlist reach to adjust the segment-specific decay rates for more
        accurate forecasting.
    """
    # Calculate the number of months since the first date in the filtered data
    min_date = monthly_data['Date'].min()
    monthly_data['Months'] = monthly_data['Date'].apply(
        lambda x: (x.year - min_date.year) * 12 + x.month - min_date.month
    )

    # Fit the exponential decay model to the monthly data
    x_data = monthly_data['Months']
    y_data = monthly_data['4_Week_MA']

    # Use initial guesses for curve fitting - max value as starting point, typical decay rate
    popt, _ = curve_fit(exponential_decay, x_data, y_data, p0=(max(y_data), 0.1))

    # Extract the decay rate (b) - this is the MLDR (Music Listener Decay Rate)
    decay_rate = popt[1]
    return decay_rate, popt 

def calculate_monthly_listener_decay(df_monthly_listeners, start_date=None, end_date=None, sample_rate=7):
    """
    Calculate the decay rate of monthly listeners from any data source.
    
    Parameters:
    -----------
    df_monthly_listeners : DataFrame
        Must contain 'Date' and 'Monthly Listeners' columns
    start_date, end_date : datetime, optional
        Date range to analyze
    sample_rate : int, optional
        Sample every nth row (default: 7 for weekly sampling)
        Set to None or 1 to use all data points
        
    Returns:
    --------
    dict:
        mldr: Monthly listener decay rate
        popt: Fitted parameters
        subset_df: Filtered DataFrame with calculated columns
        min_date: Minimum date in the dataset
        max_date: Maximum date in the dataset
    """
    # Ensure data is sorted
    df = df_monthly_listeners.sort_values(by='Date')
    
    # Sample data if needed (e.g., keep every 7th row for weekly sampling)
    if sample_rate and sample_rate > 1:
        df = sample_data(df, sample_rate)
    
    # Process anomalies
    monthly_data = remove_anomalies(df)
    
    # Get min/max dates
    min_date = monthly_data['Date'].min().to_pydatetime()
    max_date = monthly_data['Date'].max().to_pydatetime()
    
    # Filter by date if specified
    if start_date and end_date:
        mask = (monthly_data['Date'] >= start_date) & (monthly_data['Date'] <= end_date)
        subset_df = monthly_data[mask]
    else:
        subset_df = monthly_data
    
    # Calculate months since first date
    subset_df['Months'] = subset_df['Date'].apply(
        lambda x: (x.year - min_date.year) * 12 + x.month - min_date.month
    )
    
    # Calculate decay rate
    mldr, popt = calculate_decay_rate(subset_df)
    
    return {
        'mldr': mldr, 
        'popt': popt,
        'subset_df': subset_df,
        'min_date': min_date,
        'max_date': max_date
    } 