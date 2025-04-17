"""
Utilities for fitting decay curves to streaming data.

This module provides functions for analyzing listener decay patterns by
fitting mathematical curves to historical streaming data.
"""

import numpy as np
from scipy.optimize import curve_fit
from utils.decay_models.core import piecewise_exp_decay, exponential_decay
from utils.data_processing import remove_anomalies
from utils.data_processing import sample_data
import pandas as pd

def fit_decay_curve(monthly_data):
    """
    Fit an exponential decay curve to prepared monthly listener data.
    
    This is a low-level mathematical function that performs the core curve fitting:
    1. Calculates months since first date for time series analysis
    2. Fits a mathematical exponential decay curve to the data points
    3. Extracts the decay rate parameter (k) that quantifies listener decay
    
    The Monthly Listener Decay Rate (MLDR) represents how quickly an artist loses
    listeners over time. Lower values indicate better listener retention.
    
    This function expects pre-processed data with anomalies already removed.
    For complete end-to-end analysis, use analyze_listener_decay() instead.
    
    Args:
        monthly_data: DataFrame with 'Date' and '4_Week_MA' columns
                      Must be pre-processed with remove_anomalies()
        
    Returns:
        tuple: (mldr, fitted_decay_parameters) where:
               - mldr is the Monthly Listener Decay Rate (a decimal value)
               - fitted_decay_parameters is the array of fitted parameters [S0, k], where
                 S0 is the initial listeners value and k is the decay rate
    
    Example:
        >>> # Only use directly when you need just the mathematical fitting:
        >>> clean_data = remove_anomalies(raw_data)  # Pre-process first
        >>> mldr, params = fit_decay_curve(clean_data)
        >>> print(f"Decay rate: {mldr:.4f}")
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
    fitted_decay_parameters, _ = curve_fit(exponential_decay, x_data, y_data, p0=(max(y_data), 0.1))

    # Extract the decay rate (b) - this is the MLDR (Music Listener Decay Rate)
    decay_rate = fitted_decay_parameters[1]
    return decay_rate, fitted_decay_parameters

def analyze_listener_decay(df_monthly_listeners, start_date=None, end_date=None, sample_rate=7):
    """
    Complete end-to-end analysis of monthly listener decay from any data source.
    
    This high-level function is the recommended entry point for analyzing monthly
    listener data. It handles the entire workflow from raw data to final decay rate:
    
    1. Data preparation (sorting, sampling, date filtering)
    2. Anomaly detection and removal
    3. Time series conversion
    4. Mathematical curve fitting
    5. Decay rate extraction
    
    Unlike the low-level fit_decay_curve() function which just performs mathematical
    fitting, this function handles all preprocessing steps and can work with raw data
    from any source (CSV, API, database).
    
    Parameters:
    -----------
    df_monthly_listeners : DataFrame
        Raw listener data containing:
        - 'Date': Column with dates (will be converted to datetime if not already)
        - 'Monthly Listeners': Column with listener count values
    
    start_date, end_date : datetime, optional
        Date range to analyze. If provided, only data within this range will be used
        for decay rate calculation. Useful for excluding periods with unusual activity.
        
    sample_rate : int, optional
        Controls data density by sampling every nth row:
        - 1: Use all data points (no sampling)
        - 7: Weekly sampling (default, recommended for daily data)
        - 30: Monthly sampling (for very dense data)
        Higher values produce sparser datasets.
        
    Returns:
    --------
    dict: A complete results package containing:
        mldr: Monthly Listener Decay Rate (decimal value, e.g. 0.05 = 5% monthly decline)
        fitted_decay_parameters: Fitted parameters [S0, k] for the exponential decay model
              (S0 is initial listener count, k is decay rate)
        date_filtered_listener_data: Processed DataFrame with calculated columns:
                   - '4_Week_MA': Moving average (smoothed listener counts)
                   - 'Months': Months since first date
                   - 'is_anomaly': Flags for identified anomalies
        min_date: Minimum date in dataset (useful for UI date ranges)
        max_date: Maximum date in dataset (useful for UI date ranges)
        normalized_start_date: Start date normalized to first day of the month
        normalized_end_date: End date normalized to last day of the month
    
    Example:
    --------
    ```python
    # Complete analysis from a CSV file:
    df = pd.read_csv('listener_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    results = analyze_listener_decay(df)
    
    # Access key results:
    decay_rate = results['mldr']
    print(f"Monthly decay rate: {decay_rate:.2%}")  # e.g. "5.25%"
    
    # Visualize the data with matplotlib:
    plt.plot(results['date_filtered_listener_data']['Date'], results['date_filtered_listener_data']['4_Week_MA'])
    ```
    
    Notes:
    ------
    - This function is designed to be the primary entry point for decay analysis
    - For most use cases, you don't need to call the individual processing functions
    - The MLDR is a key metric for valuation models and forecasting
    """
    # Ensure data is sorted
    sorted_monthly_listeners = df_monthly_listeners.sort_values(by='Date')
    
    # Sample data if needed (e.g., keep every 7th row for weekly sampling)
    if sample_rate and sample_rate > 1:
        sorted_monthly_listeners = sample_data(sorted_monthly_listeners, sample_rate)
    
    # Process anomalies
    monthly_data = remove_anomalies(sorted_monthly_listeners)
    
    # Get min/max dates
    min_date = monthly_data['Date'].min().to_pydatetime()
    max_date = monthly_data['Date'].max().to_pydatetime()
    
    # Normalize start and end dates to month boundaries to ensure consistency
    if start_date:
        # Normalize to first day of the month
        normalized_start_date = pd.Timestamp(year=start_date.year, month=start_date.month, day=1)
    else:
        normalized_start_date = pd.Timestamp(year=min_date.year, month=min_date.month, day=1)
        
    if end_date:
        # Normalize to last day of the month
        next_month = end_date.month + 1 if end_date.month < 12 else 1
        next_month_year = end_date.year if end_date.month < 12 else end_date.year + 1
        normalized_end_date = pd.Timestamp(year=next_month_year, month=next_month, day=1) - pd.Timedelta(days=1)
    else:
        next_month = max_date.month + 1 if max_date.month < 12 else 1
        next_month_year = max_date.year if max_date.month < 12 else max_date.year + 1
        normalized_end_date = pd.Timestamp(year=next_month_year, month=next_month, day=1) - pd.Timedelta(days=1)
    
    # Filter by normalized dates
    mask = (monthly_data['Date'] >= normalized_start_date) & (monthly_data['Date'] <= normalized_end_date)
    date_filtered_listener_data = monthly_data[mask]
    
    # Calculate months since first date
    date_filtered_listener_data['Months'] = date_filtered_listener_data['Date'].apply(
        lambda x: (x.year - min_date.year) * 12 + x.month - min_date.month
    )
    
    # Calculate decay rate
    mldr, fitted_decay_parameters = fit_decay_curve(date_filtered_listener_data)
    
    return {
        'mldr': mldr, 
        'fitted_decay_parameters': fitted_decay_parameters,
        'date_filtered_listener_data': date_filtered_listener_data,
        'min_date': min_date,
        'max_date': max_date,
        'normalized_start_date': normalized_start_date,
        'normalized_end_date': normalized_end_date
    }

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

# For backward compatibility
calculate_decay_rate = fit_decay_curve
calculate_monthly_listener_decay = analyze_listener_decay 