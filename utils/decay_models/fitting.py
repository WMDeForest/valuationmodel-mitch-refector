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
        months_since_release: Array or list of months since release (time values)
        streams: Array or list of stream counts corresponding to each month
        
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
        tuple: (mldr, popt) where:
               - mldr is the Monthly Listener Decay Rate (a decimal value)
               - popt is the array of fitted parameters [S0, k], where
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
    popt, _ = curve_fit(exponential_decay, x_data, y_data, p0=(max(y_data), 0.1))

    # Extract the decay rate (b) - this is the MLDR (Music Listener Decay Rate)
    decay_rate = popt[1]
    return decay_rate, popt 

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
        popt: Fitted parameters [S0, k] for the exponential decay model
              (S0 is initial listener count, k is decay rate)
        subset_df: Processed DataFrame with calculated columns:
                   - '4_Week_MA': Moving average (smoothed listener counts)
                   - 'Months': Months since first date
                   - 'is_anomaly': Flags for identified anomalies
        min_date: Minimum date in dataset (useful for UI date ranges)
        max_date: Maximum date in dataset (useful for UI date ranges)
    
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
    plt.plot(results['subset_df']['Date'], results['subset_df']['4_Week_MA'])
    ```
    
    Notes:
    ------
    - This function is designed to be the primary entry point for decay analysis
    - For most use cases, you don't need to call the individual processing functions
    - The MLDR is a key metric for valuation models and forecasting
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
    mldr, popt = fit_decay_curve(subset_df)
    
    return {
        'mldr': mldr, 
        'popt': popt,
        'subset_df': subset_df,
        'min_date': min_date,
        'max_date': max_date
    }

# For backward compatibility
calculate_decay_rate = fit_decay_curve
calculate_monthly_listener_decay = analyze_listener_decay 