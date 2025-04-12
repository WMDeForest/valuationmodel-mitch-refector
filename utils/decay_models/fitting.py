"""
Model fitting functions for decay rate estimation.
"""
import numpy as np
from scipy.optimize import curve_fit
from utils.decay_models.core import piecewise_exp_decay, exponential_decay

def fit_segment(months_since_release, streams):
    """
    Fit exponential decay model to a segment of streaming data.
    
    Args:
        months_since_release: Array of months since release
        streams: Array of stream counts corresponding to each month
        
    Returns:
        tuple: (S0, k) parameters for the fitted model
    """
    initial_guess = [streams[0], 0.01]  
    bounds = ([0, 0], [np.inf, np.inf])  
    
    params, covariance = curve_fit(piecewise_exp_decay, months_since_release, streams, 
                                   p0=initial_guess, bounds=bounds)
    
    return params

def calculate_decay_rate(monthly_data):
    """
    Calculate decay rate from monthly streaming data.
    
    Args:
        monthly_data: DataFrame with 'Date' and '4_Week_MA' columns
        
    Returns:
        tuple: (decay_rate, fitted_parameters)
    """
    # Calculate the number of months since the first date in the filtered data
    min_date = monthly_data['Date'].min()
    monthly_data['Months'] = monthly_data['Date'].apply(
        lambda x: (x.year - min_date.year) * 12 + x.month - min_date.month
    )

    # Fit the exponential decay model to the monthly data
    x_data = monthly_data['Months']
    y_data = monthly_data['4_Week_MA']

    # Use initial guesses for curve fitting
    popt, _ = curve_fit(exponential_decay, x_data, y_data, p0=(max(y_data), 0.1))

    # Extract the decay rate (b)
    decay_rate = popt[1]
    return decay_rate, popt 