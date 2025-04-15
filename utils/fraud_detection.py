"""
Streaming Fraud Detection Utility Module.

This module provides functions to detect potentially fraudulent streaming activity
by analyzing geographic patterns and comparing against population benchmarks.
"""

import pandas as pd


def detect_streaming_fraud(listener_geography_df, population_df, threshold_percentage=20):
    """
    Detect potential streaming fraud by checking if listener counts exceed
    a threshold percentage of a country's population.
    
    Parameters
    ----------
    listener_geography_df : pandas.DataFrame
        DataFrame containing geographical distribution of listeners
    population_df : pandas.DataFrame
        DataFrame containing country population data
    threshold_percentage : float, optional
        Threshold percentage of population to flag as suspicious (default: 20)
        
    Returns
    -------
    list
        List of countries with potential streaming fraud
    """
    # Cross-reference audience data with population data
    warning_df = pd.merge(listener_geography_df, population_df, on='Country', how='left')
    
    # Calculate threshold for suspicious activity
    warning_df['ThresholdPopulation'] = warning_df['Population'] * (threshold_percentage / 100)
    
    # Flag countries with abnormally high listener numbers
    warning_df['AboveThreshold'] = warning_df['Spotify Monthly Listeners'] > warning_df['ThresholdPopulation']
    warning_df['Alert'] = warning_df['AboveThreshold'].apply(lambda x: 1 if x else 0)
    
    # Get list of countries with potential streaming fraud
    alert_countries = warning_df[warning_df['Alert'] == 1]['Country'].tolist()
    
    return alert_countries 