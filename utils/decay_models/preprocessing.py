"""
Data preprocessing functions for decay modeling.

This module contains functions for cleaning streaming data and preparing it
for decay rate modeling. The main functionality involves detecting and handling
anomalies (unusual spikes or drops) in streaming data.
"""

import pandas as pd

def remove_anomalies(data):
    """
    Clean streaming data by removing outliers using IQR method and interpolation.

    This function performs several key preprocessing steps:
    1. Calculates a 4-week moving average to smooth out daily or weekly fluctuations
    2. Resamples the smoothed data to monthly sums for more stable trend analysis
    3. Detects anomalies using the Interquartile Range (IQR) method
    4. Replaces identified anomalies with interpolated values from adjacent months

    The IQR method defines anomalies as points falling outside the range:
    [Q1 - 1.5 * IQR, Q3 + 1.5 * IQR] where:
    - Q1 is the 25th percentile
    - Q3 is the 75th percentile
    - IQR is Q3 - Q1

    Args:
        data: DataFrame with 'Date' and 'Monthly Listeners' columns
                Date column must be datetime format
                Monthly Listeners column contains the raw streaming counts
        
    Returns:
        DataFrame with following columns:
        - 'Date': Original date (now index)
        - '4_Week_MA': 4-week moving average of monthly listeners
        - 'is_anomaly': Boolean flag indicating detected anomalies
        
    Notes:
        This preprocessing step is crucial for obtaining reliable decay rate estimates
        as streaming anomalies (e.g., from viral events, playlist placements, or data errors)
        would otherwise significantly distort the decay curve fitting.
    """
    # Calculate the 4-week moving average
    data['4_Week_MA'] = data['Monthly Listeners'].rolling(window=4, min_periods=1).mean()

    # Resample the data to monthly sums
    data.set_index('Date', inplace=True)
    monthly_data = data['4_Week_MA'].resample('M').sum().reset_index()

    # Detect anomalies using the IQR method
    Q1 = monthly_data['4_Week_MA'].quantile(0.25)
    Q3 = monthly_data['4_Week_MA'].quantile(0.75)
    IQR = Q3 - Q1

    # Define bounds for anomaly detection
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Detect anomalies
    monthly_data['is_anomaly'] = (monthly_data['4_Week_MA'] < lower_bound) | (monthly_data['4_Week_MA'] > upper_bound)

    # Interpolate anomalies
    for i in range(1, len(monthly_data) - 1):
        if monthly_data.loc[i, 'is_anomaly']:
            # Replace anomalous value with average of adjacent months
            monthly_data.loc[i, '4_Week_MA'] = (monthly_data.loc[i - 1, '4_Week_MA'] + monthly_data.loc[i + 1, '4_Week_MA']) / 2

    return monthly_data 