"""
Utility functions for data processing and date handling.

This module provides reusable functions for common data processing tasks
such as date conversion, formatting, and validation.
"""

import pandas as pd

def convert_to_datetime(df, column_name, dayfirst=True):
    """
    Convert a column to datetime format with error handling.
    
    Args:
        df: DataFrame containing the date column
        column_name: Name of the column to convert
        dayfirst: Whether to interpret dates as day first format
        
    Returns:
        tuple: (DataFrame with converted dates, list of issues)
    """
    issues = []
    try:
        df[column_name] = pd.to_datetime(df[column_name], dayfirst=dayfirst, errors='coerce')
        
        # Check for unparseable dates
        if df[column_name].isna().any():
            issues.append("Some dates couldn't be parsed and have been set to 'NaT'.")
    except Exception as e:
        issues.append(f"Failed to convert {column_name} column: {e}")
    
    return df, issues

def sample_data(df, sample_rate=7):
    """
    Sample data by keeping every nth row.
    
    Args:
        df: DataFrame to sample
        sample_rate: Take every nth row
        
    Returns:
        DataFrame with sampled data
    """
    return df.iloc[::sample_rate, :]

def select_columns(df, columns):
    """
    Select only specified columns from a DataFrame.
    
    Args:
        df: DataFrame containing the columns
        columns: List of column names to select
        
    Returns:
        DataFrame with only the selected columns
    """
    return df[columns]

def rename_columns(df, column_map):
    """
    Rename columns according to a mapping.
    
    Args:
        df: DataFrame containing the columns to rename
        column_map: Dictionary mapping original column names to new names
        
    Returns:
        DataFrame with renamed columns
    """
    return df.rename(columns=column_map)

def validate_columns(df, required_columns):
    """
    Check if DataFrame contains all required columns.
    
    Args:
        df: DataFrame to validate
        required_columns: List of column names that must be present
        
    Returns:
        bool: True if all required columns are present, False otherwise
    """
    return all(column in df.columns for column in required_columns)

def extract_earliest_date(df, date_column, input_format='%b %d, %Y', output_format='%d/%m/%Y'):
    """
    Extract the earliest date from a DataFrame's date column and format it.
    
    Args:
        df: DataFrame containing the date column
        date_column: Name of the column containing dates
        input_format: Format of the dates in the input data
        output_format: Desired format for the output date
        
    Returns:
        str: The earliest date formatted according to output_format
    """
    earliest_date = pd.to_datetime(df[date_column].iloc[0], format=input_format).strftime(output_format)
    return earliest_date

def calculate_period_streams(df, cumulative_column, days_back):
    """
    Calculate streams for a specific period by comparing the most recent 
    cumulative value with an earlier value.
    
    Args:
        df: DataFrame containing streaming data with cumulative values
        cumulative_column: Name of the column containing cumulative stream values
        days_back: Number of days to look back for the comparison
        
    Returns:
        int/float: Stream count for the specified period
    """
    total_streams = df[cumulative_column].iloc[-1]
    
    if len(df) > days_back:
        period_streams = total_streams - df[cumulative_column].iloc[-(days_back + 1)]
    else:
        period_streams = total_streams
        
    return period_streams

def process_audience_geography(geography_file=None):
    """
    Process audience geography data from uploaded file.
    Returns a DataFrame with audience distribution and the USA percentage.
    
    Parameters:
    -----------
    geography_file : file object, optional
        The uploaded CSV file containing audience geography data
        
    Returns:
    --------
    tuple:
        A tuple containing two elements:
        
        listener_geography_df : pandas.DataFrame
            A DataFrame containing geographical distribution of listeners with columns:
            - 'Country': The country name
            - 'Spotify Monthly Listeners': Raw count of listeners from this country
            - 'Spotify monthly listeners (%)': Percentage of total listeners (as a decimal)
            This data is used to apply country-specific royalty rates during valuation.
            
        listener_percentage_usa : float
            The proportion of listeners from the United States as a decimal (0.0-1.0).
            This is extracted from the listener_geography_df for convenience since US streams
            are often calculated separately in royalty formulas.
            Defaults to 1.0 (100% USA) if no geography data is provided or if USA
            is not found in the data.
    """
    # Default value if no geography data
    listener_percentage_usa = 1.0
    listener_geography_df = pd.DataFrame()
    
    if geography_file:
        # Process uploaded file
        listener_geography_df = pd.read_csv(geography_file)
        
        # Extract and process geographical data
        listener_geography_df = listener_geography_df[['Country', 'Spotify Monthly Listeners']]
        listener_geography_df = listener_geography_df.groupby('Country', as_index=False)['Spotify Monthly Listeners'].sum()
        
        # Calculate percentage distribution
        total_listeners = listener_geography_df['Spotify Monthly Listeners'].sum()
        listener_geography_df['Spotify monthly listeners (%)'] = (listener_geography_df['Spotify Monthly Listeners'] / total_listeners) * 100
        
        # Normalize percentage values
        listener_geography_df["Spotify monthly listeners (%)"] = pd.to_numeric(listener_geography_df["Spotify monthly listeners (%)"], errors='coerce')
        listener_geography_df["Spotify monthly listeners (%)"] = listener_geography_df["Spotify monthly listeners (%)"] / 100
        
        # Extract US percentage for royalty calculations
        if "United States" in listener_geography_df["Country"].values:
            listener_percentage_usa = listener_geography_df.loc[listener_geography_df["Country"] == "United States", "Spotify monthly listeners (%)"].values[0]
    
    return listener_geography_df, listener_percentage_usa

def process_ownership_data(ownership_file, track_names):
    """
    Process song ownership data from uploaded file.
    
    Parameters:
    -----------
    ownership_file : file object, optional
        The uploaded CSV file containing ownership data
    track_names : list
        List of track names for which to create default ownership entries if no file is provided

    Returns:
    --------
    pandas.DataFrame:
        A DataFrame containing standardized ownership information with columns:
        - 'track_name': Name of the track
        - 'Ownership(%)': Percentage ownership as a decimal (0.0-1.0)
        - 'MLC Claimed(%)': Percentage of mechanical license claims as a decimal (0.0-1.0)
        
        Both percentage values are normalized to decimal format (e.g., 50% → 0.5).
        When no ownership file is provided, defaults to 100% ownership (1.0) and 
        0% MLC claims (0.0) for all tracks in track_names. 
    """
    if ownership_file is not None:
        # Load ownership data with encoding handling
        try:
            ownership_df = pd.read_csv(ownership_file, encoding='latin1')
        except UnicodeDecodeError:
            ownership_df = pd.read_csv(ownership_file, encoding='utf-8')
    
        # Clean and normalize ownership data
        ownership_df['Ownership(%)'] = ownership_df['Ownership(%)'].replace('', 1)
        ownership_df['MLC Claimed(%)'] = ownership_df['MLC Claimed(%)'].replace('', 0)
        ownership_df['Ownership(%)'] = pd.to_numeric(ownership_df['Ownership(%)'], errors='coerce').fillna(1)
        ownership_df['MLC Claimed(%)'] = pd.to_numeric(ownership_df['MLC Claimed(%)'], errors='coerce').fillna(0)
        
        # Convert percentages to decimal format
        ownership_df['Ownership(%)'] = ownership_df['Ownership(%)'].apply(lambda x: x / 100 if x > 1 else x)
        ownership_df['MLC Claimed(%)'] = ownership_df['MLC Claimed(%)'].apply(lambda x: x / 100 if x > 1 else x)
    else:
        # Create empty ownership dataframe if no file is uploaded
        ownership_df = pd.DataFrame({
            'track_name': track_names,
            'Ownership(%)': [1.0] * len(track_names),
            'MLC Claimed(%)': [0.0] * len(track_names)
        })
    
    return ownership_df 

def calculate_months_since_release(release_date_str, date_format="%d/%m/%Y"):
    """
    Calculate the number of months between a release date and today.
    
    Parameters:
    -----------
    release_date_str : str
        The release date as a string
    date_format : str, optional
        The format of the release date string, defaults to "%d/%m/%Y"
        
    Returns:
    --------
    int:
        The number of months since the release date
    """
    from datetime import datetime
    
    tracking_start_date = datetime.strptime(release_date_str, date_format)
    delta = datetime.today() - tracking_start_date
    return delta.days // 30

def calculate_monthly_stream_averages(streams_last_30days, streams_last_90days, streams_last_365days, months_since_release):
    """
    Calculate average monthly streams for different time periods.
    
    Parameters:
    -----------
    streams_last_30days : int
        Number of streams in the last 30 days
    streams_last_90days : int
        Number of streams in the last 90 days
    streams_last_365days : int
        Number of streams in the last 365 days
    months_since_release : int
        Number of months since the track was released
        
    Returns:
    --------
    tuple:
        (avg_monthly_streams_months_4to12, avg_monthly_streams_months_2to3)
        Average streams per month for months 4-12 and 2-3
    """
    # ===== MONTH 2-3 AVERAGE =====
    # This calculates the average monthly streams for months 2-3 (days 31-90)
    # We take the difference between 90-day streams and 30-day streams to isolate days 31-90
    # Then divide by 2 months (or 1 month if the track is very new)
    avg_monthly_streams_months_2to3 = (streams_last_90days - streams_last_30days) / (2 if months_since_release > 2 else 1)
    
    # ===== MONTH 4-12 AVERAGE =====
    # This calculates the average monthly streams for months 4-12 (days 91-365)
    # We take the difference between 365-day streams and 90-day streams to isolate days 91-365
    if months_since_release > 3:
        # If track is at least a year old, divide by 9 months (months 4-12)
        # If track is 4-11 months old, divide by the actual number of months available
        avg_monthly_streams_months_4to12 = (streams_last_365days - streams_last_90days) / (9 if months_since_release > 11 else (months_since_release - 3))
    else:
        # For very new tracks (less than 4 months), use the months 2-3 average as a proxy
        # since we don't have enough data for a separate calculation
        avg_monthly_streams_months_4to12 = avg_monthly_streams_months_2to3
        
    return avg_monthly_streams_months_4to12, avg_monthly_streams_months_2to3

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
    import numpy as np
    
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