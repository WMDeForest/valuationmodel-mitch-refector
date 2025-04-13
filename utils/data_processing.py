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

def format_date(df, column_name, format_str='%d/%m/%Y'):
    """
    Format a datetime column to a specific string format.
    
    Args:
        df: DataFrame containing the datetime column
        column_name: Name of the column to format
        format_str: Format string for the date
        
    Returns:
        DataFrame with formatted date column
    """
    df[column_name] = df[column_name].dt.strftime(format_str)
    return df 