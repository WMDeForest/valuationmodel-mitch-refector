"""
Data Loader Utility Module
--------------------------

This module provides functions for loading and accessing various data sources used
in the valuation model application. It handles CSV file loading with proper error handling
and provides clean access to common data sources.

Key functions:
- load_local_csv: Generic CSV loader with error handling
- get_mech_data: Loads mechanical royalty data
- get_rates_data: Loads worldwide rates data

Usage:
    from utils.data_loader import get_mech_data, get_rates_data
    
    mech_data = get_mech_data()
    rates_data = get_rates_data()
"""

import pandas as pd
import streamlit as st
import os

# Define file paths for local CSV files
FILE_PATH_MECH = os.path.join("data", "MECHv2.csv")
FILE_PATH_RATES = os.path.join("data", "worldwide_rates_final.csv")

def load_local_csv(file_path):
    """
    Load a local CSV file into a pandas DataFrame
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pandas.DataFrame: Loaded data or None if loading fails
    """
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Failed to load local file {file_path}: {e}")
        return None

def get_mech_data():
    """
    Load the MECH data file
    
    Returns:
        pandas.DataFrame: MECH data
    """
    return load_local_csv(FILE_PATH_MECH)

def get_rates_data():
    """
    Load the worldwide rates data file
    
    Returns:
        pandas.DataFrame: Worldwide rates data
    """
    return load_local_csv(FILE_PATH_RATES) 