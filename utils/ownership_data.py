"""
Ownership data processing utilities.

This module contains functions for processing and analyzing track ownership data,
including royalty splits, MLC claims, and standardizing ownership percentages.
"""

import pandas as pd

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