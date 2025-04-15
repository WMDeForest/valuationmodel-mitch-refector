"""
Functions for updating decay model parameters based on external factors.

This module contains functions that modify decay rate parameters based on 
external influence factors, such as streaming influence factors (formerly called playlist reach),
which can significantly affect the long-term decay patterns of music streams.
"""
import streamlit as st

def update_fitted_params(fitted_params_df, value, sp_range, SP_REACH):
    """
    Update fitted decay parameters based on stream influence factor.
    
    This function adjusts the base decay rates based on a track's stream influence factor
    (formerly called Spotify playlist reach). Tracks with higher influence factors typically 
    have slower decay rates (more sustainability) as they continue to receive streams over time.
    
    The adjustment uses a weighted average approach:
    - 67% weight on the original decay rates
    - 33% weight on the adjustment factors
    
    This balances the intrinsic track decay patterns with the boost from external factors.
    
    Args:
        fitted_params_df: DataFrame with initial fitted parameters
                         Contains segment numbers and corresponding 'k' values
        value: Stream influence factor (numeric measure of external streaming influence)
               [Formerly called Spotify playlist reach]
        sp_range: DataFrame with ranges for segmentation
                 Maps influence values to appropriate adjustment columns
        SP_REACH: DataFrame with adjustment factors for different influence levels
                 Contains modifier columns for different segments
        
    Returns:
        DataFrame: Updated fitted parameters with adjusted 'k' values
                  Returns None if the segment is not found in SP_REACH
    
    Notes:
        The adjustment process:
        1. Determine the appropriate segment based on the influence factor value
        2. Look up the corresponding adjustment column in SP_REACH
        3. Apply a weighted average (67% original, 33% adjustment)
        
        A higher influence factor typically results in a lower decay rate,
        extending the track's commercial lifetime in the forecast.
    """
    updated_fitted_params_df = fitted_params_df.copy()
    
    # Find the appropriate segment based on the influence factor value
    # (formerly called playlist reach value)
    segment = sp_range.loc[(sp_range['RangeStart'] <= value) & 
                          (sp_range['RangeEnd'] > value), 'Column 1'].iloc[0]
    
    # Validate the segment exists in SP_REACH
    if segment not in SP_REACH.columns:
        st.error(f"Error: Column '{segment}' not found in SP_REACH.")
        st.write("Available columns:", SP_REACH.columns)
        return None
    
    # Get the adjustment column and update the decay rate
    # Uses a weighted average: 67% original value, 33% influence-based adjustment
    # (formerly called playlist-based adjustment)
    column_to_append = SP_REACH[segment]
    updated_fitted_params_df['k'] = updated_fitted_params_df['k'] * 0.67 + column_to_append * 0.33

    return updated_fitted_params_df 

def get_decay_parameters(fitted_params_df, stream_influence_factor, sp_range, sp_reach):
    """
    Get decay parameters in both DataFrame and dictionary formats.
    
    This is a comprehensive function that returns both the updated parameters DataFrame
    and the dictionary (records) representation in a single call. It can be used with
    data from any source (API, CSV, database) as long as the fitted_params_df is provided
    in the expected format.
    
    Args:
        fitted_params_df: DataFrame with initial fitted parameters
        stream_influence_factor: Numeric measure of external streaming influence
        sp_range: DataFrame with ranges for segmentation
        sp_reach: DataFrame with adjustment factors for different influence levels
        
    Returns:
        tuple: (updated_params_df, updated_params_dict) where:
               - updated_params_df is the DataFrame with updated parameters
               - updated_params_dict is the list of dictionaries (records format)
               
               Returns (None, None) if parameters could not be updated
    
    Usage:
        # Get both formats:
        df, dict_format = get_decay_parameters(...)
        
        # Get only DataFrame:
        df, _ = get_decay_parameters(...)
        
        # Get only dictionary:
        _, dict_format = get_decay_parameters(...)
    """
    updated_params_df = update_fitted_params(
        fitted_params_df, 
        stream_influence_factor, 
        sp_range, 
        sp_reach
    )
    
    if updated_params_df is not None:
        updated_params_dict = updated_params_df.to_dict(orient='records')
        return updated_params_df, updated_params_dict
    
    return None, None 