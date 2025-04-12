"""
Functions for updating decay model parameters based on external factors.
"""
import streamlit as st

def update_fitted_params(fitted_params_df, value, sp_range, SP_REACH):
    """
    Update fitted decay parameters based on Spotify playlist reach.
    
    Args:
        fitted_params_df: DataFrame with initial fitted parameters
        value: Spotify playlist reach value
        sp_range: DataFrame with ranges for segmentation
        SP_REACH: DataFrame with adjustment factors
        
    Returns:
        DataFrame: Updated fitted parameters
    """
    updated_fitted_params_df = fitted_params_df.copy()
    
    # Find the appropriate segment based on the reach value
    segment = sp_range.loc[(sp_range['RangeStart'] <= value) & 
                          (sp_range['RangeEnd'] > value), 'Column 1'].iloc[0]
    
    # Validate the segment exists in SP_REACH
    if segment not in SP_REACH.columns:
        st.error(f"Error: Column '{segment}' not found in SP_REACH.")
        st.write("Available columns:", SP_REACH.columns)
        return None
    
    # Get the adjustment column and update the decay rate
    # Uses a weighted average: 67% original value, 33% playlist-based adjustment
    column_to_append = SP_REACH[segment]
    updated_fitted_params_df['k'] = updated_fitted_params_df['k'] * 0.67 + column_to_append * 0.33

    return updated_fitted_params_df 