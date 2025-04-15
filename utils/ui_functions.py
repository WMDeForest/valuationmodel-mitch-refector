import streamlit as st
import pandas as pd


def display_track_selection_ui(track_catalog_df):
    """Display the track selection UI elements and return the selected songs"""
    st.write("Data Preview:")
    st.write(track_catalog_df)
    songs = sorted(track_catalog_df['track_name'].unique(), key=lambda x: x.lower())
    selected_songs = st.multiselect('Select Songs', songs, default=songs)
    return selected_songs 

def display_financial_parameters_ui():
    """Display the financial parameters input UI and return the values"""
    # Define default discount rate locally
    default_discount_rate = 4.50  # 4.5%
    discount_rate = st.number_input('Discount Rate (%)', 
                                   min_value=0.00, 
                                   max_value=10.00, 
                                   value=default_discount_rate,
                                   step=0.01, 
                                   format="%.2f") / 100
    return discount_rate 

def format_valuation_results(valuation_df):
    """
    Format valuation results for display in the UI with commas and currency symbols.
    
    Parameters:
    -----------
    valuation_df : pd.DataFrame
        DataFrame containing valuation results with numeric values
        
    Returns:
    --------
    pd.DataFrame
        Formatted DataFrame with values converted to strings with commas and currency symbols
    """
    # Create a copy to avoid modifying the original DataFrame
    formatted_df = valuation_df.copy()
    
    # Format stream counts with commas
    formatted_df['historical_streams'] = formatted_df['historical_streams'].astype(float).apply(lambda x: f"{int(round(x)):,}")
    formatted_df['forecast_streams'] = formatted_df['forecast_streams'].astype(float).apply(lambda x: f"{int(round(x)):,}")
    
    # Format monetary values with dollar signs and commas
    formatted_df['undiscounted_future_royalty'] = formatted_df['undiscounted_future_royalty'].astype(float).apply(lambda x: f"${int(round(x)):,}")
    formatted_df['discounted_future_royalty'] = formatted_df['discounted_future_royalty'].astype(float).apply(lambda x: f"${int(round(x)):,}")
    formatted_df['historical_royalty_value'] = formatted_df['historical_royalty_value'].astype(float).apply(lambda x: f"${int(round(x)):,}")
    formatted_df['total_track_valuation'] = formatted_df['total_track_valuation'].astype(float).apply(lambda x: f"${int(round(x)):,}")
    
    return formatted_df 

def display_valuation_results(valuation_df):
    """
    Format and display valuation results in the Streamlit UI.
    
    Parameters:
    -----------
    valuation_df : pd.DataFrame
        DataFrame containing valuation results with numeric values
    """
    # Format the valuation results
    formatted_df = format_valuation_results(valuation_df)
    
    # Display the formatted results
    st.write(formatted_df)
    return formatted_df 

def display_valuation_summary(valuation_df):
    """
    Calculate summed values across all tracks and display a summary table.
    
    Parameters:
    -----------
    valuation_df : pd.DataFrame
        Formatted DataFrame containing valuation results for all tracks
        
    Returns:
    --------
    pd.DataFrame
        Summary DataFrame with totals across all tracks in the catalog
    """
    # Extract numeric values from formatted strings
    numeric_df = valuation_df.copy()
    
    # Convert string currency values back to numbers for summing
    for col in ['historical_royalty_value', 'undiscounted_future_royalty', 
                'discounted_future_royalty', 'total_track_valuation']:
        numeric_df[col] = numeric_df[col].str.replace('$', '').str.replace(',', '').astype(float)
    
    # Calculate sums across all tracks in the catalog
    catalog_valuation_summary_df = pd.DataFrame({
        'Metric': ['Historical Value', 'Undiscounted Future Value', 'Discounted Future Value', 'Total Valuation'],
        'Sum': [
            numeric_df['historical_royalty_value'].sum(),
            numeric_df['undiscounted_future_royalty'].sum(),
            numeric_df['discounted_future_royalty'].sum(),
            numeric_df['total_track_valuation'].sum()
        ]
    })
    
    # Format sum values with dollar signs and commas
    catalog_valuation_summary_df['Sum'] = catalog_valuation_summary_df['Sum'].apply(lambda x: f"${int(round(x)):,}")
    
    # Display the summary table with a title
    st.write("Summed Values:")
    st.write(catalog_valuation_summary_df)
    
    return catalog_valuation_summary_df 