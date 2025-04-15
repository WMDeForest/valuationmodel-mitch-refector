import streamlit as st
import pandas as pd
from utils.financial_parameters import DEFAULT_DISCOUNT_RATE

def display_track_selection_ui(track_catalog_df):
    """Display the track selection UI elements and return the selected songs"""
    st.write("Data Preview:")
    st.write(track_catalog_df)
    songs = sorted(track_catalog_df['track_name'].unique(), key=lambda x: x.lower())
    selected_songs = st.multiselect('Select Songs', songs, default=songs)
    return selected_songs 

def display_financial_parameters_ui():
    """Display the financial parameters input UI and return the values"""
    # Use the default value from financial_parameters module, but allow user override
    discount_rate = st.number_input('Discount Rate (%)', 
                                   min_value=0.00, 
                                   max_value=10.00, 
                                   value=DEFAULT_DISCOUNT_RATE * 100, 
                                   step=0.01, 
                                   format="%.2f") / 100
    return discount_rate 