import streamlit as st
import pandas as pd

def display_track_selection_ui(track_catalog_df):
    """Display the track selection UI elements and return the selected songs"""
    st.write("Data Preview:")
    st.write(track_catalog_df)
    songs = sorted(track_catalog_df['track_name'].unique(), key=lambda x: x.lower())
    selected_songs = st.multiselect('Select Songs', songs, default=songs)
    return selected_songs 