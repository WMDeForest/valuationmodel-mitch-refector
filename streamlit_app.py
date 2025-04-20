#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd

# Import tab components
from tabs.process_visualize_track_data import process_and_visualize_track_data
from tabs.chartmetric_tab import render_chartmetric_tab

# ===== APP INTERFACE SETUP =====
st.title('mitch_refactor_valuation_app')

# Create main navigation tabs
tab1, tab2 = st.tabs(["Process & Visualize", "CM API Search"])

with tab1:
    # Handle file uploads in the main app
    st.header("Upload Your Data Files")
    
    # 1. Artist Monthly Listeners Data
    uploaded_file = st.file_uploader("Artist Monthly Spotify Listeners", type="csv")
    artist_monthly_listeners_df = None
    if uploaded_file is not None:
        artist_monthly_listeners_df = pd.read_csv(uploaded_file)
    
    # 2. Track Catalog Data
    uploaded_catalog_file = st.file_uploader("Track Catalog CSV", type=["csv"], 
                                            help="Upload a single CSV containing data for multiple tracks.")
    
    # 3. Audience Geography Data
    uploaded_file_audience_geography = st.file_uploader("Audience Geography", type=["csv"])
    
    # 4. Ownership Data
    uploaded_file_ownership = st.file_uploader("MLC Claimed and Song Ownership", type="csv")
    
    # Pass all data to the analysis function
    process_and_visualize_track_data(
        artist_monthly_listeners_df=artist_monthly_listeners_df,
        catalog_file_data=uploaded_catalog_file,  # Pass the file object directly for parsing
        audience_geography_data=uploaded_file_audience_geography,
        ownership_data=uploaded_file_ownership
    )

with tab2:
    # Render the ChartMetric API tab
    render_chartmetric_tab()

