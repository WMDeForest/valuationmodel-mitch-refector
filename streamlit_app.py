#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import json
import streamlit as st
import re
import base64
from datetime import datetime, timedelta
from scipy.optimize import curve_fit
from io import StringIO
from sklearn.metrics import r2_score
import requests

# Replace the direct import with the initialized service
from services.chartmetric_services import chartmetric_service as chartmetric
from services.chartmetric_services.http_client import RequestsHTTPClient

# Import data loading functions from utils module
from utils.data_loader import get_mech_data, get_rates_data, load_local_csv
from utils.population_utils.population_data import get_population_data
from utils.population_utils.country_code_to_name import country_code_to_name
from utils.data_processing import (
    convert_to_datetime, 
    sample_data,
    select_columns,
    rename_columns,
    validate_columns,
    extract_earliest_date,
    calculate_period_streams,
    process_audience_geography,
    process_ownership_data,
    calculate_months_since_release,
    calculate_monthly_stream_averages,
    prepare_decay_rate_fitting_data
)
from utils.decay_rates import (
    ranges_sp,
    sp_range,
    SP_REACH_DATA,
    SP_REACH,
    fitted_params,
    fitted_params_df,
    track_lifecycle_segment_boundaries,
    DEFAULT_STREAM_INFLUENCE_FACTOR,
    DEFAULT_FORECAST_PERIODS,
    DEFAULT_FORECAST_YEARS
)

# Import decay model functions from the new modules
from utils.decay_models import (
    piecewise_exp_decay,
    exponential_decay,
    remove_anomalies,
    fit_decay_curve,
    fit_segment,
    update_fitted_params,
    get_decay_parameters,
    forecast_track_streams,
    analyze_listener_decay
)

# Import decay rate adjustment functions
from utils.decay_models.parameter_updates import (
    generate_track_decay_rates_by_month,
    create_decay_rate_dataframe,
    adjust_track_decay_rates,
    calculate_track_decay_rates_by_segment
)

# Import UI functions
from utils.ui_functions import (
    display_financial_parameters_ui, 
    display_valuation_results,
    display_valuation_summary,
    create_country_distribution_chart,
    create_yearly_revenue_chart
)

# Import financial parameters
from utils.financial_parameters import (
    PREMIUM_STREAM_PERCENTAGE, 
    AD_SUPPORTED_STREAM_PERCENTAGE,
    HISTORICAL_VALUE_TIME_ADJUSTMENT
)

# Import historical value calculation function
from utils.historical_royalty_revenue import calculate_historical_royalty_revenue, HISTORICAL_VALUATION_CUTOFF

# Import forecast projections functions
from utils.forecast_projections import (
    create_monthly_track_revenue_projections,
    aggregate_into_yearly_periods,
    apply_ownership_adjustments
)

# Import geographic analysis functions
from utils.geographic_analysis import (
    process_country_breakdown,
    get_top_countries
)

# Import fraud detection
from utils.fraud_detection import detect_streaming_fraud

# ===== MODELING FUNCTIONS =====

# ===== DATA LOADING - GLOBAL DATASETS =====
# Load country population data for analyzing geographic streaming patterns
# and detecting potential anomalies (e.g., streams exceeding realistic population penetration)
population_df = get_population_data()

# Load mechanical royalties data - contains historical royalty rates for Spotify streams
# Used to calculate historical value and forecast financial projections
mechanical_royalty_rates_df = get_mech_data()
if mechanical_royalty_rates_df is None:
    st.error("Failed to load mechanical royalties data")
    st.stop()

# Load worldwide rates data - contains country-specific royalty rates
# Used to calculate revenue projections based on geographic streaming distribution
worldwide_royalty_rates_df = get_rates_data()
if worldwide_royalty_rates_df is None:
    st.error("Failed to load worldwide rates data")
    st.stop()

# ===== APP INTERFACE SETUP =====
st.title('mitch_refactor_valuation_app')

# Create main navigation tabs
tab1, tab2= st.tabs(["File Uploader", "placeholder"])

with tab1:
    # ===== TAB 1: SPOTIFY MONTHLY LISTENERS ANALYSIS =====
    # 1. DATA UPLOAD
    uploaded_file = st.file_uploader("Artist Monthly Spotify Listeners", type="csv")

    if uploaded_file is not None:
        # ===== DATA PROCESSING SECTION =====
        # 1. DATA LOADING
        # Read the uploaded CSV file
        artist_monthly_listeners_df = pd.read_csv(uploaded_file)
        
        # 2. DATA VALIDATION
        # Check if required columns exist
        if validate_columns(artist_monthly_listeners_df, ['Date', 'Monthly Listeners']):
            # 3. DATA SELECTION
            # Keep only required columns
            columns_to_keep = ['Date', 'Monthly Listeners']
            artist_monthly_listeners_df = select_columns(artist_monthly_listeners_df, columns_to_keep)
            
            # 4. DATE CONVERSION
            # Convert 'Date' column to datetime format with error handling
            artist_monthly_listeners_df, date_issues = convert_to_datetime(artist_monthly_listeners_df, 'Date', dayfirst=True)
            
            # Display any issues with date conversion
            for issue in date_issues:
                if "Failed to convert" in issue:
                    st.error(issue)
                else:
                    st.warning(f"{issue} Please check your data.")
            
           
            # 5. INITIAL DECAY ANALYSIS
            # Calculate decay rates for the full dataset initially
            decay_analysis = analyze_listener_decay(artist_monthly_listeners_df)

            # ===== UI COMPONENTS SECTION =====
            # Get normalized dates for consistent month-based calculations
            min_date = decay_analysis['normalized_start_date'].date()  # Convert to datetime.date
            max_date = decay_analysis['normalized_end_date'].date()    # Convert to datetime.date
            
            # 6. DATE RANGE SELECTION
            st.write("Select Date Range:")
            start_date, end_date = st.slider(
                "Select date range",
                min_value=min_date,
                max_value=max_date,
                value=(min_date, max_date),
                format="YYYY-MM-DD"
            )

            # Convert slider values to Timestamp objects
            start_date = pd.Timestamp(start_date)
            end_date = pd.Timestamp(end_date)

            # Update the decay analysis if the user changes the date range
            if start_date != decay_analysis['normalized_start_date'] or end_date != decay_analysis['normalized_end_date']:
                decay_analysis = analyze_listener_decay(artist_monthly_listeners_df, start_date, end_date)
            
            # Extract required data from the analysis
            date_filtered_listener_data = decay_analysis['date_filtered_listener_data']
            mldr = decay_analysis['mldr']
            fitted_decay_parameters = decay_analysis['fitted_decay_parameters']
            normalized_start_date = decay_analysis['normalized_start_date']
            normalized_end_date = decay_analysis['normalized_end_date']
            
            # ===== RESULTS DISPLAY SECTION =====
            # 8. SHOW METRICS
            st.write(f'Exponential decay rate: {mldr}')
            
            # Always show normalized dates for transparency
            st.write(f'Normalized date range used: {normalized_start_date.strftime("%Y-%m-%d")} to {normalized_end_date.strftime("%Y-%m-%d")}')
            
            # 9. VISUALIZATION
            fig, ax = plt.subplots(figsize=(10, 4))
            # Plot the moving average
            ax.plot(date_filtered_listener_data['Date'], date_filtered_listener_data['4_Week_MA'], label='Moving Average', color='tab:blue', linewidth=2)
            # Plot the fitted decay curve using pre-calculated parameters
            ax.plot(date_filtered_listener_data['Date'], exponential_decay(date_filtered_listener_data['Months'], *fitted_decay_parameters), 
                   label='Fitted Decay Curve', color='red', linestyle='--')
            
            # Plot formatting and styling
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Monthly Listeners', fontsize=12)
            ax.set_title(f'Moving Average and Exponential Decay', fontsize=14, weight='bold')
            ax.legend()
            ax.set_ylim(bottom=0)
            plt.xticks(rotation=45)
            
            # Visual enhancements
            fig.patch.set_visible(False)
            ax.set_facecolor('none')
            ax.patch.set_alpha(0)
            plt.tight_layout()
            
            # Display the plot
            st.pyplot(fig)
        else:
            st.error("The uploaded file does not contain the required columns 'Date' and 'Monthly Listeners'.")
            
    # ===== TRACK DATA MANAGEMENT =====
    # 1. FILE UPLOADS SECTION
    #uploaded_file = st.file_uploader("Tracklist", type=["csv"])
    uploaded_files_unique = st.file_uploader("Monthly Track Spotify Streams", type=['csv'], accept_multiple_files=True)
    uploaded_file_3 = st.file_uploader("Audience Geography", type=["csv"])
    uploaded_file_ownership = st.file_uploader("MLC Claimed and Song Ownership", type="csv")

    # Store track names for UI display without processing files yet
    track_names = []
    if uploaded_files_unique:
        for file_unique in uploaded_files_unique:
            # Extract track name from the filename (expected format: "Artist - TrackName.csv")
            track_name_unique = file_unique.name.split(' - ')[1].strip()
            track_names.append(track_name_unique)

    # ===== UI DISPLAY AND TRACK SELECTION =====
    selected_songs = st.multiselect("Select Songs to Analyze", track_names)
    
    # ===== FINANCIAL PARAMETERS =====
    # The discount rate is used to:
    #  1. Convert future projected royalty earnings to present value
    #  2. Adjust historical value calculations for time value
    #  3. Account for risk and opportunity cost in the valuation model
    # The default of 4.5% represents a moderate risk profile for music royalty assets
    discount_rate = display_financial_parameters_ui()

    # ===== RUN BUTTON =====
    if st.button('Run All'):
        # Initialize data structures to store results
        track_yearly_revenue_collection = []
        export_track_streams_forecast = pd.DataFrame()
        track_valuation_summaries = []
        
        # ===== TRACK DATA PROCESSING =====
        # Initialize an empty DataFrame that will store data for all tracks (our catalog)
        track_catalog_df = pd.DataFrame()
        
        # Process each uploaded track file - iterate through all CSV files the user uploaded
        for file_unique in uploaded_files_unique:
            # Extract track name from the filename (expected format: "Artist - TrackName.csv")
            track_name_unique = file_unique.name.split(' - ')[1].strip()
            
            # Skip processing if the track wasn't selected
            if track_name_unique not in selected_songs:
                continue
                
            # Read the track's streaming data from CSV
            df_track_data_unique = pd.read_csv(file_unique)
            
            # Make column names more descriptive - 'Value' becomes 'CumulativeStreams'
            df_track_data_unique = rename_columns(df_track_data_unique, {'Value': 'CumulativeStreams'})

            # Get the first date from the data (not necessarily release date, just first tracking date)
            earliest_track_date = extract_earliest_date(df_track_data_unique, 'Date')
            
            # Extract the latest (most recent) cumulative stream count
            total_historical_track_streams = df_track_data_unique['CumulativeStreams'].iloc[-1]

            # Calculate period-specific stream counts using our utility function
            # These represent streams in the last 30/90/365 days
            track_streams_last_30days = calculate_period_streams(df_track_data_unique, 'CumulativeStreams', 30)
            track_streams_last_90days = calculate_period_streams(df_track_data_unique, 'CumulativeStreams', 90)
            track_streams_last_365days = calculate_period_streams(df_track_data_unique, 'CumulativeStreams', 365)
            
            # Pre-calculate time-based metrics to avoid recalculating after Run All button
            months_since_release_total = calculate_months_since_release(earliest_track_date)
            
            # Calculate monthly averages for different time periods
            avg_monthly_streams_months_4to12, avg_monthly_streams_months_2to3 = calculate_monthly_stream_averages(
                track_streams_last_30days,
                track_streams_last_90days,
                track_streams_last_365days,
                months_since_release_total
            )
            
            # Prepare arrays for decay rate fitting
            months_since_release, monthly_averages = prepare_decay_rate_fitting_data(
                months_since_release_total,
                avg_monthly_streams_months_4to12,
                avg_monthly_streams_months_2to3,
                track_streams_last_30days
            )
            
            # Create a single row DataFrame containing all key metrics for this track
            track_data = pd.DataFrame({
                'track_name': [track_name_unique],
                'earliest_track_date': [earliest_track_date],  # Keep original format for any other uses
                'track_streams_last_30days': [track_streams_last_30days],
                'track_streams_last_90days': [track_streams_last_90days],
                'track_streams_last_365days': [track_streams_last_365days],
                'total_historical_track_streams': [total_historical_track_streams],
                'months_since_release_total': [months_since_release_total],
                'avg_monthly_streams_months_4to12': [avg_monthly_streams_months_4to12],
                'avg_monthly_streams_months_2to3': [avg_monthly_streams_months_2to3],
                'months_since_release': [months_since_release.tolist() if hasattr(months_since_release, 'tolist') else months_since_release],
                'monthly_averages': [monthly_averages.tolist() if hasattr(monthly_averages, 'tolist') else monthly_averages]
            })
            
            # Add this track's data to our catalog of all tracks
            # Each time through the loop, we add one more row to the catalog
            track_catalog_df = pd.concat([track_catalog_df, track_data], ignore_index=True)

        # ===== AUDIENCE GEOGRAPHY PROCESSING =====
        # Process the audience geography data to determine geographic distribution of listeners
        # This data is used to apply country-specific royalty rates in revenue projections
        # If no geography data is provided, assume 100% US market for royalty calculations
        listener_geography_df, listener_percentage_usa = process_audience_geography(uploaded_file_3)

        # ===== OWNERSHIP DATA PROCESSING =====
        # Process ownership and MLC claim information to accurately calculate revenue shares
        # This ensures all calculations account for partial ownership and existing royalty claims
        # If no ownership data is provided, assume 100% ownership and 0% MLC claims
        ownership_df = process_ownership_data(uploaded_file_ownership, track_names)

        # Process each selected song
        for selected_song in selected_songs:
            # ===== 1. EXTRACT SONG DATA =====
            if selected_song not in track_catalog_df['track_name'].values:
                st.error(f"Data for {selected_song} not processed properly. Skipping...")
                continue
                
            song_data = track_catalog_df[track_catalog_df['track_name'] == selected_song].iloc[0]
            track_streams_last_365days = song_data['track_streams_last_365days']
            track_streams_last_90days = song_data['track_streams_last_90days']
            track_streams_last_30days = song_data['track_streams_last_30days']
            total_historical_track_streams = song_data['total_historical_track_streams']
            
            # Get both date formats - we need raw format for months calculation
            earliest_track_date = song_data['earliest_track_date']  # Original DD/MM/YYYY format
            
            # Pre-calculate months since release using original date format
            months_since_release_total = calculate_months_since_release(earliest_track_date)
            
            # ===== 2. UPDATE DECAY PARAMETERS =====
            # Get both DataFrame and dictionary formats of decay parameters
            decay_rates_df, updated_fitted_params = get_decay_parameters(
                fitted_params_df, 
                DEFAULT_STREAM_INFLUENCE_FACTOR, 
                sp_range, 
                SP_REACH
            )
            
            # ===== 3. RETRIEVE DECAY FITTING DATA FOR MODELING =====
            # Get arrays for decay curve fitting (only the data needed immediately)
            track_months_since_release = song_data['months_since_release']
            monthly_averages = song_data['monthly_averages']
            
            # ===== 4. FIT DECAY MODEL TO STREAM DATA =====
            params = fit_segment(track_months_since_release, monthly_averages)
            S0, track_decay_k = params
            
            # Generate track-specific decay rates for all forecast months 
            # These rates model how this individual track's streams will decline over time,
            # distinct from the artist-level decay rates calculated in the previous section
            track_monthly_decay_rates = generate_track_decay_rates_by_month(decay_rates_df, track_lifecycle_segment_boundaries)
            
            # Determine the observed time range from the track's streaming data
            track_data_start_month = min(track_months_since_release)
            track_data_end_month = max(track_months_since_release)
            
            # Create a structured DataFrame that combines model-derived decay rates with observed data
            # This DataFrame is critical for adjusting theoretical decay curves with actual observed patterns
            track_decay_rate_df = create_decay_rate_dataframe(
                track_months_since_release=list(range(1, 501)),  # Forecast for 500 months (about 41.7 years) of track lifetime
                track_monthly_decay_rates=track_monthly_decay_rates,  # Using our track-specific decay rates
                mldr=mldr,  # Monthly Listener Decay Rate from artist-level analysis
                track_data_start_month=track_data_start_month,  # First month we have actual track streaming data
                track_data_end_month=track_data_end_month       # Last month we have actual track streaming data
            )
            
            # ===== 5. ADJUST TRACK DECAY RATES BASED ON OBSERVED DATA =====
            # Apply a two-stage adjustment using observed artist and track data
            adjusted_track_decay_df, track_adjustment_info = adjust_track_decay_rates(
                track_decay_rate_df, 
                track_decay_k=track_decay_k  # Track-specific fitted decay parameter
            )
            
            # ===== 6. SEGMENT DECAY RATES BY TIME PERIOD =====
            # Calculate average decay rates for each segment
            segmented_track_decay_rates_df = calculate_track_decay_rates_by_segment(adjusted_track_decay_df, track_lifecycle_segment_boundaries)

            # ===== 7. GENERATE STREAM FORECASTS =====
            track_streams_forecast = forecast_track_streams(segmented_track_decay_rates_df, track_streams_last_30days, song_data['months_since_release_total'], DEFAULT_FORECAST_PERIODS)

            # Convert forecasts to a DataFrame - the column contains streapredictions for each future month
            track_streams_forecast_df = pd.DataFrame(track_streams_forecast)
            track_streams_forecast_df2 = track_streams_forecast_df.copy()  # Create a copy for export
            track_streams_forecast_df2['track_name'] = selected_song #add track name to the dataframe for export
            export_track_streams_forecast = pd.concat([export_track_streams_forecast, track_streams_forecast_df2], ignore_index=True)
            
            # Calculate the total predicted streams for the forecast period
            track_valuation_months = DEFAULT_FORECAST_YEARS * 12
            total_track_streams_forecast = track_streams_forecast_df.loc[:track_valuation_months, 'predicted_streams_for_month'].sum()

            # ===== 8. CALCULATE HISTORICAL VALUE =====
            # Determine the end date for royalty rate calculations
            # For older tracks, use the valuation cutoff date
            # For newer tracks, use the latest available data
            
            # Convert date from "DD/MM/YYYY" to "YYYY-MM" format for mechanical royalty rate comparison
            earliest_track_date_formatted = datetime.strptime(song_data['earliest_track_date'], "%d/%m/%Y").strftime('%Y-%m')
            
            if earliest_track_date_formatted >= HISTORICAL_VALUATION_CUTOFF:
                royalty_calculation_end_date = mechanical_royalty_rates_df['Date'].max()
            else:
                royalty_calculation_end_date = HISTORICAL_VALUATION_CUTOFF
                
            # Filter mechanical royalty data for relevant date range
            # Note: MECHv2_fixed.csv dates are already in 'YYYY-MM' format, so no conversion needed
            mask = (mechanical_royalty_rates_df['Date'] >= earliest_track_date_formatted) & (mechanical_royalty_rates_df['Date'] <= royalty_calculation_end_date)
            
            # Calculate historical royalty value using our dedicated function
            historical_royalty_value_time_adjusted = calculate_historical_royalty_revenue(
                total_historical_track_streams=total_historical_track_streams,
                mechanical_royalty_rates_df=mechanical_royalty_rates_df,
                date_range_mask=mask,
                listener_percentage_usa=listener_percentage_usa,
                discount_rate=discount_rate,
                historical_value_time_adjustment=HISTORICAL_VALUE_TIME_ADJUSTMENT,
                premium_stream_percentage=PREMIUM_STREAM_PERCENTAGE,
                ad_supported_stream_percentage=AD_SUPPORTED_STREAM_PERCENTAGE
            )

            # ===== 9. PREPARE MONTHLY FORECAST DATA =====
            # Use utility function to create projections DataFrame and apply geographic distribution
            monthly_track_revenue_projections_df = create_monthly_track_revenue_projections(
                track_name=selected_song,
                track_streams_forecast_df=track_streams_forecast_df,
                listener_geography_df=listener_geography_df,
                worldwide_royalty_rates_df=worldwide_royalty_rates_df,
                discount_rate=discount_rate
            )
            
            # Calculate total discounted and non-discounted values
            discounted_future_royalty_value = monthly_track_revenue_projections_df['DISC'].sum()
            undiscounted_future_royalty_value = monthly_track_revenue_projections_df['Total'].sum()
            total_track_valuation = discounted_future_royalty_value + historical_royalty_value_time_adjusted

            # ===== 12. STORE FORECAST SUMMARY =====
            track_valuation_summaries.append({
                'track_name': selected_song,
                'total_historical_track_streams': total_historical_track_streams,
                'total_track_streams_forecast': total_track_streams_forecast,
                'historical_royalty_value_time_adjusted': historical_royalty_value_time_adjusted,
                'undiscounted_future_royalty_value': undiscounted_future_royalty_value,
                'discounted_future_royalty_value': discounted_future_royalty_value,
                'total_track_valuation': total_track_valuation,
            })

            # ===== 13. AGGREGATE MONTHLY DATA INTO YEARLY PERIODS =====
            # Use utility function to aggregate monthly projections into yearly periods
            yearly_track_revenue_df = aggregate_into_yearly_periods(monthly_track_revenue_projections_df)
            
            # Store yearly aggregated data for plotting
            track_yearly_revenue_collection.append(yearly_track_revenue_df)

        # ===== 14. DATA EXPORT AND AGGREGATION =====
        # Create downloadable CSV file of track stream forecasts
        csv = export_track_streams_forecast.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        download_link = f'<a href="data:file/csv;base64,{b64}" download="track_streams_forecast.csv">Download Track Streams Forecast</a>'
        st.markdown(download_link, unsafe_allow_html=True)
        
        # Combine track revenue data across all tracks and summarize by year
        yearly_revenue_combined_df = pd.concat(track_yearly_revenue_collection)
        yearly_total_by_year_df = yearly_revenue_combined_df.groupby('Year')['DISC'].sum().reset_index()
        
        # Convert track valuation summaries to DataFrame for display
        track_valuation_results_df = pd.DataFrame(track_valuation_summaries)

        # ===== 15. OWNERSHIP ADJUSTMENTS =====
        # Apply ownership adjustments to valuation results
        ownership_adjusted_valuation_df = apply_ownership_adjustments(track_valuation_results_df, ownership_df)
        
        # ===== 16. DISPLAY FORMATTING =====
        # Format and display valuation results
        final_valuation_display_df = display_valuation_results(ownership_adjusted_valuation_df)

        # ===== 17. SUMMARY STATISTICS =====
        # Calculate and display summary statistics across all tracks in the catalog
        catalog_valuation_summary_df = display_valuation_summary(final_valuation_display_df)

        # ===== 18. GEOGRAPHIC DISTRIBUTION ANALYSIS =====
        # Process country-specific revenue data and get top countries
        df_country_breakdown = process_country_breakdown(listener_geography_df, monthly_track_revenue_projections_df)
        top_countries, top_10_percentage_sum = get_top_countries(df_country_breakdown)
        
        # ===== 19. VISUALIZATION: TOP COUNTRIES =====
        # Create and display country distribution chart
        fig, ax = create_country_distribution_chart(top_countries, top_10_percentage_sum)
        st.pyplot(fig)

        # ===== 20. VISUALIZATION: YEARLY INCOME =====
        # Create and display yearly revenue chart
        fig, ax = create_yearly_revenue_chart(yearly_total_by_year_df)
        st.pyplot(fig)

        # ===== 21. FRAUD DETECTION =====
        # Detect potential streaming fraud
        alert_countries = detect_streaming_fraud(listener_geography_df, population_df)
        
        # Display fraud alerts if any are detected
        if alert_countries:
            st.write("Fraud Alert. This artist has unusually high streams from these countries:")
            for country in alert_countries:
                st.write(country)

