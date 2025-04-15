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
    DEFAULT_FORECAST_PERIODS
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
    forecast_values,
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
from utils.ui_functions import display_track_selection_ui, display_financial_parameters_ui

# Import financial parameters
from utils.financial_parameters import (
    PREMIUM_STREAM_PERCENTAGE, 
    AD_SUPPORTED_STREAM_PERCENTAGE,
    HISTORICAL_VALUE_TIME_ADJUSTMENT
)

# ===== MODELING FUNCTIONS =====

# ===== DATA LOADING - GLOBAL DATASETS =====
# Load country population data for analyzing geographic streaming patterns
# and detecting potential anomalies (e.g., streams exceeding realistic population penetration)
population_df = get_population_data()

# Load mechanical royalties data - contains historical royalty rates for Spotify streams
# Used to calculate historical value and forecast financial projections
df_additional = get_mech_data()
if df_additional is None:
    st.error("Failed to load mechanical royalties data")
    st.stop()

# Load worldwide rates data - contains country-specific royalty rates
# Used to calculate revenue projections based on geographic streaming distribution
GLOBAL = get_rates_data()
if GLOBAL is None:
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
        df = pd.read_csv(uploaded_file)
        
        # 2. DATA VALIDATION
        # Check if required columns exist
        if validate_columns(df, ['Date', 'Monthly Listeners']):
            # 3. DATA SELECTION
            # Keep only required columns
            columns_to_keep = ['Date', 'Monthly Listeners']
            df = select_columns(df, columns_to_keep)
            
            # 4. DATE CONVERSION
            # Convert 'Date' column to datetime format with error handling
            df, date_issues = convert_to_datetime(df, 'Date', dayfirst=True)
            
            # Display any issues with date conversion
            for issue in date_issues:
                if "Failed to convert" in issue:
                    st.error(issue)
                else:
                    st.warning(f"{issue} Please check your data.")
            
           
            # 5. INITIAL DECAY ANALYSIS
            # Calculate decay rates and get min/max dates for the UI slider
            initial_results = analyze_listener_decay(df)

            # ===== UI COMPONENTS SECTION =====
            min_date = initial_results['min_date']
            max_date = initial_results['max_date']
            
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

            if start_date and end_date:
                # ===== CORE ANALYSIS SECTION =====
                # 7. RUN DECAY RATE ANALYSIS WITH SELECTED DATE RANGE
                results = analyze_listener_decay(df, start_date, end_date)
                subset_df = results['subset_df']
                mldr = results['mldr']
                popt = results['popt']
                
                # ===== RESULTS DISPLAY SECTION =====
                # 8. SHOW METRICS
                st.write(f'Exponential decay rate: {mldr}')
                
                # 9. VISUALIZATION
                fig, ax = plt.subplots(figsize=(10, 4))
                # Plot the moving average
                ax.plot(subset_df['Date'], subset_df['4_Week_MA'], label='Moving Average', color='tab:blue', linewidth=2)
                # Plot the fitted decay curve using pre-calculated parameters
                ax.plot(subset_df['Date'], exponential_decay(subset_df['Months'], *popt), 
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

    # 3. DATA PROCESSING SECTION
    # Only process data if track files have been uploaded
    if uploaded_files_unique:
        # ===== TRACK DATA PROCESSING =====
        # Initialize an empty DataFrame that will store data for all tracks (our catalog)
        track_catalog_df = pd.DataFrame()
        
        # Process each uploaded track file - iterate through all CSV files the user uploaded
        for file_unique in uploaded_files_unique:
            # Extract track name from the filename (expected format: "Artist - TrackName.csv")
            track_name_unique = file_unique.name.split(' - ')[1].strip()
            
            # Read the track's streaming data from CSV
            df_track_data_unique = pd.read_csv(file_unique)
            
            # Make column names more descriptive - 'Value' becomes 'CumulativeStreams'
            df_track_data_unique = rename_columns(df_track_data_unique, {'Value': 'CumulativeStreams'})

            # Get the first date from the data (not necessarily release date, just first tracking date)
            data_start_date = extract_earliest_date(df_track_data_unique, 'Date')

            # Extract the latest (most recent) cumulative stream count
            total_track_streams = df_track_data_unique['CumulativeStreams'].iloc[-1]

            # Calculate period-specific stream counts using our utility function
            # These represent streams in the last 30/90/365 days
            track_streams_last_30days = calculate_period_streams(df_track_data_unique, 'CumulativeStreams', 30)
            track_streams_last_90days = calculate_period_streams(df_track_data_unique, 'CumulativeStreams', 90)
            track_streams_last_365days = calculate_period_streams(df_track_data_unique, 'CumulativeStreams', 365)
            
            # Pre-calculate time-based metrics to avoid recalculating after Run All button
            months_since_release_total = calculate_months_since_release(data_start_date)
            
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
                'data_start_date': [data_start_date],
                'track_streams_last_30days': [track_streams_last_30days],
                'track_streams_last_90days': [track_streams_last_90days],
                'track_streams_last_365days': [track_streams_last_365days],
                'total_track_streams': [total_track_streams],
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
        ownership_df = process_ownership_data(uploaded_file_ownership, track_catalog_df['track_name'])
        
        # ===== UI DISPLAY AND TRACK SELECTION =====
        selected_songs = display_track_selection_ui(track_catalog_df)
        
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
            years_plot = []
            export_forecasts = pd.DataFrame()
            stream_forecasts = []
            weights_and_changes = []

            # Process each selected song
            for selected_song in selected_songs:
                # ===== 1. EXTRACT SONG DATA =====
                song_data = track_catalog_df[track_catalog_df['track_name'] == selected_song].iloc[0]
                track_streams_last_365days = song_data['track_streams_last_365days']
                track_streams_last_90days = song_data['track_streams_last_90days']
                track_streams_last_30days = song_data['track_streams_last_30days']
                historical = song_data['total_track_streams']
                data_start_date = song_data['data_start_date']

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
                
                # Store track adjustment metrics for reporting and quality analysis
                track_adjustment_weight = track_adjustment_info['first_adjustment_weight']
                track_average_percent_change = track_adjustment_info['first_average_percent_change']
                
                # ===== 6. SEGMENT DECAY RATES BY TIME PERIOD =====
                # Calculate average decay rates for each segment
                segmented_track_decay_rates_df = calculate_track_decay_rates_by_segment(adjusted_track_decay_df, track_lifecycle_segment_boundaries)

                # ===== 7. GENERATE STREAM FORECASTS =====
                initial_value = track_streams_last_30days
                # Retrieve months since release for forecast period calculation
                start_period = song_data['months_since_release_total']

                forecasts = forecast_values(segmented_track_decay_rates_df, initial_value, start_period, DEFAULT_FORECAST_PERIODS)

                # Convert forecasts to a DataFrame
                forecasts_df = pd.DataFrame(forecasts)
                forecasts_df2 = forecasts_df.copy()  # Create a copy for export
                forecasts_df2['track_name'] = selected_song
                export_forecasts = pd.concat([export_forecasts, forecasts_df2], ignore_index=True)
                
                # Calculate the total forecast value for the first 240 months (20 years)
                total_forecast_value = forecasts_df.loc[:240, 'forecasted_value'].sum()

                # ===== 8. CALCULATE HISTORICAL VALUE =====
                tracking_start_date = datetime.strptime(data_start_date, "%d/%m/%Y")
                start_date = tracking_start_date.strftime('%Y-%m')
                end_date = '2024-02'  # Default end date
                
                # Adjust end date if tracking date is more recent
                if tracking_start_date.strftime('%Y-%m') >= end_date:
                    end_date = df_additional['Date'].max()
                    
                # Filter mechanical royalty data for relevant date range
                # Note: MECHv2_fixed.csv dates are already in 'YYYY-MM' format, so no conversion needed
                mask = (df_additional['Date'] >= start_date) & (df_additional['Date'] <= end_date)
                
                # Calculate historical value from streams
                ad_supported = df_additional.loc[mask, 'Spotify_Ad-supported'].mean()
                premium = df_additional.loc[mask, 'Spotify_Premium'].mean()
                hist_ad = AD_SUPPORTED_STREAM_PERCENTAGE * historical * ad_supported
                hist_prem = PREMIUM_STREAM_PERCENTAGE * historical * premium
                hist_value = (hist_ad + hist_prem) * (listener_percentage_usa)
                hist_value = hist_value / ((1 + discount_rate / 12) ** HISTORICAL_VALUE_TIME_ADJUSTMENT)  # Apply time value discount

                # ===== 9. PREPARE MONTHLY FORECAST DATA =====
                monthly_forecasts_df = pd.DataFrame({
                    'track_name': [selected_song] * len(forecasts_df),
                    'month': forecasts_df['month'],
                    'forecasted_value': forecasts_df['forecasted_value']
                })

                # Add month index for time-based calculations
                monthly_forecasts_df['month_index'] = monthly_forecasts_df.index + 1
                
                # ===== 10. APPLY GEOGRAPHIC DISTRIBUTION =====
                # Add country percentage distributions
                for index, row in listener_geography_df.iterrows():
                    country = row['Country']
                    percentage = row['Spotify monthly listeners (%)']
                    monthly_forecasts_df[country + ' %'] = percentage

                # Get country-specific royalty rates
                for index, row in listener_geography_df.iterrows():
                    country = row['Country']
                    if country in GLOBAL.columns:
                        mean_final_5 = GLOBAL[country].dropna().tail(5).mean()
                        monthly_forecasts_df[country + ' Royalty Rate'] = mean_final_5

                # Calculate country-specific stream values
                for index, row in listener_geography_df.iterrows():
                    country = row['Country']
                    monthly_forecasts_df[country + ' Value'] = monthly_forecasts_df['forecasted_value'] * monthly_forecasts_df[country + ' %']

                # Calculate country-specific royalty values
                for index, row in listener_geography_df.iterrows():
                    country = row['Country']
                    monthly_forecasts_df[country + ' Royalty Value'] = monthly_forecasts_df[country + ' Value'] * monthly_forecasts_df[country + ' Royalty Rate']

                # Clean up intermediate calculation columns
                percentage_columns = [country + ' %' for country in listener_geography_df['Country']]
                monthly_forecasts_df.drop(columns=percentage_columns, inplace=True)
                
                columns_to_drop = [country + ' Value' for country in listener_geography_df['Country']] + [country + ' Royalty Rate' for country in listener_geography_df['Country']]
                monthly_forecasts_df.drop(columns=columns_to_drop, inplace=True)
                
                # ===== 11. CALCULATE TOTAL FORECAST VALUE =====
                # Sum all country royalty values
                monthly_forecasts_df['Total'] = monthly_forecasts_df[[country + ' Royalty Value' for country in listener_geography_df['Country']]].sum(axis=1)
                
                # Apply time value of money discount
                monthly_forecasts_df['DISC'] = (monthly_forecasts_df['Total']) / ((1 + discount_rate / 12) ** (monthly_forecasts_df['month_index'] + 2.5))
                
                # Calculate total discounted and non-discounted values
                new_forecast_value = monthly_forecasts_df['DISC'].sum()
                forecast_OG = monthly_forecasts_df['Total'].sum()
                Total_Value = new_forecast_value + hist_value

                # ===== 12. STORE FORECAST SUMMARY =====
                stream_forecasts.append({
                    'track_name': selected_song,
                    'historical_streams': historical,
                    'forecast_streams': total_forecast_value,
                    'hist_value': hist_value,
                    'forecast_no_disc': forecast_OG,
                    'forecast_disc': new_forecast_value,
                    'total_value': Total_Value,
                })

                weights_and_changes.append({
                    'track_name': selected_song,
                    'weight': track_adjustment_weight,
                    'average_percent_change': track_average_percent_change
                })

                # ===== 13. AGGREGATE MONTHLY DATA INTO YEARLY PERIODS =====
                rows_per_period = 12
                n_rows = len(monthly_forecasts_df)
                
                # Initialize period pattern for aggregation
                period_pattern = []
                
                # First year (9 months)
                period_pattern.extend([1] * 9)
                
                # Calculate remaining rows after first 9 months
                remaining_rows = n_rows - 9
                
                # Assign remaining months to yearly periods (12 months per year)
                for period in range(2, (remaining_rows // rows_per_period) + 2):
                    period_pattern.extend([period] * rows_per_period)

                # Ensure pattern length matches dataframe rows
                if len(period_pattern) > n_rows:
                    period_pattern = period_pattern[:n_rows]  # Trim if too long
                else:
                    period_pattern.extend([period] * (n_rows - len(period_pattern)))  # Extend if too short

                # Assign periods to months
                monthly_forecasts_df['Period'] = period_pattern

                # Group data by period and aggregate
                aggregated_df = monthly_forecasts_df.groupby('Period').agg({
                    'track_name': 'first',
                    'month': 'first',  # First month in each period
                    'DISC': 'sum'      # Sum discounted values
                }).reset_index(drop=True)

                # Rename for clarity
                aggregated_df.rename(columns={'month': 'Start_Month'}, inplace=True)

                # Keep only first 10 years
                aggregated_df = aggregated_df.head(10)

                # Replace month numbers with year numbers
                aggregated_df['Year'] = range(1, 11)
                aggregated_df.drop(columns=['Start_Month'], inplace=True)

                # Store yearly aggregated data for plotting
                years_plot.append(aggregated_df)

            # ===== 14. DATA EXPORT AND AGGREGATION =====
            # Prepare forecast data for download
            catalog_to_download = export_forecasts
            csv = catalog_to_download.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="export_forecasts.csv">Download forecasts DataFrame</a>'
            st.markdown(href, unsafe_allow_html=True)
            
            # Combine yearly data and calculate annual totals
            years_plot_df = pd.concat(years_plot)
            yearly_disc_sum_df = years_plot_df.groupby('Year')['DISC'].sum().reset_index()
            df_forecasts = pd.DataFrame(stream_forecasts)

            # ===== 15. OWNERSHIP ADJUSTMENTS =====
            # Merge forecast data with ownership information
            merged_df = df_forecasts.merge(ownership_df[['track_name', 'MLC Claimed(%)', 'Ownership(%)']], on='track_name', how='left')

            # Ensure ownership percentages are properly formatted
            merged_df['MLC Claimed(%)'] = pd.to_numeric(merged_df['MLC Claimed(%)'], errors='coerce').fillna(0)
            merged_df['Ownership(%)'] = pd.to_numeric(merged_df['Ownership(%)'], errors='coerce').fillna(1)
            
            # Adjust historical value based on MLC claims and ownership percentage
            merged_df['hist_value'] = merged_df.apply(
                lambda row: min((1 - row['MLC Claimed(%)']) * row['hist_value'], row['Ownership(%)'] * row['hist_value']),
                axis=1
            )
            
            # Adjust forecast values based on ownership percentage
            merged_df['forecast_no_disc'] = merged_df['forecast_no_disc'].astype(float) * (merged_df['Ownership(%)'])
            merged_df['forecast_disc'] = merged_df['forecast_disc'].astype(float) * (merged_df['Ownership(%)'])
            merged_df['total_value'] = merged_df['forecast_disc'] + merged_df['hist_value']
            merged_df = merged_df.drop(columns=['Ownership(%)', 'MLC Claimed(%)'])
            
            # ===== 16. DISPLAY FORMATTING =====
            # Format values for presentation with commas and currency symbols
            df_forecasts = merged_df
            
            df_forecasts['historical_streams'] = df_forecasts['historical_streams'].astype(float).apply(lambda x: f"{int(round(x)):,}")
            df_forecasts['forecast_streams'] = df_forecasts['forecast_streams'].astype(float).apply(lambda x: f"{int(round(x)):,}")
            df_forecasts['forecast_no_disc'] = df_forecasts['forecast_no_disc'].astype(float).apply(lambda x: f"${int(round(x)):,}")
            df_forecasts['forecast_disc'] = df_forecasts['forecast_disc'].astype(float).apply(lambda x: f"${int(round(x)):,}")
            df_forecasts['hist_value'] = df_forecasts['hist_value'].astype(float).apply(lambda x: f"${int(round(x)):,}")
            df_forecasts['total_value'] = df_forecasts['total_value'].astype(float).apply(lambda x: f"${int(round(x)):,}")

            # Display the formatted forecast table
            st.write(df_forecasts)

            # ===== 17. SUMMARY STATISTICS =====
            # Calculate summary totals across all tracks
            sum_df = pd.DataFrame({
                'Metric': ['hist_value', 'forecast_OG','forecast_dis', 'total_value'],
                'Sum': [
                    df_forecasts['hist_value'].apply(lambda x: int(x.replace('$', '').replace(',', ''))).sum(),
                    df_forecasts['forecast_no_disc'].apply(lambda x: int(x.replace('$', '').replace(',', ''))).sum(),
                    df_forecasts['forecast_disc'].apply(lambda x: int(x.replace('$', '').replace(',', ''))).sum(),
                    df_forecasts['total_value'].apply(lambda x: int(x.replace('$', '').replace(',', ''))).sum()
                ]
            })

            # Format summary values for display
            sum_df['Sum'] = sum_df['Sum'].apply(lambda x: f"${int(round(x)):,}")

            # Display the summary table
            st.write("Summed Values:")
            st.write(sum_df)

            # ===== 18. GEOGRAPHIC DISTRIBUTION ANALYSIS =====
            # Extract country-specific revenue data
            country_breakdown = []
            for index, row in listener_geography_df.iterrows():
                country = row['Country']
                forecast_no_disc_value = monthly_forecasts_df[country + ' Royalty Value'].sum() 
                country_breakdown.append({
                    'Country': country,
                    'forecast_no_disc': forecast_no_disc_value
                })
                
            # Process country breakdown data
            df_country_breakdown = pd.DataFrame(country_breakdown)
            df_country_breakdown['forecast_no_disc_numeric'] = df_country_breakdown['forecast_no_disc'].replace({'\$': '', ',': ''}, regex=True).astype(float)

            # Calculate total forecast value and country percentages
            total_forecast_no_disc_value = df_country_breakdown['forecast_no_disc_numeric'].sum()
            df_country_breakdown['Percentage'] = (df_country_breakdown['forecast_no_disc_numeric'] / total_forecast_no_disc_value) * 100
            
            # Get top countries by revenue contribution
            top_countries = df_country_breakdown.sort_values(by='forecast_no_disc_numeric', ascending=False).head(10)
            top_10_percentage_sum = top_countries['Percentage'].sum()
            
            # ===== 19. VISUALIZATION: TOP COUNTRIES =====
            # Create horizontal bar chart for top revenue countries
            fig, ax = plt.subplots()
            bar_color = 'teal'
            bars = ax.barh(top_countries['Country'], top_countries['forecast_no_disc_numeric'], color=bar_color)

            # Configure chart appearance
            ax.set_xlabel('% of Forecast Value')
            ax.set_title(f'Top 10 Countries Contribute {top_10_percentage_sum:.1f}% to Total Forecast Value')
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${int(x):,}"))
            max_value = top_countries['forecast_no_disc_numeric'].max()
            ax.set_xlim(0, max_value * 1.25)
            ax.set_xticks([])
            
            # Add percentage labels to bars
            for bar, percentage in zip(bars, top_countries['Percentage']):
                width = bar.get_width()
                ax.text(width + (width * 0.01), bar.get_y() + bar.get_height() / 2, 
                        f'{percentage:.1f}%', va='center', ha='left', 
                        fontsize=10, color='black')

            # Display the country distribution chart
            st.pyplot(fig)

            # ===== 20. VISUALIZATION: YEARLY INCOME =====
            # Create bar chart for yearly revenue projection
            fig, ax = plt.subplots()
            bar_color = 'teal'
            bars = ax.bar(yearly_disc_sum_df['Year'], yearly_disc_sum_df['DISC'], color=bar_color)

            # Configure chart appearance
            ax.set_xlabel('Year')
            ax.set_title('Income by Year (discounted)')
            ax.set_ylabel('')
            ax.yaxis.set_visible(False)
            max_value = yearly_disc_sum_df['DISC'].max()
            ax.set_ylim(0, max_value * 1.25)

            # Add value labels to bars
            for bar, value in zip(bars, yearly_disc_sum_df['DISC']):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, height, f'${int(value)}', 
                        va='bottom', ha='center', fontsize=10, color='black')

            # Display the yearly income chart
            st.pyplot(fig)

            # ===== 21. FRAUD DETECTION =====
            # Cross-reference audience data with population data
            warning_df = pd.merge(listener_geography_df, population_df, on='Country', how='left')
            
            # Calculate threshold for suspicious activity (20% of population)
            warning_df['TwentyPercentPopulation'] = warning_df['Population'] * 0.20
            
            # Flag countries with abnormally high listener numbers
            warning_df['Above20Percent'] = warning_df['Spotify Monthly Listeners'] > warning_df['TwentyPercentPopulation']
            warning_df['Alert'] = warning_df['Above20Percent'].apply(lambda x: 1 if x else 0)
            
            # Get list of countries with potential streaming fraud
            alert_countries = warning_df[warning_df['Alert'] == 1]['Country']
            
            # Display fraud alerts if any are detected
            if not alert_countries.empty:
                st.write("Fraud Alert. This artist has unusually high streams from these countries:")
                for country in alert_countries:
                    st.write(country)

