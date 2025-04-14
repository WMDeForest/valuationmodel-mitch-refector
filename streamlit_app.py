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
    validate_columns
)
from utils.decay_rates import (
    ranges_sp,
    sp_range,
    SP_REACH_DATA,
    SP_REACH,
    fitted_params,
    fitted_params_df,
    breakpoints,
)

# Import decay model functions from the new modules
from utils.decay_models import (
    piecewise_exp_decay,
    exponential_decay,
    remove_anomalies,
    fit_decay_curve,
    fit_segment,
    update_fitted_params,
    forecast_values,
    analyze_listener_decay,
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
    # 1. FILE UPLOADS FOR ADDITIONAL DATA
    #uploaded_file = st.file_uploader("Tracklist", type=["csv"])
    uploaded_files_unique = st.file_uploader("Monthly Track Spotify Streams", type=['csv'], accept_multiple_files=True)
    uploaded_file_3 = st.file_uploader("Audience Geography", type=["csv"])
    uploaded_file_ownership = st.file_uploader("MLC Claimed and Song Ownership", type="csv")

    # 2. TRACK DATA PROCESSING
    track_summary_list = []

    if uploaded_files_unique:
        # Process each uploaded track file
        for file_unique in uploaded_files_unique:
            # Read the track data CSV
            df_track_data_unique = pd.read_csv(file_unique)

            # Extract release date from first row
            release_date_unique = pd.to_datetime(df_track_data_unique['Date'].iloc[0], format='%b %d, %Y').strftime('%d/%m/%Y')

            # Get total streams (most recent cumulative value)
            total_value_unique = df_track_data_unique['Value'].iloc[-1]

            # Calculate period-specific stream counts
            # Last 30 days streams 
            if len(df_track_data_unique) > 30:
                spotify_streams_1m_unique = total_value_unique - df_track_data_unique['Value'].iloc[-31]
            else:
                spotify_streams_1m_unique = total_value_unique
            
            # Last 90 days streams
            if len(df_track_data_unique) > 90:
                spotify_streams_3m_unique = total_value_unique - df_track_data_unique['Value'].iloc[-91]
            else:
                spotify_streams_3m_unique = total_value_unique
            
            # Last 12 months streams (last 365 days)
            if len(df_track_data_unique) > 365:
                spotify_streams_12m_unique = total_value_unique - df_track_data_unique['Value'].iloc[-366]
            else:
                spotify_streams_12m_unique = total_value_unique
            
            # Extract track name from filename
            track_name_unique = file_unique.name.split(' - ')[1].strip()

            # Add track data to summary list
            track_summary_list.append({
                'Track': track_name_unique,                      # Track Name
                'Release date': release_date_unique,             # Release Date
                'Spotify Streams 1m': spotify_streams_1m_unique, # Last 30 days Streams
                'Spotify Streams 3m': spotify_streams_3m_unique, # Last 90 Days Streams
                'Spotify Streams 12m': spotify_streams_12m_unique, # Last 365 Days Streams
                'Spotify Streams Total': total_value_unique      # Total Value
            })

        # 3. TRACK SUMMARY DATAFRAME CREATION
        track_summary_df_unique = pd.DataFrame(track_summary_list)
        df = track_summary_df_unique

        # 4. OWNERSHIP DATA PROCESSING
        if uploaded_file_ownership is not None:
            # Load ownership data with encoding handling
            try:
                ownership_df = pd.read_csv(uploaded_file_ownership, encoding='latin1')
            except UnicodeDecodeError:
                ownership_df = pd.read_csv(uploaded_file_ownership, encoding='utf-8')
        
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
                'Track': df['Track'],
                'Ownership(%)': [None] * len(df),
                'MLC Claimed(%)': [None] * len(df)
            })
        
        # 5. DATA TRANSFORMATION FOR ANALYSIS
        # Format dates for royalty calculations
        df_additional['Date'] = pd.to_datetime(df_additional['Date'], format='%b-%y')
        
        # Map column names to standardized internal names
        if 'Spotify Streams 1m' in df.columns:
            df['streams_last_month'] = df['Spotify Streams 1m']
        if 'Spotify Streams 3m' in df.columns:
            df['total_streams_3_months'] = df['Spotify Streams 3m']
        if 'Spotify Streams 12m' in df.columns:
            df['total_streams_12_months'] = df['Spotify Streams 12m']
        if 'Spotify Streams Total' in df.columns:
            df['historical'] = df['Spotify Streams Total']
        if 'Release date' in df.columns:
            df['release_date'] = df['Release date']

        # Clean up dataframe by removing original columns after mapping
        columns_to_drop = ["Release date", "Spotify Streams 1m", "Spotify Streams 3m", "Spotify Streams 12m", "Spotify Streams Total"]
        df.drop(columns=columns_to_drop, inplace=True)

        # 6. FORECAST PARAMETERS SETUP
        stream_influence_factor = 1000
        forecast_periods = 400
        current_date = datetime.today()

        # 7. USER INTERFACE FOR TRACK SELECTION
        st.write("Data Preview:")
        st.write(df)

        songs = sorted(df['Track'].unique(), key=lambda x: x.lower())
        selected_songs = st.multiselect('Select Songs', songs)

        # 8. AUDIENCE GEOGRAPHY PROCESSING
        if uploaded_file_3:
            audience_df = pd.read_csv(uploaded_file_3)

            # Extract and process geographical data
            audience_df = audience_df[['Country', 'Spotify Monthly Listeners']]
            audience_df = audience_df.groupby('Country', as_index=False)['Spotify Monthly Listeners'].sum()
            
            # Calculate percentage distribution
            total_listeners = audience_df['Spotify Monthly Listeners'].sum()
            audience_df['Spotify monthly listeners (%)'] = (audience_df['Spotify Monthly Listeners'] / total_listeners) * 100
            
            # Normalize percentage values
            audience_df["Spotify monthly listeners (%)"] = pd.to_numeric(audience_df["Spotify monthly listeners (%)"], errors='coerce')
            audience_df["Spotify monthly listeners (%)"] = audience_df["Spotify monthly listeners (%)"] / 100
            
            # Extract US percentage for royalty calculations
            percentage_usa = audience_df.loc[audience_df["Country"] == "United States", "Spotify monthly listeners (%)"].values[0]
            
        # 9. FINANCIAL PARAMETERS
        discount_rate = st.number_input('Discount Rate (%)', min_value=0.00, max_value=10.00, value=0.00, step=0.01, format="%.2f")/100

        # 10. INITIALIZE RESULTS STORAGE
        song_forecasts = []
        weights_and_changes = []

        if st.button('Run All'):
            # Initialize data structures to store results
            years_plot = []
            export_forecasts = pd.DataFrame()
            stream_forecasts = []  # Changed from song_forecasts to stream_forecasts
            weights_and_changes = []

            # Process each selected song
            for selected_song in selected_songs:
                # ===== 1. EXTRACT SONG DATA =====
                song_data = df[df['Track'] == selected_song].iloc[0]

                value = stream_influence_factor
                total_streams_12_months = song_data['total_streams_12_months']
                total_streams_3_months = song_data['total_streams_3_months']
                streams_last_month = song_data['streams_last_month']
                historical = song_data['historical']
                release_date = song_data['release_date']

                # ===== 2. UPDATE DECAY PARAMETERS =====
                updated_fitted_params_df = update_fitted_params(fitted_params_df, stream_influence_factor, sp_range, SP_REACH)
                if updated_fitted_params_df is not None:
                    updated_fitted_params = updated_fitted_params_df.to_dict(orient='records')

                # ===== 3. CALCULATE TIME SINCE RELEASE AND AVERAGE STREAMS =====
                release_date = datetime.strptime(release_date, "%d/%m/%Y")
                delta = current_date - release_date
                months_since_release_total = delta.days // 30
                
                # Calculate monthly averages for different time periods
                monthly_avg_3_months = (total_streams_3_months - streams_last_month) / (2 if months_since_release_total > 2 else 1)
                monthly_avg_last_month = streams_last_month

                if months_since_release_total > 3:
                    monthly_avg_12_months = (total_streams_12_months - total_streams_3_months) / (9 if months_since_release_total > 11 else (months_since_release_total - 3))
                else:
                    monthly_avg_12_months = monthly_avg_3_months

                # Prepare arrays for decay rate fitting
                months_since_release = np.array([
                    max((months_since_release_total - 11), 0),
                    max((months_since_release_total - 2), 0),
                    months_since_release_total - 0
                ])
                
                monthly_averages = np.array([monthly_avg_12_months, monthly_avg_3_months, monthly_avg_last_month])

                # ===== 4. FIT DECAY MODEL TO STREAM DATA =====
                params = fit_segment(months_since_release, monthly_averages)
                S0, k = params
                decay_rates_df = updated_fitted_params_df

                # Generate decay rates for all months
                months_since_release_all = list(range(1, 500))
                decay_rate_list = []

                for month in months_since_release_all:
                    for i in range(len(breakpoints) - 1):
                        if breakpoints[i] <= month < breakpoints[i + 1]:
                            decay_rate = decay_rates_df.loc[i, 'k']
                            decay_rate_list.append(decay_rate)
                            break

                # ===== 5. ADJUST DECAY RATES BASED ON OBSERVED DATA =====
                final_df = pd.DataFrame({
                    'months_since_release': months_since_release_all,
                    'decay_rate': decay_rate_list
                })

                # Apply measured decay rate to the observed time period
                start_month = min(months_since_release)
                end_month = max(months_since_release)
                final_df.loc[(final_df['months_since_release'] >= start_month) & 
                            (final_df['months_since_release'] <= end_month), 'mldr'] = mldr

                # Calculate percentage change between model and observed decay
                final_df['percent_change'] = ((final_df['mldr'] - final_df['decay_rate']) / final_df['decay_rate']) * 100
                average_percent_change = final_df['percent_change'].mean()
                
                # Apply weighting based on direction of change
                if average_percent_change > 0:
                    weight = min(1, max(0, average_percent_change / 100))
                else:
                    weight = 0
                    
                # First adjustment of decay rates
                final_df['adjusted_decay_rate'] = final_df['decay_rate'] * (1 + (average_percent_change * weight) / 100)
                
                # Apply fitted decay rate to observed period
                final_df.loc[(final_df['months_since_release'] >= start_month) & 
                            (final_df['months_since_release'] <= end_month), 'new_decay_rate'] = k

                # Compare adjusted decay rate with newly fitted rate
                final_df['percent_change_new_vs_adjusted'] = ((final_df['new_decay_rate'] - final_df['adjusted_decay_rate']) / final_df['adjusted_decay_rate']) * 100
                average_percent_change_new_vs_adjusted = final_df['percent_change_new_vs_adjusted'].mean()
                
                # Final adjustment of decay rates
                weight_new = 1 if average_percent_change_new_vs_adjusted > 0 else 0
                final_df['final_adjusted_decay_rate'] = final_df['adjusted_decay_rate'] * (1 + (average_percent_change_new_vs_adjusted * weight_new) / 100)
                
                # Clean up intermediate calculation columns
                final_df.drop(['decay_rate', 'mldr', 'percent_change'], axis=1, inplace=True)

                # ===== 6. SEGMENT DECAY RATES BY TIME PERIOD =====
                segments = []
                avg_decay_rates = []

                for i in range(len(breakpoints) - 1):
                    start = breakpoints[i]
                    end = breakpoints[i + 1] - 1
                    segment_data = final_df[(final_df['months_since_release'] >= start) & (final_df['months_since_release'] <= end)]
                    avg_decay_rate = segment_data['final_adjusted_decay_rate'].mean()
                    segments.append(i + 1)
                    avg_decay_rates.append(avg_decay_rate)

                consolidated_df = pd.DataFrame({
                    'segment': segments,
                    'k': avg_decay_rates
                })

                # ===== 7. GENERATE STREAM FORECASTS =====
                initial_value = streams_last_month
                start_period = months_since_release_total

                forecasts = forecast_values(consolidated_df, initial_value, start_period, forecast_periods)

                # Convert forecasts to a DataFrame
                forecasts_df = pd.DataFrame(forecasts)
                forecasts_df2 = forecasts_df.copy()  # Create a copy for export
                forecasts_df2['track_name_unique'] = selected_song
                export_forecasts = pd.concat([export_forecasts, forecasts_df2], ignore_index=True)
                
                # Calculate the total forecast value for the first 240 months (20 years)
                total_forecast_value = forecasts_df.loc[:240, 'forecasted_value'].sum()

                # ===== 8. CALCULATE HISTORICAL VALUE =====
                release_date = datetime.strptime(song_data['release_date'], "%d/%m/%Y")
                start_date = release_date.strftime('%Y-%m')
                end_date = '2024-02'  # Default end date
                
                # Adjust end date if release date is more recent
                if release_date.strftime('%Y-%m') >= end_date:
                    end_date = df_additional['Date'].max().strftime('%Y-%m')
                    
                # Filter mechanical royalty data for relevant date range
                mask = (df_additional['Date'] >= start_date) & (df_additional['Date'] <= end_date)
                
                # Calculate historical value from streams
                ad_supported = df_additional.loc[mask, 'Spotify_Ad-supported'].mean()
                premium = df_additional.loc[mask, 'Spotify_Premium'].mean()
                hist_ad = 0.6 * historical * ad_supported
                hist_prem = 0.4 * historical * premium
                hist_value = (hist_ad + hist_prem) * (percentage_usa)
                hist_value = hist_value / ((1 + discount_rate / 12) ** 3)  # Apply time value discount

                # ===== 9. PREPARE MONTHLY FORECAST DATA =====
                monthly_forecasts_df = pd.DataFrame({
                    'Track': [selected_song] * len(forecasts_df),
                    'Month': forecasts_df['month'],
                    'Forecasted Value': forecasts_df['forecasted_value']
                })

                # Add month index for time-based calculations
                monthly_forecasts_df['Month Index'] = monthly_forecasts_df.index + 1
                
                # ===== 10. APPLY GEOGRAPHIC DISTRIBUTION =====
                # Add country percentage distributions
                for index, row in audience_df.iterrows():
                    country = row['Country']
                    percentage = row['Spotify monthly listeners (%)']
                    monthly_forecasts_df[country + ' %'] = percentage

                # Get country-specific royalty rates
                for index, row in audience_df.iterrows():
                    country = row['Country']
                    if country in GLOBAL.columns:
                        mean_final_5 = GLOBAL[country].dropna().tail(5).mean()
                        monthly_forecasts_df[country + ' Royalty Rate'] = mean_final_5

                # Calculate country-specific stream values
                for index, row in audience_df.iterrows():
                    country = row['Country']
                    monthly_forecasts_df[country + ' Value'] = monthly_forecasts_df['Forecasted Value'] * monthly_forecasts_df[country + ' %']

                # Calculate country-specific royalty values
                for index, row in audience_df.iterrows():
                    country = row['Country']
                    monthly_forecasts_df[country + ' Royalty Value'] = monthly_forecasts_df[country + ' Value'] * monthly_forecasts_df[country + ' Royalty Rate']

                # Clean up intermediate calculation columns
                percentage_columns = [country + ' %' for country in audience_df['Country']]
                monthly_forecasts_df.drop(columns=percentage_columns, inplace=True)
                
                columns_to_drop = [country + ' Value' for country in audience_df['Country']] + [country + ' Royalty Rate' for country in audience_df['Country']]
                monthly_forecasts_df.drop(columns=columns_to_drop, inplace=True)
                
                # ===== 11. CALCULATE TOTAL FORECAST VALUE =====
                # Sum all country royalty values
                monthly_forecasts_df['Total'] = monthly_forecasts_df[[country + ' Royalty Value' for country in audience_df['Country']]].sum(axis=1)
                
                # Apply time value of money discount
                monthly_forecasts_df['DISC'] = (monthly_forecasts_df['Total']) / ((1 + discount_rate / 12) ** (monthly_forecasts_df['Month Index'] + 2.5))
                
                # Calculate total discounted and non-discounted values
                new_forecast_value = monthly_forecasts_df['DISC'].sum()
                forecast_OG = monthly_forecasts_df['Total'].sum()
                Total_Value = new_forecast_value + hist_value

                # ===== 12. STORE FORECAST SUMMARY =====
                song_forecasts.append({
                    'Track': selected_song,
                    'Historical': historical,
                    'Forecast': total_forecast_value,
                    'hist_value': hist_value,
                    'Forecast_no_disc': forecast_OG,
                    'Forecast_disc': new_forecast_value,
                    'Total_Value': Total_Value,
                })

                weights_and_changes.append({
                    'Track': selected_song,
                    'Weight': weight,
                    'Average Percent Change': average_percent_change
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
                    'Track': 'first',
                    'Month': 'first',  # First month in each period
                    'DISC': 'sum'      # Sum discounted values
                }).reset_index(drop=True)

                # Rename for clarity
                aggregated_df.rename(columns={'Month': 'Start_Month'}, inplace=True)

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
            df_forecasts = pd.DataFrame(song_forecasts)

            # ===== 15. OWNERSHIP ADJUSTMENTS =====
            # Merge forecast data with ownership information
            merged_df = df_forecasts.merge(ownership_df[['Track', 'MLC Claimed(%)', 'Ownership(%)']], on='Track', how='left')

            # Ensure ownership percentages are properly formatted
            merged_df['MLC Claimed(%)'] = pd.to_numeric(merged_df['MLC Claimed(%)'], errors='coerce').fillna(0)
            merged_df['Ownership(%)'] = pd.to_numeric(merged_df['Ownership(%)'], errors='coerce').fillna(1)
            
            # Adjust historical value based on MLC claims and ownership percentage
            merged_df['hist_value'] = merged_df.apply(
                lambda row: min((1 - row['MLC Claimed(%)']) * row['hist_value'], row['Ownership(%)'] * row['hist_value']),
                axis=1
            )
            
            # Adjust forecast values based on ownership percentage
            merged_df['Forecast_no_disc'] = merged_df['Forecast_no_disc'].astype(float) * (merged_df['Ownership(%)'])
            merged_df['Forecast_disc'] = merged_df['Forecast_disc'].astype(float) * (merged_df['Ownership(%)'])
            merged_df['Total_Value'] = merged_df['Forecast_disc'] + merged_df['hist_value']
            merged_df = merged_df.drop(columns=['Ownership(%)', 'MLC Claimed(%)'])
            
            # ===== 16. DISPLAY FORMATTING =====
            # Format values for presentation with commas and currency symbols
            df_forecasts = merged_df
            
            df_forecasts['Historical'] = df_forecasts['Historical'].astype(float).apply(lambda x: f"{int(round(x)):,}")
            df_forecasts['Forecast'] = df_forecasts['Forecast'].astype(float).apply(lambda x: f"{int(round(x)):,}")
            df_forecasts['Forecast_no_disc'] = df_forecasts['Forecast_no_disc'].astype(float).apply(lambda x: f"${int(round(x)):,}")
            df_forecasts['Forecast_disc'] = df_forecasts['Forecast_disc'].astype(float).apply(lambda x: f"${int(round(x)):,}")
            df_forecasts['hist_value'] = df_forecasts['hist_value'].astype(float).apply(lambda x: f"${int(round(x)):,}")
            df_forecasts['Total_Value'] = df_forecasts['Total_Value'].astype(float).apply(lambda x: f"${int(round(x)):,}")

            # Display the formatted forecast table
            st.write(df_forecasts)

            # ===== 17. SUMMARY STATISTICS =====
            # Calculate summary totals across all tracks
            sum_df = pd.DataFrame({
                'Metric': ['hist_value', 'Forecast_OG','Forecast_dis', 'Total_Value'],
                'Sum': [
                    df_forecasts['hist_value'].apply(lambda x: int(x.replace('$', '').replace(',', ''))).sum(),
                    df_forecasts['Forecast_no_disc'].apply(lambda x: int(x.replace('$', '').replace(',', ''))).sum(),
                    df_forecasts['Forecast_disc'].apply(lambda x: int(x.replace('$', '').replace(',', ''))).sum(),
                    df_forecasts['Total_Value'].apply(lambda x: int(x.replace('$', '').replace(',', ''))).sum()
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
            for index, row in audience_df.iterrows():
                country = row['Country']
                forecast_no_disc_value = monthly_forecasts_df[country + ' Royalty Value'].sum() 
                country_breakdown.append({
                    'Country': country,
                    'Forecast_no_disc': forecast_no_disc_value
                })
                
            # Process country breakdown data
            df_country_breakdown = pd.DataFrame(country_breakdown)
            df_country_breakdown['Forecast_no_disc_numeric'] = df_country_breakdown['Forecast_no_disc'].replace({'\$': '', ',': ''}, regex=True).astype(float)

            # Calculate total forecast value and country percentages
            total_forecast_no_disc_value = df_country_breakdown['Forecast_no_disc_numeric'].sum()
            df_country_breakdown['Percentage'] = (df_country_breakdown['Forecast_no_disc_numeric'] / total_forecast_no_disc_value) * 100
            
            # Get top countries by revenue contribution
            top_countries = df_country_breakdown.sort_values(by='Forecast_no_disc_numeric', ascending=False).head(10)
            top_10_percentage_sum = top_countries['Percentage'].sum()
            
            # ===== 19. VISUALIZATION: TOP COUNTRIES =====
            # Create horizontal bar chart for top revenue countries
            fig, ax = plt.subplots()
            bar_color = 'teal'
            bars = ax.barh(top_countries['Country'], top_countries['Forecast_no_disc_numeric'], color=bar_color)

            # Configure chart appearance
            ax.set_xlabel('% of Forecast Value')
            ax.set_title(f'Top 10 Countries Contribute {top_10_percentage_sum:.1f}% to Total Forecast Value')
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${int(x):,}"))
            max_value = top_countries['Forecast_no_disc_numeric'].max()
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
            warning_df = pd.merge(audience_df, population_df, on='Country', how='left')
            
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

