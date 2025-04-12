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
    calculate_decay_rate,
    fit_segment,
    update_fitted_params,
    forecast_values,
)

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

    uploaded_file = st.file_uploader("Spotify streams - Weekly - 24 Months", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        ##UUPP1##
        
            # Convert 'Date' column to datetime format (assuming the 'Date' column exists)
        if 'Date' in df.columns:
            # Try converting 'Date' column to datetime format with error handling
            try:
                df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
            except Exception as e:
                st.error(f"Failed to convert Date column: {e}")

            # Check for any NaT values which might indicate parsing errors
            if df['Date'].isna().any():
                st.warning("Some dates couldn't be parsed and have been set to 'NaT'. Please check your data.")

        
        # Keep only the 'Date' and 'Monthly Listeners' columns, renaming 'Monthly Listeners' to 'Streams'
        df = df[['Date', 'Monthly Listeners']].rename(columns={'Monthly Listeners': 'Streams'})
        
        # Keep every 7th row
        df = df.iloc[::7, :]
        
        # Optionally, you can format the date to be in 'DD/MM/YYYY' format if needed
        df['Date'] = df['Date'].dt.strftime('%d/%m/%Y')
        ##UUPP1##
        
        
        # Ensure the data has the expected columns
        if 'Date' in df.columns and 'Streams' in df.columns:
            
            if 'Date' in df.columns:
                # Try converting 'Date' column to datetime format with error handling
                try:
                    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
                except Exception as e:
                    st.error(f"Failed to convert Date column: {e}")
        
                # Check for any NaT values which might indicate parsing errors
                if df['Date'].isna().any():
                    st.warning("Some dates couldn't be parsed and have been set to 'NaT'. Please check your data.")
            df = df.sort_values(by='Date')

            # Remove anomalies
            monthly_data = remove_anomalies(df)

            # Determine the range of dates
            min_date = monthly_data['Date'].min().to_pydatetime()  # Convert to datetime
            max_date = monthly_data['Date'].max().to_pydatetime()  # Convert to datetime

            # Create a slider for date range selection
            st.write("Select Date Range:")
            start_date, end_date = st.slider(
                "Select date range",
                min_value=min_date,
                max_value=max_date,
                value=(min_date, max_date),
                format="YYYY-MM-DD"
            )

            # Convert slider values back to Timestamp
            start_date = pd.Timestamp(start_date)
            end_date = pd.Timestamp(end_date)

            if start_date and end_date:
                # Filter the data based on the selected date range
                mask = (monthly_data['Date'] >= start_date) & (monthly_data['Date'] <= end_date)
                subset_df = monthly_data[mask]

                # Add Months column for the filtered subset
                subset_df['Months'] = subset_df['Date'].apply(lambda x: (x.year - min_date.year) * 12 + x.month - min_date.month)

                # Calculate the decay rate
                mldr, popt = calculate_decay_rate(subset_df)
                st.write(f'Exponential decay ratemldr: {mldr}')

                # Plot the data and the fitted curve
                fig, ax = plt.subplots(figsize=(10, 4))  # Half the height
                ax.plot(subset_df['Date'], subset_df['4_Week_MA'], label='Moving Average', color='tab:blue', linewidth=2)
                ax.plot(subset_df['Date'], exponential_decay(subset_df['Months'], *popt), label='Fitted Decay Curve', color='red', linestyle='--')
                ax.set_xlabel('Date', fontsize=12)
                ax.set_ylabel('Streams', fontsize=12)
                ax.set_title(f'Moving Average and Exponential Decay', fontsize=14, weight='bold')
                ax.legend()
                ax.set_ylim(bottom=0)
                plt.xticks(rotation=45)
                
                # Remove background color
                fig.patch.set_visible(False)
                ax.set_facecolor('none')  # Transparent background for the plot area
                ax.patch.set_alpha(0)     # Remove background color of the axes
                plt.tight_layout()        # Adjust plot to fit labels
                st.pyplot(fig)
        else:
            st.error("The uploaded file does not contain the required columns 'Date' and 'Streams'.")
            
        
    #uploaded_file = st.file_uploader("Tracklist", type=["csv"])
    uploaded_files_unique = st.file_uploader("Upload multiple CSV files for track data", type=['csv'], accept_multiple_files=True)
    #uploaded_file_additional = st.file_uploader("Mechanical Royalties USA", type=["csv"])
    uploaded_file_3 = st.file_uploader("Audience Geography", type=["csv"])
    #uploaded_file_global = st.file_uploader("Worldwide Mechanical Royalties", type="csv")
    uploaded_file_ownership = st.file_uploader("MLC Claimed and Song Ownership", type="csv")

    #if uploaded_file_global is not None:
    #    try:
    #        # Attempt to read the file using 'latin1' encoding
    #        GLOBAL = pd.read_csv(uploaded_file_global, encoding='latin1')
    #        st.success("")
    #    except Exception as e:
    #        st.error(f"Failed to load the file: {e}")
    #else:
    #    st.stop()


    #if uploaded_file is not None:
    #    df = pd.read_csv(uploaded_file, usecols=["Track", "Release date", "Spotify Streams 1m", "Spotify Streams 3m", "Spotify Streams 12m", "Spotify Streams Total"])

    track_summary_list = []

    if uploaded_files_unique:
        for file_unique in uploaded_files_unique:
            # Read each CSV file
            df_track_data_unique = pd.read_csv(file_unique)

            # Extract the release date (first value in the 'Date' column)
            release_date_unique = pd.to_datetime(df_track_data_unique['Date'].iloc[0], format='%b %d, %Y').strftime('%d/%m/%Y')

            # Extract the total (last value in the 'Value' column)
            total_value_unique = df_track_data_unique['Value'].iloc[-1]

            # After extracting total_value_unique
            if len(df_track_data_unique) > 30:
                spotify_streams_1m_unique = total_value_unique - df_track_data_unique['Value'].iloc[-31]
            else:
                spotify_streams_1m_unique = total_value_unique  # or handle it differently if there are not enough rows
            
            if len(df_track_data_unique) > 90:
                spotify_streams_3m_unique = total_value_unique - df_track_data_unique['Value'].iloc[-91]
            else:
                spotify_streams_3m_unique = total_value_unique  # or handle it differently if there are not enough rows
            
            if len(df_track_data_unique) > 365:
                spotify_streams_12m_unique = total_value_unique - df_track_data_unique['Value'].iloc[-366]
            else:
                spotify_streams_12m_unique = total_value_unique  # or handle it differently if there are not enough rows
            
            
            # Get the track name from the file name (without the .csv extension)
            track_name_unique = file_unique.name.split(' - ')[1].strip()


            track_summary_list.append({
                'Track': track_name_unique,                      # Track Name
                'Release date': release_date_unique,            # Release Date
                'Spotify Streams 1m': spotify_streams_1m_unique,  # Last Month Streams
                'Spotify Streams 3m': spotify_streams_3m_unique,  # Last 90 Days Streams
                'Spotify Streams 12m': spotify_streams_12m_unique, # Last 365 Days Streams
                'Spotify Streams Total': total_value_unique       # Total Value
            })

        # Convert the list of track data to a DataFrame
        track_summary_df_unique = pd.DataFrame(track_summary_list)

        # Display the DataFrame with track names, release dates, and total values
        #st.write("Track Summary Data:")
        #st.write(track_summary_df_unique)

    #####################
        #df_additional = pd.read_csv(uploaded_file_additional)
        df = track_summary_df_unique
        if uploaded_file_ownership is not None:
            try:
                ownership_df = pd.read_csv(uploaded_file_ownership, encoding='latin1')  # or 'ISO-8859-1'
            except UnicodeDecodeError:
                ownership_df = pd.read_csv(uploaded_file_ownership, encoding='utf-8')
        
            # Data cleaning and transformation
            ownership_df['Ownership(%)'] = ownership_df['Ownership(%)'].replace('', 1)
            ownership_df['MLC Claimed(%)'] = ownership_df['MLC Claimed(%)'].replace('', 0)
            ownership_df['Ownership(%)'] = pd.to_numeric(ownership_df['Ownership(%)'], errors='coerce').fillna(1)
            ownership_df['MLC Claimed(%)'] = pd.to_numeric(ownership_df['MLC Claimed(%)'], errors='coerce').fillna(0)
        
            ownership_df['Ownership(%)'] = ownership_df['Ownership(%)'].apply(lambda x: x / 100 if x > 1 else x)
            ownership_df['MLC Claimed(%)'] = ownership_df['MLC Claimed(%)'].apply(lambda x: x / 100 if x > 1 else x)
        
            # If ownership file is not uploaded, create the DataFrame manually
        else:
            #st.write(df)
            ownership_df = pd.DataFrame({
                
                'Track': df['Track'],  # Take track names from the main df
                'Ownership(%)': [None] * len(df),  # Create blank column
                'MLC Claimed(%)': [None] * len(df)  # Create blank column
            })
        
        df_additional['Date'] = pd.to_datetime(df_additional['Date'], format='%b-%y')
        
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

        columns_to_drop = ["Release date", "Spotify Streams 1m", "Spotify Streams 3m", "Spotify Streams 12m", "Spotify Streams Total"]
        df.drop(columns=columns_to_drop, inplace=True)

        # Stream influence factor (formerly called sp_playlist_reach)
        stream_influence_factor = 1000
        forecast_periods = 400
        current_date = datetime.today()

        st.write("Data Preview:")
        st.write(df)

        songs = sorted(df['Track'].unique(), key=lambda x: x.lower())
        selected_songs = st.multiselect('Select Songs', songs)

        ##UUPP2##
        if uploaded_file_3:
            audience_df = pd.read_csv(uploaded_file_3)

            # Select the 'Country' and 'Monthly Listeners' columns
            audience_df = audience_df[['Country', 'Spotify Monthly Listeners']]
            
            # Group by 'Country' and sum the 'Monthly Listeners'
            audience_df = audience_df.groupby('Country', as_index=False)['Spotify Monthly Listeners'].sum()
        
            # Calculate the total listeners across all countries
            total_listeners = audience_df['Spotify Monthly Listeners'].sum()
            
            # Create a new column 'Spotify monthly listeners (%)' for the percentage of total listeners per country
            audience_df['Spotify monthly listeners (%)'] = (audience_df['Spotify Monthly Listeners'] / total_listeners) * 100

        #st.write(audience_df)
        ##UUPP2##


        
        
        #audience_df = pd.read_csv(uploaded_file_3, usecols=["Country", "Spotify monthly listeners (%)", "Spotify monthly listeners"])
            audience_df["Spotify monthly listeners (%)"] = pd.to_numeric(audience_df["Spotify monthly listeners (%)"], errors='coerce')
            audience_df["Spotify monthly listeners (%)"] = audience_df["Spotify monthly listeners (%)"] / 100
            percentage_usa = audience_df.loc[audience_df["Country"] == "United States", "Spotify monthly listeners (%)"].values[0]
            
        discount_rate = st.number_input('Discount Rate (%)', min_value=0.00, max_value=10.00, value=0.00, step=0.01, format="%.2f")/100


        song_forecasts = []
        weights_and_changes = []

        if st.button('Run All'):
            years_plot = []
            export_forecasts = pd.DataFrame()
            stream_forecasts = []  # Changed from song_forecasts to stream_forecasts
            weights_and_changes = []

            for selected_song in selected_songs:
                song_data = df[df['Track'] == selected_song].iloc[0]

                value = stream_influence_factor
                total_streams_12_months = song_data['total_streams_12_months']
                total_streams_3_months = song_data['total_streams_3_months']
                streams_last_month = song_data['streams_last_month']
                historical = song_data['historical']
                release_date = song_data['release_date']

                updated_fitted_params_df = update_fitted_params(fitted_params_df, stream_influence_factor, sp_range, SP_REACH)
                #st.write(current_date)
                #st.write(release_date)
                if updated_fitted_params_df is not None:
                    updated_fitted_params = updated_fitted_params_df.to_dict(orient='records')

                ###potential update###3UUPP4
                release_date = datetime.strptime(release_date, "%d/%m/%Y")
                delta = current_date - release_date
                months_since_release_total = delta.days // 30
                
                monthly_avg_3_months = (total_streams_3_months - streams_last_month) / (2 if months_since_release_total > 2 else 1)
            
                
                monthly_avg_last_month = streams_last_month

                if months_since_release_total > 3:
                    monthly_avg_12_months = (total_streams_12_months - total_streams_3_months) / (9 if months_since_release_total > 11 else (months_since_release_total - 3))
                else:
                    monthly_avg_12_months = monthly_avg_3_months

            

                months_since_release = np.array([
                    max((months_since_release_total - 11), 0),
                    max((months_since_release_total - 2), 0),
                    months_since_release_total - 0
                ])
                
                monthly_averages = np.array([monthly_avg_12_months, monthly_avg_3_months, monthly_avg_last_month])
            

                months_since_release = np.array([
                    max((months_since_release_total - 11), 0),
                    max((months_since_release_total - 2), 0),
                    months_since_release_total - 0
                ])
                
                monthly_averages = np.array([monthly_avg_12_months, monthly_avg_3_months, monthly_avg_last_month])
                
                params = fit_segment(months_since_release, monthly_averages)
                S0, k = params
                decay_rates_df = updated_fitted_params_df

                months_since_release_all = list(range(1, 500))
                decay_rate_list = []

                for month in months_since_release_all:
                    for i in range(len(breakpoints) - 1):
                        if breakpoints[i] <= month < breakpoints[i + 1]:
                            decay_rate = decay_rates_df.loc[i, 'k']
                            decay_rate_list.append(decay_rate)
                            break

                final_df = pd.DataFrame({
                    'months_since_release': months_since_release_all,
                    'decay_rate': decay_rate_list
                })

                start_month = min(months_since_release)
                end_month = max(months_since_release)
                final_df.loc[(final_df['months_since_release'] >= start_month) & 
                            (final_df['months_since_release'] <= end_month), 'mldr'] = mldr

                final_df['percent_change'] = ((final_df['mldr'] - final_df['decay_rate']) / final_df['decay_rate']) * 100
                average_percent_change = final_df['percent_change'].mean()
                
                
                if average_percent_change > 0:
                    weight = min(1, max(0, average_percent_change / 100))
                else:
                    weight = 0
            
                
                    
                final_df['adjusted_decay_rate'] = final_df['decay_rate'] * (1 + (average_percent_change * weight) / 100)
                
                
                
                start_month = min(months_since_release)
                end_month = max(months_since_release)
                final_df.loc[(final_df['months_since_release'] >= start_month) & 
                            (final_df['months_since_release'] <= end_month), 'new_decay_rate'] = k

                # Step 4: Compare the adjusted decay rate to the new_decay_rate (k)
                final_df['percent_change_new_vs_adjusted'] = ((final_df['new_decay_rate'] - final_df['adjusted_decay_rate']) / final_df['adjusted_decay_rate']) * 100
                average_percent_change_new_vs_adjusted = final_df['percent_change_new_vs_adjusted'].mean()
                

                # Step 5: Adjust the adjusted decay rate based on the new decay rate comparison
                weight_new = 1 if average_percent_change_new_vs_adjusted > 0 else 0


                final_df['final_adjusted_decay_rate'] = final_df['adjusted_decay_rate'] * (1 + (average_percent_change_new_vs_adjusted * weight_new) / 100)
                




                final_df.drop(['decay_rate', 'mldr', 'percent_change'], axis=1, inplace=True)

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

                initial_value = streams_last_month
                start_period = months_since_release_total

                forecasts = forecast_values(consolidated_df, initial_value, start_period, forecast_periods)

                # Convert forecasts to a DataFrame
                forecasts_df = pd.DataFrame(forecasts)
                forecasts_df2 = forecasts_df
                forecasts_df2['track_name_unique'] = selected_song
                export_forecasts = pd.concat([export_forecasts, forecasts_df2], ignore_index=True)
                
                
                
                
                
                
                # Calculate the total forecast value for the first 240 months
                total_forecast_value = forecasts_df.loc[:240, 'forecasted_value'].sum()

                release_date = song_data['release_date']  # Example release date
                ###potential UUPP5
                release_date = datetime.strptime(release_date, "%d/%m/%Y")
                start_date = release_date.strftime('%Y-%m')
                end_date = '2024-02'  # Default end date
                if release_date.strftime('%Y-%m') >= end_date:
                    end_date = df_additional['Date'].max().strftime('%Y-%m')
                    
                
                mask = (df_additional['Date'] >= start_date) & (df_additional['Date'] <= end_date)
                #HISTORICAL VALUE
                ad_supported = df_additional.loc[mask, 'Spotify_Ad-supported'].mean()
                premium = df_additional.loc[mask, 'Spotify_Premium'].mean()
                hist_ad = 0.6 * historical * ad_supported
                hist_prem = 0.4 * historical * premium
                hist_value = (hist_ad + hist_prem) * (percentage_usa)
                hist_value = hist_value / ((1 + discount_rate / 12) ** 3)
                

                
    
                
                
                
                
                

                #final_5_ad_supported = df_additional['Spotify_Ad-supported'].tail(5).mean()
                #final_5_premium = df_additional['Spotify_Premium'].tail(5).mean()
                
                #forecast_ad = 0.6 * total_forecast_value * final_5_ad_supported
                #forecast_prem = 0.4 * total_forecast_value * final_5_premium
                #forecast_OG = forecast_ad + forecast_prem
                
                
                
                
                monthly_forecasts_df = pd.DataFrame({
                    'Track': [selected_song] * len(forecasts_df),
                    'Month': forecasts_df['month'],
                    'Forecasted Value': forecasts_df['forecasted_value']
                })

                # Step 1: Add columns for each country's percentage from audience_df to monthly_forecasts_df
                monthly_forecasts_df['Month Index'] = monthly_forecasts_df.index + 1
                for index, row in audience_df.iterrows():
                    country = row['Country']
                    percentage = row['Spotify monthly listeners (%)']
                    monthly_forecasts_df[country + ' %'] = percentage

                # Step 2: Calculate the mean of the final 5 values for each country's royalty rate from GLOBAL
                for index, row in audience_df.iterrows():
                    country = row['Country']
                    if country in GLOBAL.columns:
                        mean_final_5 = GLOBAL[country].dropna().tail(5).mean()
                        monthly_forecasts_df[country + ' Royalty Rate'] = mean_final_5

                # Multiply the forecasted value with each country's percentage and add the result as new columns
                for index, row in audience_df.iterrows():
                    country = row['Country']
                    monthly_forecasts_df[country + ' Value'] = monthly_forecasts_df['Forecasted Value'] * monthly_forecasts_df[country + ' %']

                # Multiply each country's streams by its corresponding royalty rate
                for index, row in audience_df.iterrows():
                    country = row['Country']
                    monthly_forecasts_df[country + ' Royalty Value'] = monthly_forecasts_df[country + ' Value'] * monthly_forecasts_df[country + ' Royalty Rate']

                # Drop the percentage columns
                percentage_columns = [country + ' %' for country in audience_df['Country']]
                monthly_forecasts_df.drop(columns=percentage_columns, inplace=True)

                
                columns_to_drop = [country + ' Value' for country in audience_df['Country']] + [country + ' Royalty Rate' for country in audience_df['Country']]
                monthly_forecasts_df.drop(columns=columns_to_drop, inplace=True)
                
                monthly_forecasts_df['Total'] = monthly_forecasts_df[[country + ' Royalty Value' for country in audience_df['Country']]].sum(axis=1)

                
                
                
                monthly_forecasts_df['DISC'] = (monthly_forecasts_df['Total']) / ((1 + discount_rate / 12) ** (monthly_forecasts_df['Month Index'] + 2.5))
                
                

                
                new_forecast_value = monthly_forecasts_df['DISC'].sum()
                forecast_OG = monthly_forecasts_df['Total'].sum()
                Total_Value = new_forecast_value + hist_value

                # Append the forecast data with the new forecast value
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

                rows_per_period = 12
                n_rows = len(monthly_forecasts_df)
                
                # Initialize the period pattern list
                period_pattern = []
                
                # Start with 9 occurrences of '1'
                period_pattern.extend([1] * 9)
                
                # Calculate remaining rows after adding the first 9 '1's
                remaining_rows = n_rows - 9
                
                # Continue with the usual pattern of 12 occurrences per period
                for period in range(2, (remaining_rows // rows_per_period) + 2):
                    period_pattern.extend([period] * rows_per_period)

                # Trim or extend the pattern to exactly match the number of rows
                if len(period_pattern) > n_rows:
                    period_pattern = period_pattern[:n_rows]  # Trim the list if it's too long
                else:
                    period_pattern.extend([period] * (n_rows - len(period_pattern)))  # Extend the last period if it's too short

                # Assign the Period column based on the pattern
                monthly_forecasts_df['Period'] = period_pattern

                # Group by 'Period' and aggregate the sum
                aggregated_df = monthly_forecasts_df.groupby('Period').agg({
                    'Track': 'first',
                    'Month': 'first',  # Use the first month in each period
                    'DISC': 'sum'
                }).reset_index(drop=True)

                # Optionally rename columns for clarity
                aggregated_df.rename(columns={'Month': 'Start_Month'}, inplace=True)

                aggregated_df = aggregated_df.head(10)

                # Replace the 'Start_Month' column with 'Year'
                aggregated_df['Year'] = range(1, 11)

                # Drop the 'Start_Month' column
                aggregated_df.drop(columns=['Start_Month'], inplace=True)

                years_plot.append(aggregated_df)











            
                # Display song forecasts DataFrame
            catalog_to_download = export_forecasts
            csv = catalog_to_download.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="export_forecasts.csv">Download forecasts DataFrame</a>'
            st.markdown(href, unsafe_allow_html=True)
            
            years_plot_df = pd.concat(years_plot)
            yearly_disc_sum_df = years_plot_df.groupby('Year')['DISC'].sum().reset_index()
            df_forecasts = pd.DataFrame(song_forecasts)

            merged_df = df_forecasts.merge(ownership_df[['Track', 'MLC Claimed(%)', 'Ownership(%)']], on='Track', how='left')

            # Ensure 'MLC Claimed(%)' and 'Ownership(%)' are treated as floats
            merged_df['MLC Claimed(%)'] = pd.to_numeric(merged_df['MLC Claimed(%)'], errors='coerce').fillna(0)
            merged_df['Ownership(%)'] = pd.to_numeric(merged_df['Ownership(%)'], errors='coerce').fillna(1)
            
            # Calculate the new hist_value after considering MLC Claimed(%)
            merged_df['hist_value'] = merged_df.apply(
                lambda row: min((1 - row['MLC Claimed(%)']) * row['hist_value'], row['Ownership(%)'] * row['hist_value']),
                axis=1
            )
            
            
            # Calculate new forecast values after considering Ownership(%)
            merged_df['Forecast_no_disc'] = merged_df['Forecast_no_disc'].astype(float) * (merged_df['Ownership(%)'])
            merged_df['Forecast_disc'] = merged_df['Forecast_disc'].astype(float) * (merged_df['Ownership(%)'])
            merged_df['Total_Value']  = merged_df['Forecast_disc'] + merged_df['hist_value']
            merged_df = merged_df.drop(columns=['Ownership(%)', 'MLC Claimed(%)'])
            
            df_forecasts = merged_df
            
            df_forecasts['Historical'] = df_forecasts['Historical'].astype(float).apply(lambda x: f"{int(round(x)):,}")
            df_forecasts['Forecast'] = df_forecasts['Forecast'].astype(float).apply(lambda x: f"{int(round(x)):,}")
            df_forecasts['Forecast_no_disc'] = df_forecasts['Forecast_no_disc'].astype(float).apply(lambda x: f"${int(round(x)):,}")
            df_forecasts['Forecast_disc'] = df_forecasts['Forecast_disc'].astype(float).apply(lambda x: f"${int(round(x)):,}")
            df_forecasts['hist_value'] = df_forecasts['hist_value'].astype(float).apply(lambda x: f"${int(round(x)):,}")
            df_forecasts['Total_Value'] = df_forecasts['Total_Value'].astype(float).apply(lambda x: f"${int(round(x)):,}")

            st.write(df_forecasts)

            # Calculate summed values for the summary DataFrame
            sum_df = pd.DataFrame({
                'Metric': ['hist_value', 'Forecast_OG','Forecast_dis', 'Total_Value'],
                'Sum': [
                    df_forecasts['hist_value'].apply(lambda x: int(x.replace('$', '').replace(',', ''))).sum(),
                    df_forecasts['Forecast_no_disc'].apply(lambda x: int(x.replace('$', '').replace(',', ''))).sum(),
                    df_forecasts['Forecast_disc'].apply(lambda x: int(x.replace('$', '').replace(',', ''))).sum(),
                    df_forecasts['Total_Value'].apply(lambda x: int(x.replace('$', '').replace(',', ''))).sum()
                ]
            })

            sum_df['Sum'] = sum_df['Sum'].apply(lambda x: f"${int(round(x)):,}")

            st.write("Summed Values:")
            st.write(sum_df)


            country_breakdown = []
            for index, row in audience_df.iterrows():
                country = row['Country']
                forecast_no_disc_value = monthly_forecasts_df[country + ' Royalty Value'].sum() 
                country_breakdown.append({
                    'Country': country,
                    'Forecast_no_disc': forecast_no_disc_value
                })
            # Create a DataFrame for the country breakdown
            df_country_breakdown = pd.DataFrame(country_breakdown)
            df_country_breakdown['Forecast_no_disc_numeric'] = df_country_breakdown['Forecast_no_disc'].replace({'\$': '', ',': ''}, regex=True).astype(float)

            # Calculate the total forecast value
            total_forecast_no_disc_value = df_country_breakdown['Forecast_no_disc_numeric'].sum()
            
            # Calculate the percentage for each country
            df_country_breakdown['Percentage'] = (df_country_breakdown['Forecast_no_disc_numeric'] / total_forecast_no_disc_value) * 100
            
            # Sort by 'Forecast_no_disc_numeric' and select the top 10 countries
            top_countries = df_country_breakdown.sort_values(by='Forecast_no_disc_numeric', ascending=False).head(10)
            top_10_percentage_sum = top_countries['Percentage'].sum()
            fig, ax = plt.subplots()
            bar_color = 'teal'
            bars = ax.barh(top_countries['Country'], top_countries['Forecast_no_disc_numeric'], color=bar_color)

            ax.set_xlabel('% of Forecast Value')
            ax.set_title(f'Top 10 Countries Contribute {top_10_percentage_sum:.1f}% to Total Forecast Value')

            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${int(x):,}"))

            max_value = top_countries['Forecast_no_disc_numeric'].max()
            ax.set_xlim(0, max_value * 1.25)
            ax.set_xticks([])
            for bar, percentage in zip(bars, top_countries['Percentage']):
                width = bar.get_width()
                ax.text(width + (width * 0.01), bar.get_y() + bar.get_height() / 2, 
                        f'{percentage:.1f}%', va='center', ha='left', 
                        fontsize=10, color='black')

            st.pyplot(fig)

            bar_color = 'teal'
            fig, ax = plt.subplots()

            bars = ax.bar(yearly_disc_sum_df['Year'], yearly_disc_sum_df['DISC'], color=bar_color)

            ax.set_xlabel('Year')
            ax.set_title('Income by Year (discounted)')

            ax.set_ylabel('')

            ax.yaxis.set_visible(False)

            max_value = yearly_disc_sum_df['DISC'].max()
            ax.set_ylim(0, max_value * 1.25)

            for bar, value in zip(bars, yearly_disc_sum_df['DISC']):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, height, f'${int(value)}', 
                        va='bottom', ha='center', fontsize=10, color='black')

            st.pyplot(fig)

            warning_df = pd.merge(audience_df, population_df, on='Country', how='left')
            
            # Calculate 20% of the population
            warning_df['TwentyPercentPopulation'] = warning_df['Population'] * 0.20
            
            # Check if Spotify monthly listeners are greater than 20% of the population
            warning_df['Above20Percent'] = warning_df['Spotify Monthly Listeners'] > warning_df['TwentyPercentPopulation']
            
            # Add a column with 1 if above 20%
            warning_df['Alert'] = warning_df['Above20Percent'].apply(lambda x: 1 if x else 0)
            
            # Get the list of countries with fraud alert (Alert = 1)
            alert_countries = warning_df[warning_df['Alert'] == 1]['Country']
            
            # Check if there are any countries to display
            if not alert_countries.empty:
                st.write("Fraud Alert. This artist has unusually high streams from these countries:")
                for country in alert_countries:
                    st.write(country)

