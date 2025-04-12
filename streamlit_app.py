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

population_df = get_population_data()

ranges_sp = {
    'Column 1': list(range(1, 11)),
    'RangeStart': [0, 10000, 30000, 50000, 75000, 110000, 160000, 250000, 410000, 950000],
    'RangeEnd': [10000, 30000, 50000, 75000, 110000, 160000, 250000, 410000, 950000, 1E18]
}
sp_range = pd.DataFrame(ranges_sp)

SP_REACH_DATA = {
    'Unnamed: 0': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    1: [0.094000, 0.064033, 0.060000, 0.050000, 0.035000, 0.030000, 0.030000, 0.030000, 0.030000, 0.015000, 0.015000, 0.020000],
    2: [0.089111, 0.061430, 0.055739, 0.044959, 0.031111, 0.026667, 0.026667, 0.026877, 0.026667, 0.013990, 0.013333, 0.020000],
    3: [0.084222, 0.058826, 0.051478, 0.039918, 0.027222, 0.023333, 0.023333, 0.023754, 0.023333, 0.012980, 0.011667, 0.020000],
    4: [0.079333, 0.056222, 0.047218, 0.034877, 0.023333, 0.020000, 0.020000, 0.020631, 0.020000, 0.011971, 0.010000, 0.020000],
    5: [0.074444, 0.053619, 0.042957, 0.029836, 0.019444, 0.016667, 0.016667, 0.017508, 0.016667, 0.010961, 0.008333, 0.020000],
    6: [0.069556, 0.051015, 0.038696, 0.024795, 0.015556, 0.013333, 0.013333, 0.014385, 0.013333, 0.009951, 0.006667, 0.020000],
    7: [0.064667, 0.048411, 0.034435, 0.019754, 0.011667, 0.010000, 0.010000, 0.011262, 0.010000, 0.008941, 0.005000, 0.020000],
    8: [0.059778, 0.045808, 0.030174, 0.014713, 0.007778, 0.006667, 0.006667, 0.008138, 0.006667, 0.007931, 0.003333, 0.020000],
    9: [0.054889, 0.043204, 0.025913, 0.009672, 0.003889, 0.003333, 0.003333, 0.005015, 0.003333, 0.006921, 0.001667, 0.020000],
    10: [0.050000, 0.040600, 0.021653, 0.004631, 0.010000, 0.010000, 0.010000, 0.001892, 0.010000, 0.005912, 0.010000, 0.020000]
}

# Create the DataFrame
SP_REACH = pd.DataFrame(SP_REACH_DATA)

fitted_params = [
    {'segment': 1, 'S0': 7239.425562317985, 'k': 0.06741191851584262},
    {'segment': 2, 'S0': 6465.440296195081, 'k': 0.03291507714354558},
    {'segment': 3, 'S0': 6478.639247351713, 'k': 0.03334620907608441},
    {'segment': 4, 'S0': 5755.53795902042, 'k': 0.021404012549575913},
    {'segment': 5, 'S0': 6023.220319977014, 'k': 0.02461834982301452},
    {'segment': 6, 'S0': 6712.835052107982, 'k': 0.03183160108111365},
    {'segment': 7, 'S0': 6371.457552382675, 'k': 0.029059156192761115},
    {'segment': 8, 'S0': 5954.231622567404, 'k': 0.02577913683190864},
    {'segment': 9, 'S0': 4932.65240022657, 'k': 0.017941231431835854},
    {'segment': 10, 'S0': 3936.0657447490344, 'k': 0.009790878919164516},
    {'segment': 11, 'S0': 4947.555706076349, 'k': 0.016324033736761206},
    {'segment': 12, 'S0': 4000, 'k': 0.0092302}
]


fitted_params_df = pd.DataFrame(fitted_params)

# Define breakpoints for segments
breakpoints = [1, 3, 6, 9, 12, 15, 18, 21, 24, 27, 36, 48, 100000]

# Define the piecewise exponential decay function
def piecewise_exp_decay(x, S0, k):
    return S0 * np.exp(-k * x)

# Function to fit model to a segment of data
def fit_segment(months_since_release, streams):
    initial_guess = [streams[0], 0.01]  
    bounds = ([0, 0], [np.inf, np.inf])  
    
    params, covariance = curve_fit(piecewise_exp_decay, months_since_release, streams, p0=initial_guess, bounds=bounds)
    
    return params

# Update fitted parameters based on Spotify Playlist Reach
def update_fitted_params(fitted_params_df, value, sp_range, SP_REACH):
    updated_fitted_params_df = fitted_params_df.copy()
    
    segment = sp_range.loc[(sp_range['RangeStart'] <= value) & (sp_range['RangeEnd'] > value), 'Column 1'].iloc[0]
    
    if segment not in SP_REACH.columns:
        st.error(f"Error: Column '{segment}' not found in SP_REACH.")
        st.write("Available columns:", SP_REACH.columns)
        return None
    
    column_to_append = SP_REACH[segment]
    updated_fitted_params_df['k'] = updated_fitted_params_df['k'] * 0.67 + column_to_append * 0.33

    return updated_fitted_params_df

# Function to forecast values
def forecast_values(consolidated_df, initial_value, start_period, forecast_periods):
    params = consolidated_df.to_dict(orient='records')
    forecasts = []
    current_value = initial_value
    
    for i in range(forecast_periods):
        current_segment = 0
        current_month = start_period + i
        
        while current_month >= sum(len(range(breakpoints[j] + 1, breakpoints[j + 1] + 1)) for j in range(current_segment + 1)):
            current_segment += 1
        
        current_segment_params = params[current_segment]
        S0 = current_value
        k = current_segment_params['k']
        
        forecast_value = S0 * np.exp(-k * (1))
        
        forecasts.append({
            'month': current_month,
            'forecasted_value': forecast_value,
            'segment_used': current_segment + 1,
            'time_used': current_month - start_period + 1
        })
        
        current_value = forecast_value
    
    return forecasts


def exponential_decay(x, a, b):
    return a * np.exp(-b * x)

def remove_anomalies(data):
    # Calculate the 4-week moving average
    data['4_Week_MA'] = data['Streams'].rolling(window=4, min_periods=1).mean()

    # Resample the data to monthly sums
    data.set_index('Date', inplace=True)
    monthly_data = data['4_Week_MA'].resample('M').sum().reset_index()

    # Detect anomalies using the IQR method
    Q1 = monthly_data['4_Week_MA'].quantile(0.25)
    Q3 = monthly_data['4_Week_MA'].quantile(0.75)
    IQR = Q3 - Q1

    # Define bounds for anomaly detection
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Detect anomalies
    monthly_data['is_anomaly'] = (monthly_data['4_Week_MA'] < lower_bound) | (monthly_data['4_Week_MA'] > upper_bound)

    # Interpolate anomalies
    for i in range(1, len(monthly_data) - 1):
        if monthly_data.loc[i, 'is_anomaly']:
            monthly_data.loc[i, '4_Week_MA'] = (monthly_data.loc[i - 1, '4_Week_MA'] + monthly_data.loc[i + 1, '4_Week_MA']) / 2

    return monthly_data

def calculate_decay_rate(monthly_data):
    # Calculate the number of months since the first date in the filtered data
    min_date = monthly_data['Date'].min()
    monthly_data['Months'] = monthly_data['Date'].apply(lambda x: (x.year - min_date.year) * 12 + x.month - min_date.month)

    # Fit the exponential decay model to the monthly data
    x_data = monthly_data['Months']
    y_data = monthly_data['4_Week_MA']

    # Use initial guesses for curve fitting
    popt, _ = curve_fit(exponential_decay, x_data, y_data, p0=(max(y_data), 0.1))

    # Extract the decay rate (b)
    decay_rate = popt[1]
    return decay_rate, popt  # Ensure both decay_rate and popt are returned


st.title('mitch_refactor_valuation_app')

# Load the mechanical royalties data from utils module
df_additional = get_mech_data()
if df_additional is None:
    st.error("Failed to load mechanical royalties data")
    st.stop()

# Load the worldwide rates data from utils module
GLOBAL = get_rates_data()
if GLOBAL is None:
    st.error("Failed to load worldwide rates data")
    st.stop()

tab1, tab2, tab3 = st.tabs(["API Search", "File Uploader", "Backtest"])

def calculate_graph(df, discount_rate, selected_songs):
    song_forecasts = []
    weights_and_changes = []

    years_plot = []
    export_forecasts = pd.DataFrame()

    for selected_song in selected_songs:
        song_data = df[df['Track'] == selected_song].iloc[0]

        value = sp_playlist_reach
        total_streams_12_months = song_data['total_streams_12_months']
        total_streams_3_months = song_data['total_streams_3_months']
        streams_last_month = song_data['streams_last_month']
        historical = song_data['historical']
        release_date = song_data['release_date']

        updated_fitted_params_df = update_fitted_params(fitted_params_df, sp_playlist_reach, sp_range, SP_REACH)
        # st.write(current_date)
        # st.write(release_date)
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
        # HISTORICAL VALUE
        ad_supported = df_additional.loc[mask, 'Spotify_Ad-supported'].mean()
        premium = df_additional.loc[mask, 'Spotify_Premium'].mean()
        hist_ad = 0.6 * historical * ad_supported
        hist_prem = 0.4 * historical * premium
        hist_value = (hist_ad + hist_prem) * (percentage_usa)
        hist_value = hist_value / ((1 + discount_rate / 12) ** 3)

        # final_5_ad_supported = df_additional['Spotify_Ad-supported'].tail(5).mean()
        # final_5_premium = df_additional['Spotify_Premium'].tail(5).mean()

        # forecast_ad = 0.6 * total_forecast_value * final_5_ad_supported
        # forecast_prem = 0.4 * total_forecast_value * final_5_premium
        # forecast_OG = forecast_ad + forecast_prem

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

    catalog_to_download = export_forecasts
    csv = catalog_to_download.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="export_forecasts.csv">Download forecasts DataFrame</a>'
    st.markdown(href, unsafe_allow_html=True)

    years_plot_df = pd.concat(years_plot)
    yearly_disc_sum_df = years_plot_df.groupby('Year')['DISC'].sum().reset_index()
    df_forecasts = pd.DataFrame(song_forecasts)

    merged_df = df_forecasts.merge(ownership_df[['Track', 'MLC Claimed(%)', 'Ownership(%)']], on='Track', how='left')

    merged_df['MLC Claimed(%)'] = pd.to_numeric(merged_df['MLC Claimed(%)'], errors='coerce').fillna(0)
    merged_df['Ownership(%)'] = pd.to_numeric(merged_df['Ownership(%)'], errors='coerce').fillna(1)

    merged_df['hist_value'] = merged_df.apply(
        lambda row: min((1 - row['MLC Claimed(%)']) * row['hist_value'], row['Ownership(%)'] * row['hist_value']),
        axis=1
    )

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
    df_country_breakdown = pd.DataFrame(country_breakdown)
    df_country_breakdown['Forecast_no_disc_numeric'] = df_country_breakdown['Forecast_no_disc'].replace({'\$': '', ',': ''}, regex=True).astype(float)

    total_forecast_no_disc_value = df_country_breakdown['Forecast_no_disc_numeric'].sum()

    df_country_breakdown['Percentage'] = (df_country_breakdown['Forecast_no_disc_numeric'] / total_forecast_no_disc_value) * 100

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
    warning_df['TwentyPercentPopulation'] = warning_df['Population'] * 0.20
    warning_df['Above20Percent'] = warning_df['Spotify Monthly Listeners'] > warning_df['TwentyPercentPopulation']
    warning_df['Alert'] = warning_df['Above20Percent'].apply(lambda x: 1 if x else 0)
    alert_countries = warning_df[warning_df['Alert'] == 1]['Country']
    if not alert_countries.empty:
        st.write("Fraud Alert. This artist has unusually high streams from these countries:")
        for country in alert_countries:
            st.write(country)


@st.fragment
def component_handler(df):

    discount_rate = (
        st.number_input(
            "Discount Rate (%)",
            key="text_input_value",
            min_value=0.00,
            max_value=10.00,
            value=4.50,
            step=0.01,
            format="%.2f",
        )
        / 100
    )

    songs = sorted(df["Track"].unique(), key=lambda x: x.lower())
    selected_songs = st.multiselect(
        "Select Songs", songs, default=songs, key="selected_songs"
    )

    if discount_rate and selected_songs:
        calculate_graph(df, discount_rate=discount_rate, selected_songs=selected_songs)
    else:
        if st.button("Run All"):
            calculate_graph(
                df=df,
                discount_rate=st.session_state.text_input_value,
                selected_songs=st.session_state.selected_songs,
            )

with tab1:
    st.title("Artist and Track ID Form")
    
    # Initialize session state variables if they don't exist
    if 'artist_data' not in st.session_state:
        st.session_state.artist_data = None
    if 'monthly_data' not in st.session_state:
        st.session_state.monthly_data = None
    if 'date_range' not in st.session_state:
        st.session_state.date_range = {
            'min_date': None,
            'max_date': None,
            'start_date': None,
            'end_date': None
        }
    if 'mldr' not in st.session_state:
        st.session_state.mldr = None
    if 'track_summary_list' not in st.session_state:
        st.session_state.track_summary_list = []
    
    # Function to load test data
    def load_test_data():
        st.session_state['artist_id'] = "4276517"  # Thymes
        st.session_state['track_id'] = "60300793"  # Free as a bird
        
    # Initialize session state for input fields if not already done
    if 'artist_id' not in st.session_state:
        st.session_state['artist_id'] = ""
    if 'track_id' not in st.session_state:
        st.session_state['track_id'] = ""
        
    with st.form(key="artist_track_form"):
        st.subheader("Enter Details")
        artist_id = st.text_input("Artist ID", placeholder="Enter Artist ID", value=st.session_state['artist_id'], key="artist_id_input")
        track_ids_input = st.text_input("Track ID", placeholder="Enter Track ID", value=st.session_state['track_id'], key="track_id_input")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            submit_button = st.form_submit_button("Submit")
        with col2:
            load_test_button = st.form_submit_button("Load Test Data", help="Load test data with Artist ID: Thymes (4276517) and Track ID: Free as a bird (60300793)")
            
    # Handle the load test button click
    if load_test_button:
        load_test_data()
        st.rerun()  # Rerun the app to reflect the changes in the form fields

    if submit_button:
        start_time = time.time()
        st.session_state.track_summary_list = []  # Reset the list when form is submitted

        data = chartmetric.get_artist_spotify_stats(artist_id)
        df = pd.DataFrame(data)
        df.rename(columns={'timestp': 'Date', 'value': 'Monthly Listeners', 'diff': 'Difference'}, inplace=True)
        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%b %d, %Y')
        df['Difference'] = df['Difference'].fillna(0).astype(int)

        if 'Date' in df.columns:
            try:
                df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
            except Exception as e:
                st.error(f"Failed to convert Date column: {e}")

        if df['Date'].isna().any():
            st.warning("Some dates couldn't be parsed and have been set to 'NaT'. Please check your data.")

        df = df[['Date', 'Monthly Listeners']].rename(columns={'Monthly Listeners': 'Streams'})
        df = df.iloc[::7, :]
        df = df.sort_values(by='Date')
        
        # Remove anomalies and store in session state
        st.session_state.monthly_data = remove_anomalies(df)
        st.session_state.artist_data = df
        
        # Update date range in session state
        st.session_state.date_range = {
            'min_date': st.session_state.monthly_data['Date'].min().to_pydatetime(),
            'max_date': st.session_state.monthly_data['Date'].max().to_pydatetime(),
            'start_date': st.session_state.monthly_data['Date'].min().to_pydatetime(),
            'end_date': st.session_state.monthly_data['Date'].max().to_pydatetime()
        }

    # Only show date range slider and plot if we have data
    if st.session_state.monthly_data is not None:
        st.write("Select Date Range:")
        try:
            selected_start, selected_end = st.slider(
                "Select date range",
                min_value=st.session_state.date_range['min_date'],
                max_value=st.session_state.date_range['max_date'],
                value=(st.session_state.date_range['start_date'], 
                      st.session_state.date_range['end_date']),
                format="YYYY-MM-DD",
                key='date_slider'
            )
            
            # Update session state with new selection
            st.session_state.date_range['start_date'] = selected_start
            st.session_state.date_range['end_date'] = selected_end
            
            # Filter data based on selection
            mask = (st.session_state.monthly_data['Date'] >= selected_start) & (st.session_state.monthly_data['Date'] <= selected_end)
            subset_df = st.session_state.monthly_data[mask].copy()
            
            if len(subset_df) > 0:
                # Calculate months for decay rate
                min_date = subset_df['Date'].min()
                subset_df['Months'] = subset_df['Date'].apply(
                    lambda x: (x.year - min_date.year) * 12 + x.month - min_date.month
                )
                
                # Calculate decay rate and store in session state
                mldr, popt = calculate_decay_rate(subset_df)
                st.session_state.mldr = mldr
                st.write(f'Exponential decay ratemldr: {mldr}')
                
                # Create plot
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(subset_df['Date'], subset_df['4_Week_MA'], 
                       label='Moving Average', color='tab:blue', linewidth=2)
                ax.plot(subset_df['Date'], 
                       exponential_decay(subset_df['Months'], *popt), 
                       label='Fitted Decay Curve', color='red', linestyle='--')
                
                ax.set_xlabel('Date', fontsize=12)
                ax.set_ylabel('Monthly Listeners', fontsize=12)  # Changed from 'Streams' to 'Monthly Listeners'
                ax.set_title('Moving Average and Exponential Decay', 
                           fontsize=14, weight='bold')
                ax.legend()
                ax.set_ylim(bottom=0)
                plt.xticks(rotation=45)
                
                fig.patch.set_visible(False)
                ax.set_facecolor('none')
                ax.patch.set_alpha(0)
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.warning("No data available for the selected date range.")
        except Exception as e:
            st.error(f"Error updating date range: {str(e)}")
            st.write("Please try adjusting the date range again.")

        track_ids = track_ids_input.split(',')
        for track_id in track_ids:
            track_id = track_id.strip()
            df_track_data_unique = chartmetric.get_track_sp_streams_campare(track_id=track_id)
            time.sleep(1)
            track_detail = chartmetric.get_track_detail(track_id=track_id)
            time.sleep(1)

            df_track_data_unique = pd.DataFrame(df_track_data_unique)
            df_track_data_unique.rename(columns={'timestp': 'Date', 'value': 'Value'}, inplace=True)
            
            # Display date range information
            start_date = pd.to_datetime(df_track_data_unique['Date'].min())
            end_date = pd.to_datetime(df_track_data_unique['Date'].max())
            st.write(f"Track: {track_detail.name}")
            st.write(f"Data range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            st.write(f"Total days of data: {(end_date - start_date).days}")
            
            df_track_data_unique['Date'] = pd.to_datetime(df_track_data_unique['Date']).dt.strftime('%b %d, %Y')
            release_date_unique = pd.to_datetime(df_track_data_unique['Date'].iloc[0], format='%b %d, %Y').strftime('%d/%m/%Y')
            total_value_unique = df_track_data_unique['Value'].iloc[-1]
            
            if len(df_track_data_unique) > 30:
                spotify_streams_1m_unique = total_value_unique - df_track_data_unique['Value'].iloc[-31]
            else:
                spotify_streams_1m_unique = total_value_unique
            
            if len(df_track_data_unique) > 90:
                spotify_streams_3m_unique = total_value_unique - df_track_data_unique['Value'].iloc[-91]
            else:
                spotify_streams_3m_unique = total_value_unique
            
            if len(df_track_data_unique) > 365:
                spotify_streams_12m_unique = total_value_unique - df_track_data_unique['Value'].iloc[-366]
            else:
                spotify_streams_12m_unique = total_value_unique
            
            track_name_unique = track_detail.name

            st.session_state.track_summary_list.append({
                'Track': track_name_unique,
                'Release date': release_date_unique,
                'Spotify Streams 1m': spotify_streams_1m_unique,
                'Spotify Streams 3m': spotify_streams_3m_unique,
                'Spotify Streams 12m': spotify_streams_12m_unique,
                'Spotify Streams Total': total_value_unique
            })

        track_summary_df_unique = pd.DataFrame(st.session_state.track_summary_list)
        df = track_summary_df_unique
        ownership_df = pd.DataFrame({
            'Track': df['Track'],
            'Ownership(%)': [None] * len(df),
            'MLC Claimed(%)': [None] * len(df)
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

        sp_playlist_reach = 1000
        forecast_periods = 400
        current_date = datetime.today()

        st.write("Data Preview:")
        st.write(df)

        # Third Endpoint Calls
        # Use a specific date (August 12, 2024) as mentioned in the documentation
        api_params = {
            'limit': 50,       # Request the maximum number of cities (50)
            'date': '2024-08-12'  # Specific date snapshot (August 12, 2024)
        }
        
        # Call the API directly to get the raw response
        response = chartmetric._ChartMetricService__get_artist_track_where_people_listen_request(artist_id, api_params)
        
        # Check if the API response has the expected structure
        if "obj" in response and "cities" in response["obj"]:
            raw_obj = response["obj"]
            
            # Get city data
            cities_data = []
            for city, entries in raw_obj["cities"].items():
                for entry in entries:
                    city_entry = {
                        'City': city,
                        'Country': entry.get('code2', ''),
                        'Spotify Monthly Listeners': entry.get('listeners', 0),
                        'Date': api_params['date']
                    }
                    cities_data.append(city_entry)
            
            # Create DataFrame with city data
            cities_df = pd.DataFrame(cities_data)
            
            st.write(f"### Top Cities for {api_params['date']}: Found {len(cities_data)} cities")
            st.write(cities_df)
            
            # Group by Country (similar to Tab 2 approach)
            country_df = cities_df.groupby('Country')['Spotify Monthly Listeners'].sum().reset_index()
            
            st.write("### Country Totals (from city data):")
            st.write(country_df)
            
            # Download link for the data
            csv = cities_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="city_data_{api_params["date"]}.csv">Download city data as CSV</a>'
            st.markdown(href, unsafe_allow_html=True)
            
            # Use the city-based country data for further processing
            audience_df = country_df
            
            # Convert country codes to full names to match GLOBAL dataset
            audience_df['Original_Country'] = audience_df['Country']  # Save original code
            audience_df['Country'] = audience_df['Country'].apply(country_code_to_name)
            
            # Process the country data
            total_listeners = audience_df['Spotify Monthly Listeners'].sum()
            audience_df['Spotify monthly listeners (%)'] = (audience_df['Spotify Monthly Listeners'] / total_listeners) * 100
            
            st.write("### Final Processed Data (with country names):")
            st.write(audience_df)
            
            audience_df["Spotify monthly listeners (%)"] = pd.to_numeric(audience_df["Spotify monthly listeners (%)"], errors='coerce')
            audience_df["Spotify monthly listeners (%)"] = audience_df["Spotify monthly listeners (%)"] / 100
            
            # Check if United States exists before trying to access it
            if "United States" in audience_df["Country"].values:
                percentage_usa = audience_df.loc[audience_df["Country"] == "United States", "Spotify monthly listeners (%)"].values[0]
            else:
                st.error("No United States data found - this will cause an error in further processing")
                # This will error out if United States is not found, which is what we want for debugging
                percentage_usa = audience_df.loc[audience_df["Country"] == "United States", "Spotify monthly listeners (%)"].values[0]
        else:
            st.error("Could not retrieve city data for the specified date. API response:")
            st.write(response)
            st.stop()
            
        component_handler(df)

with tab2:

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

        sp_playlist_reach = 1000
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

                value = sp_playlist_reach
                total_streams_12_months = song_data['total_streams_12_months']
                total_streams_3_months = song_data['total_streams_3_months']
                streams_last_month = song_data['streams_last_month']
                historical = song_data['historical']
                release_date = song_data['release_date']

                updated_fitted_params_df = update_fitted_params(fitted_params_df, sp_playlist_reach, sp_range, SP_REACH)
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

def split_listener_history_for_backtesting(df):
    """
    Split the artist listener history data into training and validation sets,
    rounding to whole months.
    
    Args:
        df: DataFrame with 'Date' and 'Monthly Listeners' columns
        
    Returns:
        tuple: (training_df, validation_df) or (None, None) if data is insufficient
    """
    # Convert Date to datetime if it's not already
    df['Date'] = pd.to_datetime(df['Date'], format='%b %d, %Y')
    
    # Sort by date
    df = df.sort_values('Date')
    
    # Get the first and last dates
    start_date = df['Date'].min()
    end_date = df['Date'].max()
    
    # Round start date to beginning of month and end date to end of month
    start_date = start_date.replace(day=1)
    end_date = (end_date.replace(day=1) + pd.DateOffset(months=1) - pd.DateOffset(days=1))
    
    # Calculate total months of data
    total_months = ((end_date.year - start_date.year) * 12 + end_date.month - start_date.month)
    
    # Check if we have minimum required data (27 months)
    if total_months < 27:
        return None, None
        
    # Calculate the cutoff date for validation (exactly 24 months from the end)
    validation_start = end_date - pd.DateOffset(months=24)
    validation_start = validation_start.replace(day=1)  # Start of month
    
    # Split the data
    training_df = df[df['Date'] < validation_start].copy()
    validation_df = df[df['Date'] >= validation_start].copy()
    
    return training_df, validation_df

def split_track_streaming_for_backtesting(df):
    """
    Split the track streaming data into training and validation sets,
    rounding to whole months.
    
    Args:
        df: DataFrame with 'Date' and 'Value' columns
        
    Returns:
        tuple: (training_df, validation_df) or (None, None) if data is insufficient
    """
    # Convert Date to datetime if it's not already
    df['Date'] = pd.to_datetime(df['Date'], format='%b %d, %Y')
    
    # Sort by date
    df = df.sort_values('Date')
    
    # Get the first and last dates
    start_date = df['Date'].min()
    end_date = df['Date'].max()
    
    # Round start date to beginning of month and end date to end of month
    start_date = start_date.replace(day=1)
    end_date = (end_date.replace(day=1) + pd.DateOffset(months=1) - pd.DateOffset(days=1))
    
    # Calculate total months of data
    total_months = ((end_date.year - start_date.year) * 12 + end_date.month - start_date.month)
    
    # Check if we have minimum required data (27 months)
    if total_months < 27:
        return None, None
        
    # Calculate the cutoff date for validation (exactly 24 months from the end)
    validation_start = end_date - pd.DateOffset(months=24)
    validation_start = validation_start.replace(day=1)  # Start of month
    
    # Split the data
    training_df = df[df['Date'] < validation_start].copy()
    validation_df = df[df['Date'] >= validation_start].copy()
    
    return training_df, validation_df

with tab3:
    st.title("Backtest Model Accuracy")
    
    # First file upload - Artist Listener History
    uploaded_file = st.file_uploader("Artist Listener History", type="csv", key="spotify_streams_tab3")
    
    # Second file upload - Historical Track Streaming Data
    uploaded_files_unique = st.file_uploader("Historical Track Streaming Data", type=['csv'], accept_multiple_files=True, key="track_data_tab3")
    
    # Process the data only if both types of files are uploaded
    if uploaded_file is not None and uploaded_files_unique:
        # Process Artist Listener History
        listener_df = pd.read_csv(uploaded_file)
        
        # Basic data validation for listener history
        if 'Date' not in listener_df.columns or 'Monthly Listeners' not in listener_df.columns:
            st.error("The Artist Listener History file must contain 'Date' and 'Monthly Listeners' columns.")
            st.stop()
            
        # Split listener history data
        listener_train_df, listener_val_df = split_listener_history_for_backtesting(listener_df)
        
        if listener_train_df is None:
            st.error("Insufficient data in Artist Listener History for backtesting. Need at least 27 months of historical data.")
            st.stop()

        # Calculate mldr from listener history data
        listener_train_df['Date'] = pd.to_datetime(listener_train_df['Date'])
        listener_train_df['YearMonth'] = listener_train_df['Date'].dt.strftime('%Y-%m')
        monthly_listeners = listener_train_df.groupby('YearMonth').agg({
            'Monthly Listeners': 'mean',
            'Date': 'first'
        }).reset_index()
        
        # Calculate months since first date
        min_date = monthly_listeners['Date'].min()
        monthly_listeners['Months'] = monthly_listeners['Date'].apply(
            lambda x: (x.year - min_date.year) * 12 + x.month - min_date.month
        )
        
        # Fit exponential decay model
        x_data = monthly_listeners['Months']
        y_data = monthly_listeners['Monthly Listeners']
        popt, _ = curve_fit(exponential_decay, x_data, y_data, p0=(max(y_data), 0.1))
        mldr = popt[1]  # This is the decay rate
        
        # Display the Artist MLDR prominently
        st.write("## Artist MLDR (Decay Rate)")
        st.info(f"**Artist MLDR: {mldr:.6f}**")
        st.write("This exponential decay rate will be used in the model calculations.")
            
        # Display information about the listener history split
        st.write("Artist Listener History Data Split:")
        st.write(f"Training period: {listener_train_df['Date'].min().strftime('%Y-%m-%d')} to {listener_train_df['Date'].max().strftime('%Y-%m-%d')}")
        st.write(f"Validation period: {listener_val_df['Date'].min().strftime('%Y-%m-%d')} to {listener_val_df['Date'].max().strftime('%Y-%m-%d')}")
        
        # Process Track Streaming Data
        track_data_splits = []
        track_summary_list = []  # New list for track summaries
        
        for file in uploaded_files_unique:
            track_df = pd.read_csv(file)
            
            # Basic data validation for track streaming data
            if 'Date' not in track_df.columns or 'Value' not in track_df.columns:
                st.error(f"File {file.name} must contain 'Date' and 'Value' columns.")
                continue
            
            # Calculate streaming metrics as in tab2
            total_value = track_df['Value'].iloc[-1]  # Total historical streams
            streams_last_month = total_value - track_df['Value'].iloc[-31] if len(track_df) > 30 else total_value
            streams_3_months = total_value - track_df['Value'].iloc[-91] if len(track_df) > 90 else total_value
            streams_12_months = total_value - track_df['Value'].iloc[-366] if len(track_df) > 365 else total_value
            release_date = pd.to_datetime(track_df['Date'].iloc[0]).strftime('%d/%m/%Y')
            
            track_name = file.name.split(' - ')[1].strip()
            
            # Add to summary list
            track_summary_list.append({
                'Track': track_name,
                'streams_last_month': streams_last_month,
                'total_streams_3_months': streams_3_months,
                'total_streams_12_months': streams_12_months,
                'historical': total_value,
                'release_date': release_date
            })
                
            # Split track streaming data
            track_train_df, track_val_df = split_track_streaming_for_backtesting(track_df)
            
            if track_train_df is None:
                st.warning(f"Insufficient data in {file.name} for backtesting. Skipping this track.")
                continue
            
            track_data_splits.append({
                'track_name': track_name,
                'training_df': track_train_df,
                'validation_df': track_val_df
            })
        
        if not track_data_splits:
            st.error("No valid track data available for backtesting.")
            st.stop()
        
        # Display track summary data
        st.write("\nData Preview:")
        track_summary_df = pd.DataFrame(track_summary_list)
        st.write(track_summary_df)
        
        # Display track data splits information
        st.write("\nTrack Streaming Data Splits:")
        for track_split in track_data_splits:
            st.write(f"\nTrack: {track_split['track_name']}")
            st.write(f"Training period: {track_split['training_df']['Date'].min().strftime('%Y-%m-%d')} to {track_split['training_df']['Date'].max().strftime('%Y-%m-%d')}")
            st.write(f"Validation period: {track_split['validation_df']['Date'].min().strftime('%Y-%m-%d')} to {track_split['validation_df']['Date'].max().strftime('%Y-%m-%d')}")
        
        if st.button("Run Backtest"):
            st.write("Running backtest analysis...")
            
            track_metrics_list = []
            
            for track_split in track_data_splits:
                st.write(f"\nAnalyzing track: {track_split['track_name']}")
                
                # Prepare training data for forecasting
                training_df = track_split['training_df']
                validation_df = track_split['validation_df']
                
                # Calculate key metrics from training data
                # Get the last month's streams (new streams, not cumulative)
                if len(training_df) > 30:
                    last_month_streams = training_df['Value'].iloc[-1] - training_df['Value'].iloc[-31]
                else:
                    last_month_streams = training_df['Value'].iloc[-1]

                # Calculate 3-month streams (new streams)
                if len(training_df) > 90:
                    last_3_months = training_df['Value'].iloc[-1] - training_df['Value'].iloc[-91]
                else:
                    last_3_months = training_df['Value'].iloc[-1]

                # Calculate 12-month streams (new streams)
                if len(training_df) > 365:
                    last_12_months = training_df['Value'].iloc[-1] - training_df['Value'].iloc[-366]
                else:
                    last_12_months = training_df['Value'].iloc[-1]

                historical_total = training_df['Value'].iloc[-1]  # This should still be cumulative
                release_date = training_df['Date'].min()

                # Create forecast input data structure
                forecast_input = pd.DataFrame({
                    'Track': [track_split['track_name']],
                    'streams_last_month': [last_month_streams],
                    'total_streams_3_months': [last_3_months],
                    'total_streams_12_months': [last_12_months],
                    'historical': [historical_total],
                    'release_date': [release_date.strftime('%d/%m/%Y')]
                })

                # Prepare monthly actual values from validation data
                validation_df['Date'] = pd.to_datetime(validation_df['Date'])
                validation_df['YearMonth'] = validation_df['Date'].dt.strftime('%Y-%m')
                monthly_actual = validation_df.groupby('YearMonth').agg({
                    'Value': lambda x: x.iloc[-1] - x.iloc[0],  # Calculate new streams in each month
                    'Date': 'first'
                }).reset_index()

                # Generate forecasts using training data
                sp_playlist_reach = 1000  # Default value as in tab2
                current_date = training_df['Date'].max()
                
                # Update fitted parameters based on playlist reach
                updated_fitted_params_df = update_fitted_params(fitted_params_df, sp_playlist_reach, sp_range, SP_REACH)
                
                # Calculate months since release for forecasting
                months_since_release_total = (current_date - release_date).days // 30
                
                # Calculate monthly averages
                monthly_avg_3_months = (last_3_months - last_month_streams) / (2 if months_since_release_total > 2 else 1)
                monthly_avg_last_month = last_month_streams
                monthly_avg_12_months = (last_12_months - last_3_months) / (9 if months_since_release_total > 11 else (months_since_release_total - 3))
                
                # Prepare data for decay rate calculation
                months_since_release = np.array([
                    max((months_since_release_total - 11), 0),
                    max((months_since_release_total - 2), 0),
                    months_since_release_total - 0
                ])
                monthly_averages = np.array([monthly_avg_12_months, monthly_avg_3_months, monthly_avg_last_month])
                
                # Calculate decay rates
                params = fit_segment(months_since_release, monthly_averages)
                S0, k = params
                
                # Generate forecasts for validation period
                validation_months = len(validation_df)
                consolidated_df = pd.DataFrame({
                    'segment': range(1, len(breakpoints)),
                    'k': [k] * (len(breakpoints) - 1)  # Using the calculated decay rate for all segments
                })
                
                forecasts = forecast_values(consolidated_df, last_month_streams, months_since_release_total, validation_months)
                forecast_df = pd.DataFrame(forecasts)
                
                # Convert month numbers to actual dates starting from the end of training period
                last_training_date = training_df['Date'].max()
                forecast_df['Date'] = [last_training_date + pd.DateOffset(months=i+1) for i in range(len(forecast_df))]
                
                # Prepare monthly actual values
                validation_df['Date'] = pd.to_datetime(validation_df['Date'])
                validation_df['YearMonth'] = validation_df['Date'].dt.strftime('%Y-%m')
                monthly_actual = validation_df.groupby('YearMonth').agg({
                    'Value': lambda x: x.iloc[-1] - x.iloc[0],  # Calculate new streams in each month
                    'Date': 'first'
                }).reset_index()

                # Generate forecasts for validation period
                validation_months = len(validation_df['Date'].dt.to_period('M').unique())
                
                # Calculate months since release for all periods
                months_since_release_all = list(range(1, 500))
                decay_rate_list = []

                # Get initial decay rates for each month
                for month in months_since_release_all:
                    for i in range(len(breakpoints) - 1):
                        if breakpoints[i] <= month < breakpoints[i + 1]:
                            decay_rate = updated_fitted_params_df.loc[i, 'k']
                            decay_rate_list.append(decay_rate)
                            break

                # Create final_df with initial decay rates
                final_df = pd.DataFrame({
                    'months_since_release': months_since_release_all,
                    'decay_rate': decay_rate_list
                })

                # Add mldr to relevant months
                start_month = min(months_since_release)
                end_month = max(months_since_release)
                final_df.loc[(final_df['months_since_release'] >= start_month) & 
                            (final_df['months_since_release'] <= end_month), 'mldr'] = mldr

                # Calculate percent change and adjust decay rates
                final_df['percent_change'] = ((final_df['mldr'] - final_df['decay_rate']) / final_df['decay_rate']) * 100
                average_percent_change = final_df['percent_change'].mean()
                
                if average_percent_change > 0:
                    weight = min(1, max(0, average_percent_change / 100))
                else:
                    weight = 0
                    
                final_df['adjusted_decay_rate'] = final_df['decay_rate'] * (1 + (average_percent_change * weight) / 100)
                
                # Add new decay rate and calculate final adjustment
                final_df.loc[(final_df['months_since_release'] >= start_month) & 
                            (final_df['months_since_release'] <= end_month), 'new_decay_rate'] = k

                final_df['percent_change_new_vs_adjusted'] = ((final_df['new_decay_rate'] - final_df['adjusted_decay_rate']) / final_df['adjusted_decay_rate']) * 100
                average_percent_change_new_vs_adjusted = final_df['percent_change_new_vs_adjusted'].mean()

                weight_new = 1 if average_percent_change_new_vs_adjusted > 0 else 0
                final_df['final_adjusted_decay_rate'] = final_df['adjusted_decay_rate'] * (1 + (average_percent_change_new_vs_adjusted * weight_new) / 100)

                # Calculate decay rates using the same method as tab2
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

                # Generate forecasts using the same method as tab2
                forecasts = forecast_values(consolidated_df, last_month_streams, months_since_release_total, validation_months)
                forecast_df = pd.DataFrame(forecasts)
                
                # Convert month numbers to actual dates
                last_training_date = training_df['Date'].max()
                forecast_df['Date'] = [last_training_date + pd.DateOffset(months=i+1) for i in range(len(forecast_df))]
                forecast_df['YearMonth'] = forecast_df['Date'].dt.strftime('%Y-%m')
                forecast_df['forecasted_value'] = forecast_df['forecasted_value']  # These are already monthly values

                # Ensure we have matching months between actual and predicted
                monthly_actual = monthly_actual.set_index('YearMonth')
                monthly_predicted = forecast_df.set_index('YearMonth')

                # Get common months
                common_months = monthly_actual.index.intersection(monthly_predicted.index)

                # Filter both DataFrames to only include common months
                monthly_actual = monthly_actual.loc[common_months].reset_index()
                monthly_predicted = monthly_predicted.loc[common_months].reset_index()

                # Calculate accuracy metrics using monthly data
                actual_values = monthly_actual['Value'].values
                predicted_values = monthly_predicted['forecasted_value'].values
                
                # MAPE calculation
                mape = np.mean(np.abs((actual_values - predicted_values) / actual_values)) * 100
                
                # MAE calculation
                mae = np.mean(np.abs(actual_values - predicted_values))
                
                # R-squared calculation
                ss_res = np.sum((actual_values - predicted_values) ** 2)
                ss_tot = np.sum((actual_values - np.mean(actual_values)) ** 2)
                r2 = 1 - (ss_res / ss_tot)
                
                # Monthly variance calculation
                monthly_variance = np.std(np.abs((actual_values - predicted_values) / actual_values)) * 100
                
                # Display results
                st.write("\nAccuracy Metrics:")
                st.write(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
                st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
                st.write(f"R-squared Value: {r2:.4f}")
                st.write(f"Monthly Variance: {monthly_variance:.2f}%")
                
                # Create visualization of actual vs predicted values (monthly)
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(monthly_actual['Date'], monthly_actual['Value'], 
                       label='Actual', color='blue', marker='o')
                ax.plot(monthly_predicted['Date'], monthly_predicted['forecasted_value'], 
                       label='Predicted', color='red', linestyle='--', marker='o')
                ax.set_title(f'Monthly Actual vs Predicted Streams - {track_split["track_name"]}')
                ax.set_xlabel('Month')
                ax.set_ylabel('Monthly Streams')
                ax.legend()
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Create monthly comparison table
                comparison_df = pd.DataFrame({
                    'Month': monthly_actual['Date'],
                    'Actual Monthly Streams': monthly_actual['Value'],
                    'Predicted Monthly Streams': monthly_predicted['forecasted_value']
                })
                comparison_df['Absolute Error'] = np.abs(comparison_df['Actual Monthly Streams'] - comparison_df['Predicted Monthly Streams'])
                comparison_df['Percentage Error'] = (comparison_df['Absolute Error'] / comparison_df['Actual Monthly Streams']) * 100

                # Calculate all accuracy metrics
                mape = np.mean(np.abs((actual_values - predicted_values) / actual_values)) * 100
                mae = np.mean(np.abs(actual_values - predicted_values))
                r2 = r2_score(actual_values, predicted_values)
                monthly_variance = np.std(np.abs((actual_values - predicted_values) / actual_values)) * 100
                
                # Store track-specific metrics (all numeric, no formatting)
                track_metrics = {
                    'Track': track_split['track_name'],
                    'MAPE': mape,  # Store raw numeric value
                    'MAE': mae,
                    'R-squared': r2,
                    'Monthly Variance': monthly_variance,
                    'Mean Monthly Streams (Actual)': np.mean(actual_values),
                    'Mean Monthly Streams (Predicted)': np.mean(predicted_values),
                    'Total Streams (Actual)': np.sum(actual_values),
                    'Total Streams (Predicted)': np.sum(predicted_values)
                }
                
                # Add metrics to list for overall calculations
                track_metrics_list.append(track_metrics)
                
                # Create formatted version for display only
                display_metrics = {
                    'Track': track_split['track_name'],
                    'MAPE (%)': f"{mape:.2f}%",
                    'MAE': f"{int(mae):,}",
                    'R-squared': f"{r2:.4f}",
                    'Monthly Variance (%)': f"{monthly_variance:.2f}%",
                    'Mean Monthly Streams (Actual)': f"{int(np.mean(actual_values)):,}",
                    'Mean Monthly Streams (Predicted)': f"{int(np.mean(predicted_values)):,}",
                    'Total Streams (Actual)': f"{int(np.sum(actual_values)):,}",
                    'Total Streams (Predicted)': f"{int(np.sum(predicted_values)):,}"
                }

                # Display track-specific metrics
                st.write(f"\nAccuracy Metrics for {track_split['track_name']}:")
                metrics_df = pd.DataFrame([display_metrics]).set_index('Track')
                st.write(metrics_df)

                # Format numbers with commas in comparison table
                comparison_df['Actual Monthly Streams'] = comparison_df['Actual Monthly Streams'].apply(lambda x: f"{int(x):,}")
                comparison_df['Predicted Monthly Streams'] = comparison_df['Predicted Monthly Streams'].apply(lambda x: f"{int(x):,}")
                comparison_df['Absolute Error'] = comparison_df['Absolute Error'].apply(lambda x: f"{int(x):,}")
                comparison_df['Percentage Error'] = comparison_df['Percentage Error'].apply(lambda x: f"{x:.2f}%")
                
                # Display monthly comparison
                st.write("\nMonthly Comparison:")
                comparison_df.set_index('Month', inplace=True)
                st.write(comparison_df)
                
                # Create visualization
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
                
                # Plot 1: Actual vs Predicted
                ax1.plot(monthly_actual['Date'], actual_values, 
                        label='Actual', color='blue', marker='o')
                ax1.plot(monthly_predicted['Date'], predicted_values, 
                        label='Predicted', color='red', linestyle='--', marker='o')
                ax1.set_title(f'Monthly Actual vs Predicted Streams - {track_split["track_name"]}')
                ax1.set_xlabel('Month')
                ax1.set_ylabel('Monthly Streams')
                ax1.legend()
                plt.setp(ax1.xaxis.get_ticklabels(), rotation=45)
                
                # Plot 2: Percentage Error Over Time
                percentage_errors = (np.abs(actual_values - predicted_values) / actual_values) * 100
                ax2.plot(monthly_actual['Date'], percentage_errors, 
                        color='purple', marker='o')
                ax2.set_title('Prediction Error Over Time')
                ax2.set_xlabel('Month')
                ax2.set_ylabel('Percentage Error (%)')
                plt.setp(ax2.xaxis.get_ticklabels(), rotation=45)
                
                plt.tight_layout()
                st.pyplot(fig)

            # After processing all tracks, calculate and display overall metrics
            if track_metrics_list:
                st.write("\nOverall Model Performance:")
                overall_metrics = pd.DataFrame(track_metrics_list)
                
                overall_summary = pd.DataFrame({
                    'Metric': [
                        'Average MAPE',
                        'Average MAE',
                        'Average R-squared',
                        'Average Monthly Variance',
                        'Number of Tracks Analyzed'
                    ],
                    'Value': [
                        f"{overall_metrics['MAPE'].mean():.2f}%",  # Using new column name
                        f"{int(overall_metrics['MAE'].mean()):,}",
                        f"{overall_metrics['R-squared'].mean():.4f}",
                        f"{overall_metrics['Monthly Variance'].mean():.2f}%",  # Using new column name
                        f"{len(track_metrics_list)}"
                    ]
                })
                st.write(overall_summary)
