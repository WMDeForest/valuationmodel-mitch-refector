"""
Archived calculate_graph function from streamlit_app.py.
This function was previously used in tab1 for generating forecasts and visualizations of song streaming data.
It processes song data, calculates decay rates, generates streaming forecasts, and creates visualizations 
of revenue projections across different countries and time periods.
This code is not active in the current application and is preserved for reference only.
"""

def calculate_graph(df, discount_rate, selected_songs):
    song_forecasts = []
    weights_and_changes = []

    years_plot = []
    export_forecasts = pd.DataFrame()

    for selected_song in selected_songs:
        song_data = df[df['Track'] == selected_song].iloc[0]

        value = stream_influence_factor
        total_streams_12_months = song_data['total_streams_12_months']
        total_streams_3_months = song_data['total_streams_3_months']
        streams_last_month = song_data['streams_last_month']
        historical = song_data['historical']
        release_date = song_data['release_date']

        updated_fitted_params_df = update_fitted_params(fitted_params_df, stream_influence_factor, sp_range, SP_REACH)
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