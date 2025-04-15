# Music Streaming Decay Models

## Overview

This directory contains the core components of our music streaming valuation model, specifically focused on modeling, measuring, and forecasting the decay patterns of streaming over time. This is crucial for accurate financial valuations of music assets.

The model's fundamental insight is that music streaming typically follows exponential decay patterns that can be modeled mathematically. By analyzing historical streaming data, we can extract decay parameters that allow us to generate accurate forecasts of future streams.

## Key Concepts

### Exponential Decay

The mathematical foundation of our model is the exponential decay function:

```
S(t) = S₀ * e^(-kt)
```

Where:
- `S(t)` is the number of streams at time t
- `S₀` is the initial number of streams
- `k` is the decay rate (higher values = faster decay)
- `t` is time (typically measured in months since release)

### MLDR (Music Listener Decay Rate)

The MLDR is a key metric calculated from an artist's historical streaming patterns. It represents the overall rate at which an artist's music typically decays in popularity. This metric helps tailor track-specific decay rates to match the artist's general audience retention characteristics.

### Segmented Decay

Our model recognizes that decay rates aren't constant throughout a track's lifetime. Instead, we use different decay rates for different time periods:
- Early period (months 1-3): Often rapid decay as initial promotion ends
- Middle period (months 4-12): Moderate decay as the track settles
- Later periods (months 13+): Slower decay as the track finds its long-term audience

### Stream Influence Factor Adjustment (formerly Playlist Reach)

Tracks with significant external influence factors typically show slower decay rates. Our model adjusts the basic decay rates based on a track's stream influence factor (formerly called playlist reach), providing more accurate forecasts for tracks with stronger external streaming influences.

## Module Structure

### `core.py`

Contains the fundamental mathematical decay functions:
- `piecewise_exp_decay`: Core exponential decay function used in forecasting
- `exponential_decay`: Simpler version used in initial fitting

### `preprocessing.py`

Handles data cleaning and preparation:
- `remove_anomalies`: Identifies and corrects outliers in streaming data using IQR method

### `fitting.py`

Extracts decay parameters from historical data:
- `calculate_decay_rate`: Estimates the overall MLDR from artist streaming history
- `fit_segment`: Fits decay parameters to specific time segments of streaming data

### `parameter_updates.py`

Adjusts decay parameters based on external factors:
- `update_fitted_params`: Modifies decay rates based on Spotify playlist reach

### `forecasting.py`

Generates future streaming predictions:
- `forecast_track_streams`: Creates month-by-month projections using segmented decay rates

## Data Flow & Process

Here's how the components work together in a typical analysis:

1. **Data Cleaning**
   - Raw streaming data is processed with `remove_anomalies` to handle outliers
   - A 4-week moving average is applied to smooth out weekly fluctuations
   - Data is resampled to monthly totals for more stable analysis

2. **Artist-Level Analysis**
   - The overall MLDR is calculated with `calculate_decay_rate`
   - This provides a baseline understanding of the artist's listener retention

3. **Track-Specific Modeling**
   - Track streaming history is segmented into time periods
   - Decay parameters for each segment are estimated with `fit_segment`
   - These parameters are adjusted based on Spotify playlist reach using `update_fitted_params`
   - The artist's MLDR influences the final decay rates used

4. **Forecasting**
   - Using the adjusted decay parameters, `forecast_track_streams` generates streams forecasts
   - Different decay rates are applied to different future time periods
   - Monthly stream forecasts cascade, with each month's output becoming the starting point for the next

5. **Financial Valuation**
   - Stream forecasts are converted to revenue projections (in the main app)
   - Adjustments for ownership percentages and rights distributions are applied
   - Final valuation includes discounted cash flow calculations

## Key Parameters

- **Breakpoints**: Define the boundaries between different time segments (e.g., months 1-3, 4-12, 13-36, etc.)
- **Fitted Parameters**: Base decay rates for each time segment
- **SP_REACH**: Adjustment factors for different levels of stream influence (formerly playlist inclusion)
- **Discount Rate**: Used in financial calculations to account for time value of money
- **Stream Influence Factor**: Numeric value (default 1000) that influences decay rate adjustments

## Example Usage

```python
# Simplified example workflow
from utils.decay_models import remove_anomalies, calculate_decay_rate, update_fitted_params, forecast_track_streams

# 1. Clean the data
clean_data = remove_anomalies(raw_streaming_data)

# 2. Calculate artist-level decay rate
mldr, _ = calculate_decay_rate(clean_data)

# 3. Update decay parameters based on stream influence factor
adjusted_params = update_fitted_params(fitted_params_df, stream_influence_factor, sp_range, SP_REACH)

# 4. Generate forecasts
forecasts = forecast_track_streams(adjusted_params, current_streams, months_since_release, forecast_periods)
```

## Future Enhancements

Potential areas for model improvement:
- Machine learning approaches to decay modeling
- Genre-specific decay patterns
- Seasonal adjustment factors
- Integration of marketing activity data
- Improved handling of catalog revival events

## New High-Level API

For most use cases, the new high-level API function `analyze_listener_decay` (formerly `calculate_monthly_listener_decay`) is recommended:

```python
from utils.decay_models import analyze_listener_decay

# Load data from any source (CSV, API, database)
# DataFrame must have 'Date' and 'Monthly Listeners' columns
df = load_data_from_source()

# Process everything in one call
results = analyze_listener_decay(df, sample_rate=7)  # Weekly sampling

# Access results
mldr = results['mldr']  # The Monthly Listener Decay Rate
popt = results['popt']  # Fitted parameters for the decay function
plot_df = results['subset_df']  # Processed data (with anomalies removed)
min_date = results['min_date']  # Minimum date in dataset (for UI)
max_date = results['max_date']  # Maximum date in dataset (for UI)
```

This function handles the entire workflow including:
1. Data sampling
2. Data sorting 
3. Anomaly detection and removal
4. Date range filtering
5. Decay rate calculation

It's designed to work seamlessly with data from any source, making it ideal for integration with APIs, databases, or CSV uploads. 