# Track Forecasting Pipeline

This directory contains scripts for generating and managing track stream forecasts using historical data stored in the database.

## Key Files

- `forecast_streams.py`: Main script that connects to the database, retrieves training data, generates forecasts, and stores results
- `test_forecast.py`: Testing script for validating the forecasting process on a single track

## Database Tables

The scripts interact with the following database tables:

1. **backtest_track_daily_training_data**: Source of historical track streaming data
   - `id`: Unique identifier
   - `cm_track_id`: ChartMetric track identifier
   - `cm_artist_id`: ChartMetric artist identifier
   - `date`: Date of the streaming data
   - `daily_streams`: Number of streams on that date
   - `days_from_release`: Days since the track was released
   - `created_at`: Timestamp when the record was created

2. **backtest_artist_mldr**: Source of artist Monthly Listener Decay Rate values
   - `id`: Unique identifier
   - `cm_artist_id`: ChartMetric artist identifier
   - `mldr`: Monthly Listener Decay Rate value
   - `created_at`: Timestamp when the record was created
   - `backtest_artist_daily_training_data_id`: Reference to training data

3. **backtest_track_streams_forecast**: Destination for forecast results
   - `id`: Unique identifier
   - `month`: Month number in the forecast (1, 2, 3, etc.)
   - `forecasted_value`: Predicted number of streams for that month
   - `segment_used`: Lifecycle segment used for this forecast (1, 2, 3, etc.)
   - `time_used`: Time period used in forecast calculation
   - `cm_track_id`: ChartMetric track identifier
   - `cm_artist_id`: ChartMetric artist identifier
   - `training_data_id`: Reference to the training data used
   - `created_at`: Timestamp when the forecast was created

## Forecasting Process

The forecasting process follows these steps:

1. Connect to the database and retrieve tracks that need forecasting
2. For each track:
   - Retrieve the track's daily streaming data
   - Calculate key metrics (last 30 days streams, months since release)
   - Get the artist's MLDR value from the artist table
   - Generate decay parameters and apply adjustments
   - Calculate segmented decay rates for different lifecycle phases
   - Generate stream forecasts for each future month
   - Store the forecasts in the database

## Usage

### Running the Main Forecasting Script

```bash
# From the project root directory
python backtest_pipeline/track_forecasts/forecast_streams.py
```

This will process all tracks that don't yet have forecasts in the database.

### Testing a Single Track

```bash
# From the project root directory
python backtest_pipeline/track_forecasts/test_forecast.py
```

This will:
1. Find a track with available data that hasn't been forecasted yet
2. Generate forecasts for that track
3. Output diagnostic information
4. Save the forecasts to a CSV file for inspection

## Notes

- The script uses the same forecasting algorithm as the Streamlit app, ensuring consistency
- Artist MLDR values are critical for accurate forecasting - tracks without an artist MLDR will be skipped
- The forecasting period is set to `DEFAULT_TRACK_STREAMS_FORECAST_PERIOD` (typically 400 months)
- Track decay rates are segmented based on `track_lifecycle_segment_boundaries` from the decay_rates module
