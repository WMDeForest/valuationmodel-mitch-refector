# Backtest SQL Scripts

This folder contains SQL scripts used for creating and managing backtest datasets that are used to validate the performance of predictive models.

## Scripts

### [backtest_artist_training_data_creation.sql](backtest_artist_training_data_creation.sql)
Creates the training dataset for backtesting artist monthly listeners models. This script:
- Identifies artists with sufficient data in both training and validation periods
- Extracts pre-April 2023 data for qualified artists
- Populates the `backtest_artist_daily_training_data` table with this historical data

### [track_training_data_creation.sql](track_training_data_creation.sql)
Creates the training dataset for backtesting track daily streams models. This script:
- Identifies tracks with sufficient data in both training and validation periods
- Filters tracks belonging to batch 2
- Extracts pre-April 2023 data for qualified tracks
- Populates the `backtest_track_daily_training_data` table with this historical data

## Dataset Structure

The backtest datasets typically contain:
- Historical data (prior to April 1, 2023) used for training prediction models
- Validation data (April 1, 2023 to March 31, 2025) used to evaluate model performance

### Table Schemas

**backtest_artist_daily_training_data**
| Column | Description |
|--------|-------------|
| id | Primary key, auto-incrementing |
| artist_id | Reference to the artist |
| sp_streams_date | Date of the streaming data point |
| monthly_listeners | Count of monthly listeners for that date |
| created_at | Timestamp when the record was created |

**backtest_track_daily_training_data**
| Column | Description |
|--------|-------------|
| id | Primary key, auto-incrementing (implicit) |
| cm_track_id | Reference to the track |
| cm_artist_id | Reference to the artist |
| date | Date of the streaming data point |
| daily_streams | Count of daily streams for that date |
| days_from_release | Number of days since the track was released |

### Current Dataset Statistics:

**Artist Data:**
- 1656 unique artists in the training dataset
- Reduced from 2212 artists in the source data
- Reduction due to filtering out artists without sufficient training data (90+ days) or validation data (730+ days)

**Track Data:**
- 9,974 unique tracks in the training dataset
- Reduced from 24,188 tracks in the source data
- Reduction due to filtering out tracks without sufficient training data (90+ days) or validation data (730+ days)

## Usage

These scripts are primarily for data preparation and should be run before model training and evaluation. After running these scripts, the prepared data can be used with the analytical and predictive components in the main application. 