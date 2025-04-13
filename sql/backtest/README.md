# Backtest SQL Scripts

This folder contains SQL scripts used for creating and managing backtest datasets that are used to validate the performance of predictive models.

## Scripts

### [artist_training_data_creation.sql](artist_training_data_creation.sql)
Creates the training dataset for backtesting artist monthly listeners models. This script:
- Identifies artists with sufficient data in both training and validation periods
- Extracts pre-April 2023 data for qualified artists
- Populates the `artist_daily_training_data` table with this historical data

## Dataset Structure

The backtest datasets typically contain:
- Historical data (prior to April 1, 2023) used for training prediction models
- Validation data (April 1, 2023 to March 31, 2025) used to evaluate model performance

## Usage

These scripts are primarily for data preparation and should be run before model training and evaluation. After running these scripts, the prepared data can be used with the analytical and predictive components in the main application. 