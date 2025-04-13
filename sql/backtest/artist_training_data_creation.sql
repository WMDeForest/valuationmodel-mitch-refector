-- =========================================================================
-- BACKTEST ARTIST TRAINING DATA CREATION
-- =========================================================================
-- Purpose: Creates a training dataset for artist monthly listeners backtest
--          by selecting historical data for qualified artists
-- =========================================================================

-- TABLE SCHEMA
-- backtest_artist_daily_training_data
-- -------------------------------------------------------------------------
-- id                  - Primary key, auto-incrementing
-- cm_artist_id        - Reference to the artist
-- sp_streams_date     - Date of the streaming data point
-- monthly_listeners   - Count of monthly listeners for that date
-- created_at          - Timestamp when the record was created
-- -------------------------------------------------------------------------

-- This script creates training data for backtesting artist performance models.
-- It identifies artists with sufficient historical and validation period data,
-- then extracts their pre-April 2023 data to create a training dataset.

-- Selection criteria:
-- 1. Artists must have at least 90 days of data before April 1, 2023 (training period)
-- 2. Artists must have at least 730 days of data between April 1, 2023 and March 31, 2025 (validation period)

-- For qualified artists, we copy:
--   - cm_artist_id
--   - sp_streams_date
--   - monthly_listeners
-- But only for dates before April 1, 2023 (the training period)

INSERT INTO backtest_artist_daily_training_data (cm_artist_id, sp_streams_date, monthly_listeners)
SELECT
  cm_artist_id,
  sp_streams_date,
  monthly_listeners
FROM artist_monthly_listeners_history
WHERE
  sp_streams_date < '2023-04-01'
  AND cm_artist_id IN (
    SELECT cm_artist_id
    FROM artist_monthly_listeners_history
    GROUP BY cm_artist_id
    HAVING
      COUNT(*) FILTER (WHERE sp_streams_date < '2023-04-01') >= 90 AND
      COUNT(*) FILTER (WHERE sp_streams_date BETWEEN '2023-04-01' AND '2025-03-31') >= 730
  ); 

-- =========================================================================
-- RESULTS SUMMARY
-- =========================================================================
-- Initial filtering:
-- After applying the data validation criteria, the initial training dataset
-- contained 1,656 unique artists, down from 2,212 in the source data.
-- This reduction occurred due to eliminating artists without sufficient
-- training data (90+ days) or validation data (730+ days).
--
-- Data consistency:
-- After removal of artists without corresponding tracks in the track training
-- dataset, the final count is 1,565 unique artists.
-- This ensures that all artists in the dataset have associated track data
-- available for forecasting models. 