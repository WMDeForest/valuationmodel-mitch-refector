-- =========================================================================
-- BACKTEST TRACK TRAINING DATA CREATION
-- =========================================================================
-- Purpose: Creates a training dataset for track daily streams backtest
--          by selecting historical data for qualified tracks
-- =========================================================================

-- TABLE SCHEMA
-- backtest_track_daily_training_data
-- -------------------------------------------------------------------------
-- id                  - Primary key, auto-incrementing (implicit)
-- cm_track_id         - Reference to the track
-- cm_artist_id        - Reference to the artist
-- date                - Date of the streaming data point
-- daily_streams       - Count of daily streams for that date
-- days_from_release   - Number of days since the track was released
-- -------------------------------------------------------------------------

-- This script creates training data for backtesting track performance models.
-- It identifies tracks with sufficient historical and validation period data,
-- then extracts their pre-April 2023 data to create a training dataset.

-- Selection criteria:
-- 1. Tracks must have at least 90 days of data before April 1, 2023 (training period)
-- 2. Tracks must have at least 730 days of data between April 1, 2023 and March 31, 2025 (validation period)
-- 3. Tracks must belong to batch 2

-- For qualified tracks, we copy:
--   - cm_track_id
--   - cm_artist_id
--   - date
--   - daily_streams
--   - days_from_release
-- But only for dates before April 1, 2023 (the training period)

INSERT INTO backtest_track_daily_training_data (
  cm_track_id, cm_artist_id, date, daily_streams, days_from_release
)
SELECT
  cm_track_id,
  cm_artist_id,
  date,
  daily_streams,
  days_from_release
FROM historical_streams_data
WHERE
  batch_id = 2
  AND date < '2023-04-01'
  AND cm_track_id IN (
    SELECT cm_track_id
    FROM historical_streams_data
    WHERE batch_id = 2
    GROUP BY cm_track_id
    HAVING
      COUNT(*) FILTER (WHERE date < '2023-04-01') >= 90 AND
      COUNT(*) FILTER (WHERE date BETWEEN '2023-04-01' AND '2025-03-31') >= 730
  );

-- =========================================================================
-- RESULTS SUMMARY
-- =========================================================================
-- Initial filtering:
-- After applying the data validation criteria, the initial training dataset
-- contained 9,974 unique tracks, down from 24,188 in the source data.
-- This reduction occurred due to eliminating tracks without sufficient
-- training data (90+ days) or validation data (730+ days).
--
-- Orphaned track cleanup:
-- Additional cleanup was performed to remove orphaned tracks that didn't have
-- corresponding artist data in the artist training dataset. This ensures
-- that when running forecasting models, we have both artist and track data
-- available for all entities.
--
-- Final dataset:
-- 8,318 unique tracks across 1,565 unique artists
-- This represents a further reduction of 1,656 tracks from the initial filtered set. 