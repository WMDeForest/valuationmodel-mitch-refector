-- =========================================================================
-- BACKTEST ARTIST TRAINING DATA CREATION
-- =========================================================================
-- Purpose: Creates a training dataset for artist monthly listeners backtest
--          by selecting historical data for qualified artists
-- =========================================================================

-- This script creates training data for backtesting artist performance models.
-- It identifies artists with sufficient historical and validation period data,
-- then extracts their pre-April 2023 data to create a training dataset.

-- Selection criteria:
-- 1. Artists must have at least 90 days of data before April 1, 2023 (training period)
-- 2. Artists must have at least 730 days of data between April 1, 2023 and March 31, 2025 (validation period)

-- For qualified artists, we copy:
--   - artist_id
--   - sp_streams_date
--   - monthly_listeners
-- But only for dates before April 1, 2023 (the training period)

INSERT INTO artist_daily_training_data (artist_id, sp_streams_date, monthly_listeners)
SELECT
  artist_id,
  sp_streams_date,
  monthly_listeners
FROM artist_monthly_listeners_history
WHERE
  sp_streams_date < '2023-04-01'
  AND artist_id IN (
    SELECT artist_id
    FROM artist_monthly_listeners_history
    GROUP BY artist_id
    HAVING
      COUNT(*) FILTER (WHERE sp_streams_date < '2023-04-01') >= 90 AND
      COUNT(*) FILTER (WHERE sp_streams_date BETWEEN '2023-04-01' AND '2025-03-31') >= 730
  ); 