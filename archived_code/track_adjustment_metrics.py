"""
ARCHIVED CODE - TRACK ADJUSTMENT METRICS

This file documents code that was removed from streamlit_app.py on a code cleanup.

LOCATION OF REMOVED CODE:
The code was removed from streamlit_app.py around line 406-407, within the track valuation 
loop that processes each selected song.

REMOVED CODE:
```python
# Store track adjustment metrics for reporting and quality analysis
track_adjustment_weight = track_adjustment_info['first_adjustment_weight']
track_average_percent_change = track_adjustment_info['first_average_percent_change']
```

CONTEXT:
These variables were storing metrics from the track decay rate adjustment process:
- track_adjustment_weight: A value between 0-1 indicating how much weight was given to 
  observed data when adjusting theoretical decay rates
- track_average_percent_change: The average percentage difference between observed and 
  theoretical decay rates

REASON FOR REMOVAL:
These metrics were stored but never used elsewhere in the application. They were originally 
intended for reporting and quality analysis but those features weren't implemented.

If future development requires analyzing the model's adjustment behavior, these metrics 
could be restored. They're available in the track_adjustment_info dictionary returned 
by the adjust_track_decay_rates() function in utils/decay_models/parameter_updates.py.

FULL CONTEXT OF SURROUNDING CODE:
```python
# Apply a two-stage adjustment using observed artist and track data
adjusted_track_decay_df, track_adjustment_info = adjust_track_decay_rates(
    track_decay_rate_df, 
    track_decay_k=track_decay_k  # Track-specific fitted decay parameter
)

# Store track adjustment metrics for reporting and quality analysis
track_adjustment_weight = track_adjustment_info['first_adjustment_weight']
track_average_percent_change = track_adjustment_info['first_average_percent_change']

# ===== 6. SEGMENT DECAY RATES BY TIME PERIOD =====
# Calculate average decay rates for each segment
segmented_track_decay_rates_df = calculate_track_decay_rates_by_segment(adjusted_track_decay_df, track_lifecycle_segment_boundaries)
```
""" 