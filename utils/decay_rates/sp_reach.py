"""
This module contains the SP_REACH data, which defines decay rates for different streaming
volume ranges across various time segments. This data is crucial for predicting how tracks
will perform over time based on their current streaming volume.

The data structure consists of:
- Unnamed: 0: Time segment identifier (1-12)
- Columns 1-10: Streaming volume ranges (matching the ranges defined in volume_ranges.py)
- Values: Decay rates for each combination of time segment and streaming volume

Time segments represent different periods in a track's lifecycle:
1. Months 1-2
2. Months 3-5
3. Months 6-8
4. Months 9-11
5. Months 12-14
6. Months 15-17
7. Months 18-20
8. Months 21-23
9. Months 24-26
10. Months 27-35
11. Months 36-47
12. Months 48+

The decay rates in this matrix are empirically derived from historical streaming data
and represent how quickly streams decay in each time segment for tracks in different
streaming volume ranges. Higher decay rates indicate faster decline in streams.
"""

import pandas as pd

# Define SP_REACH data
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

# Create DataFrame from SP_REACH data
SP_REACH = pd.DataFrame(SP_REACH_DATA) 