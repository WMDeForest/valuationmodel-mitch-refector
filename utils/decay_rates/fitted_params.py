"""
This module contains the fitted parameters and breakpoints used in the exponential decay model
for predicting streaming patterns. These parameters were derived from fitting exponential decay
curves to historical streaming data across different time segments.

The module contains two main components:

1. Fitted Parameters:
   - Each segment has two parameters:
     * S0: Initial value (starting point for the decay curve)
     * k: Decay rate (how quickly streams decline)
   - These parameters are stored in a DataFrame for easy access and manipulation

2. Breakpoints:
   - Define the boundaries between different time segments
   - Used to determine which decay parameters to apply at different points in a track's lifecycle
   - The breakpoints array defines the start of each segment:
     * Segment 1: Months 1-2
     * Segment 2: Months 3-5
     * Segment 3: Months 6-8
     * Segment 4: Months 9-11
     * Segment 5: Months 12-14
     * Segment 6: Months 15-17
     * Segment 7: Months 18-20
     * Segment 8: Months 21-23
     * Segment 9: Months 24-26
     * Segment 10: Months 27-35
     * Segment 11: Months 36-47
     * Segment 12: Months 48+

These parameters are used in conjunction with the SP_REACH data to create a more accurate
prediction model that accounts for both time-based decay patterns and streaming volume effects.
"""

import pandas as pd

# Define fitted parameters
fitted_params = [
    {'segment': 1, 'S0': 7239.425562317985, 'k': 0.06741191851584262},
    {'segment': 2, 'S0': 6465.440296195081, 'k': 0.03291507714354558},
    {'segment': 3, 'S0': 6478.639247351713, 'k': 0.03334620907608441},
    {'segment': 4, 'S0': 5755.53795902042, 'k': 0.021404012549575913},
    {'segment': 5, 'S0': 6023.220319977014, 'k': 0.02461834982301452},
    {'segment': 6, 'S0': 6712.835052107982, 'k': 0.03183160108111365},
    {'segment': 7, 'S0': 6371.457552382675, 'k': 0.029059156192761115},
    {'segment': 8, 'S0': 5954.231622567404, 'k': 0.02577913683190864},
    {'segment': 9, 'S0': 4932.65240022657, 'k': 0.017941231431835854},
    {'segment': 10, 'S0': 3936.0657447490344, 'k': 0.009790878919164516},
    {'segment': 11, 'S0': 4947.555706076349, 'k': 0.016324033736761206},
    {'segment': 12, 'S0': 4000, 'k': 0.0092302}
]

# Create DataFrame from fitted parameters
fitted_params_df = pd.DataFrame(fitted_params)

# Define breakpoints for segments
breakpoints = [1, 3, 6, 9, 12, 15, 18, 21, 24, 27, 36, 48, 100000] 