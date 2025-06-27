# Add this near the top with other type aliases
#represents the number of points to show in the graph and it's min and max (going from 2h to 4h)
DEFAULT_POINTS = 24
MIN_POINTS = 24
MAX_POINTS = 48

# Number of points (equivalent to hours) to subtract for prediction area
# 12 points = 1 hour (assuming 5-minute intervals)
PREDICTION_HOUR_OFFSET = 12

DOUBLE_CLICK_THRESHOLD: int = 500  # milliseconds