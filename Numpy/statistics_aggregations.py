import numpy as np

data = np.random.randn(100, 5)      # 100 samples, 5 features

# Basic stats
print(np.mean(data))    # Mean
print(np.std(data))     # Standard deviation
print(np.var(data))     # Variance
print(np.median(data))      # Median
print(np.max(data), np.min(data))   # Maximum and minimum

# Along a specific axis
print(np.mean(data, axis=0))    # Mean of each column
print(np.mean(data, axis=1))    # Mean of each row

# Sums
print(np.sum(data))
print(np.cumsum(data))  # Cumulative sum: An array of shape = (100, 5) where each element is the sum from the start upto it
