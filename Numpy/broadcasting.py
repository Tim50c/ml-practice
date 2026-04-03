import numpy as np

'''
Broadcasting Rules:
1. Arrays with fewer dimensions are padded with ones on the left
2. Arrays with size 1 along a dimension are stretched to match the other array
3. If sizes don't match and neither is 1, broadcasting fails
'''

# Add scalar to array
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(a + 10)   # Add 10 to every element

# Add 1D array to 2D array
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
b = np.array([10, 11, 12])
print(a + b)    # Add b to each row of a

# Normalize data
data = np.random.randn(100, 5)
mean = np.mean(data, axis=0)
std = np.std(data, axis=0)
norm_data = (data - mean) / std
print(norm_data)
