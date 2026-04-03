import numpy as np

arr = np.arange(10)
print(arr[2:5]) # From index 2 to index 5 (exclude the last)
print(arr[::2]) # Every 2nd element
print(arr[::-1])    # Reverse

# 2D indexing
matrix = np.arange(12).reshape(3, 4)
print(matrix)
print(matrix[1, 3])
print(matrix[0, :]) # First row
print(matrix[:, 1]) # Second column
print(matrix[0:2, 1:3]) # Submatrix

# Boolean indexing
arr = np.array([1, 4, 5, 7, 8, 3, 9, 2, 6])
mask = arr > 5
print(arr[mask])
print(arr[arr > 5])

# Fancy indexing
indices = [0, 2, 4]
print(arr[indices])