import numpy as np

# Element-wise operations (broadcasting)
arr = np.array([1, 2, 3, 4])
print(arr + 10)     # Every element increased by 10
print(arr * 10)     # Every element multiplied by 10
print(arr ** 2)     # Every element squared

# Array operations
arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([5, 6, 7, 8])
print(arr1 + arr2)      # Element by element summing
print(arr1 * arr2)      # Element by element multiplying

# Matrix multiplication
matrix1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
matrix2 = np.array([[2, 4, 8], [6, 1, 3], [7, 5, 9]])
print(matrix1 @ matrix2)    # Matrix multiplication
print(np.dot(matrix1, matrix2))     # Also matrix multiplication
print(np.matmul(matrix1, matrix2))  # Another matrix multiplication
print(matrix1 * matrix2)    # Element-wise multiplication