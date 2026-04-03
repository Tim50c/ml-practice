import numpy as np

# Dot product
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
dot_product = np.dot(v1, v2)

# Matrix operations
A = np.random.randn(3, 3)
A_inv = np.linalg.inv(A)    # Inverse of A
det = np.linalg.det(A)  # Determinant of A
eigenvalues, eigenvectors = np.linalg.eig(A) # Eigendecomposition
print(A)
print(A_inv)
print(det)
print(eigenvalues)
print(eigenvectors)

# Solving linear system (Ax = b)
A = np.array([[3, 1], [1, 2]])
b = np.array([9, 8])
x = np.linalg.solve(A, b)   # Solution to Ax = b
print(x)

# Norms
A = np.array([3, 4])
l2_norm = np.linalg.norm(A)
print(l2_norm)
l1_norm = np.linalg.norm(A, 1)
print(l1_norm)