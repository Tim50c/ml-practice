import numpy as np
from numpy.ma.extras import hstack

arr = np.arange(12)
reshaped = arr.reshape(3, 4)
flattened = reshaped.flatten()
transposed = np.transpose(reshaped)
transposed_2 = reshaped.T

# Adding/Removing dimensions
expanded = arr[:, np.newaxis, np.newaxis]   # New dimensions: (12, 1, 1)
squeezed = np.squeeze(expanded, 2)  # Remove dimensions of size 1 (axis indexed 2)
print(expanded.shape)
print(squeezed.shape)

# Concatenate
arr1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # dim = (3, 3)
arr2 = np.array([[11, 12, 13], [14, 15, 16], [17, 18, 19]]) # dim = (3, 3)
vstack = np.vstack((arr1, arr2))    # Stack vertically, new dim = (3, 6)
print(vstack)
hstack = np.hstack((arr1, arr2))    # Stack horizontally, new dim = (6, 3)
print(hstack)
concat_v = np.concatenate((arr1, arr2), axis=0) # Same as vstack
print(concat_v)
concat_h = np.concatenate((arr1, arr2), axis=1)   # Same as hstack
print(concat_h)