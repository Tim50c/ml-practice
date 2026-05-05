import numpy as np

# Clip values (prevent extreme values)
a = np.array([-5, 0, 16, -1, 2, 8])
print(np.clip(a,0,5))

# Where (conditional selection)
b = np.array([1, 2, 3, 4, 5, 6, 7])
print(np.where(b > 4, b, 0))

# Unique values
c = np.array([0,1,0,2,3,4,4,3,6,4,1,0,2])
unique = np.unique(c)
print(unique)
values, counts = np.unique(c, return_counts=True)
print(values)
print(counts)

# Sorting
d = np.array([1, 5, 2, 1, 6, 4, 3, 8, 0])
sorted = np.sort(d)
print(sorted)
indices = np.argsort(d)     # Indices that would sort the array
print(indices)

# Random sampling (for train/test split)
np.random.seed(42)      # Reproducibility
shuffled = np.random.permutation(10)     # Shuffled [0, 1, 2,..., 9]
print(shuffled)
choice = np.random.choice(shuffled, 5, replace=False)
print(choice)