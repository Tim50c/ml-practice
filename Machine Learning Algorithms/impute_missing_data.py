import numpy as np
from typing import Literal


def impute_missing_data(data: np.ndarray, strategy: Literal['mean', 'median', 'mode'] = 'mean') -> np.ndarray:
    """
    Impute missing values in a 2D array using the specified strategy.

    Args:
        data: 2D numpy array with missing values represented as np.nan
        strategy: Imputation strategy - 'mean', 'median', or 'mode'

    Returns:
        A new 2D numpy array with missing values imputed.

    Raises:
        ValueError: If data is not a 2D array or if an invalid strategy is provided.
        TypeError: If the input data type cannot support NaN values.
    """
    # 1. Input Validation
    if not isinstance(data, np.ndarray):
        raise TypeError("Input 'data' must be a numpy ndarray.")

    if data.ndim != 2:
        raise ValueError(f"Expected a 2D array, but got shape {data.shape}")

    if not np.issubdtype(data.dtype, np.floating):
        raise TypeError("Input array must be of a floating-point type to support np.nan values.")

    valid_strategies = {'mean', 'median', 'mode'}
    if strategy not in valid_strategies:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from {valid_strategies}.")

    # 2. Defensive Copying (Avoid side effects)
    imputed_data = data.copy()

    # Fast exit if there are no missing values
    nan_mask = np.isnan(imputed_data)
    if not np.any(nan_mask):
        return imputed_data

    # 3. Vectorized Metric Calculation
    if strategy == 'mean':
        # Calculates column-wise means ignoring NaNs
        replace_vals = np.nanmean(imputed_data, axis=0)

    elif strategy == 'median':
        # Calculates column-wise medians ignoring NaNs
        replace_vals = np.nanmedian(imputed_data, axis=0)

    elif strategy == 'mode':
        # NumPy doesn't have a built-in nanmode, so we vectorize column-wise efficiently
        replace_vals = np.empty(imputed_data.shape[1])
        for col_idx in range(imputed_data.shape[1]):
            col = imputed_data[:, col_idx]
            valid_elements = col[~np.isnan(col)]

            if valid_elements.size == 0:
                replace_vals[col_idx] = np.nan
            else:
                vals, counts = np.unique(valid_elements, return_counts=True)
                replace_vals[col_idx] = vals[np.argmax(counts)]

    # 4. Handle Edge Case: Columns that are entirely NaN
    if np.any(np.isnan(replace_vals)):
        # Fall back to 0.0 (or you could raise an error depending on business logic)
        replace_vals = np.nan_to_num(replace_vals, nan=0.0)

    # 5. Broad-casted Vectorized Imputation
    # np.where selects from 'replace_vals' where nan_mask is True, otherwise keeps original data
    # We use np.broadcast_to to seamlessly stretch replace_vals to match the 2D shape
    fill_grid = np.broadcast_to(replace_vals, imputed_data.shape)
    imputed_data = np.where(nan_mask, fill_grid, imputed_data)

    return imputed_data


# Test execution
if __name__ == "__main__":
    data = np.array([[1.0, 2.0], [3.0, np.nan], [5.0, 6.0]])
    result = impute_missing_data(data, 'mean')
    print("Imputed Results:")
    print(np.round(result, 4).tolist())
