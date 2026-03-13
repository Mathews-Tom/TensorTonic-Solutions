"""
## Impute Missing Values (mean/median)

Fill NaN values in each feature column using either column mean or column median.

Missing value imputation is a fundamental data preprocessing step that replaces NaN (Not a Number) values with meaningful estimates. Your function should compute the mean or median for each column using only the observed (non-NaN) values, then fill all NaN positions in that column with the computed statistic.

### Function Arguments

- `X: array-like, shape (N, D)` - Data with possible np.nan
- `strategy: 'mean' or 'median'` - Imputation method

### Examples

Input: X=[[1,nan],[3,5]], strategy='mean'
Output: [[1,5],[3,5]]

Input: X=[[nan,2],[nan,4]], strategy='median'
Output: [[0,2],[0,4]] (all-NaN col → 0)

Input: X=[1,nan,3,nan,5], strategy='mean'
Output: [1,3,3,3,5] (1D case)

### Hints

- Hint 1: Use `np.isnan()` to find NaN positions, then `np.logical_not()` for valid values.
- Hint 2: For 2D arrays, iterate over columns. Use `np.mean()` for observed values.
- Hint 3: Handle all-NaN columns by checking `np.any()` before computing statistics.

### Requirements

- Return NumPy array (N, D), no NaNs if imputable
- Compute statistic per column on observed values only
- Leave fully-NaN columns as-is or fill with 0 (fill with 0)
- Do not change non-NaN values
- Return a copy, don't modify input
- Handle integer inputs by upcasting to float

### Constraints

- Handle integer inputs by upcasting to float
- NumPy only; time limit: 300ms
"""

import numpy as np

def impute_missing(X, strategy='mean'):
    """
    Fill NaN values in each feature column using column mean or median.
    """
    X = np.asarray(X, dtype=float)
    X_out = X.copy()

    if X_out.ndim == 1:
        mask = np.isnan(X_out)
        if np.all(mask):
            fill = 0.0
        else:
            if strategy == 'mean':
                fill = np.mean(X_out[~mask])
            elif strategy == 'median':
                fill = np.median(X_out[~mask])
            else:
                raise ValueError("strategy must be 'mean' or 'median'")
        X_out[mask] = fill
        return X_out

    elif X_out.ndim == 2:
        N, D = X_out.shape
        for d in range(D):
            col = X_out[:, d]
            mask = np.isnan(col)

            if np.all(mask):
                fill = 0.0
            else:
                if strategy == 'mean':
                    fill = np.mean(col[~mask])
                elif strategy == 'median':
                    fill = np.median(col[~mask])
                else:
                    raise ValueError("strategy must be 'mean' or 'median'")

            col[mask] = fill
            X_out[:, d] = col

        return X_out

    else:
        raise ValueError("Input must be 1D or 2D array")