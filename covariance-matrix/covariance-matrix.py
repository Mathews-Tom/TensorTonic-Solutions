"""
## Compute Covariance Matrix

Compute the covariance matrix from a dataset without using np.cov. The covariance matrix shows how features vary together and is fundamental to many ML algorithms.

### Step 1: Center the Data

$$\mu =\text{mean}(X,\text{axis}=0)$$
$$X_{centered} =X-\mu$$

### Step 2: Compute Covariance Matrix

$$\Sigma = \frac{1}{N-1}X_{centered}^TX_{centered}$$
 
Where: X has shape (N, D), μ has shape (D,), Σ has shape (D, D)

### Function Arguments

- X: list[list[float]] | np.ndarray - Dataset with shape (N, D)

### Examples

Input: X=[[1, 2], [2, 3], [3, 4]]
Output: [[1.0, 1.0], [1.0, 1.0]]

Input: X=[[1, 0], [0, 1]]
Output: [[0.5, -0.5], [-0.5, 0.5]]

Input: X=[[1, 2, 3]]
Output: None (only 1 sample)

### Hints

Hint 1: Use np.asarray() to convert input and check shape with .shape and .ndim. Use np.mean() to compute feature means.
Hint 2: Center data by subtracting mean for matrix multiplication.
Hint 3: Divide by (N-1) for sample covariance and handle edge cases by returning None.

### Requirements

- Return np.ndarray of shape (D, D) with covariance values
- Return None for invalid input (N < 2 or not 2D)
- Must be vectorized (no loops over data points)
- Cannot use np.cov function
- Use sample covariance (divide by N-1, not N)

### Constraints

- Dataset size: N ≤ 10,000, D ≤ 1,000
- Numerical precision: relative tolerance ≤ 1e-8
- Libraries: NumPy only
"""

import numpy as np

def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X.
    """
    X = np.asarray(X, dtype=float)

    # Validate input
    if X.ndim != 2:
        return None

    N, D = X.shape
    if N < 2:
        return None

    # Step 1: center the data
    mu = np.mean(X, axis=0)
    X_centered = X - mu

    # Step 2: compute covariance matrix
    cov = (X_centered.T @ X_centered) / (N - 1)

    return cov