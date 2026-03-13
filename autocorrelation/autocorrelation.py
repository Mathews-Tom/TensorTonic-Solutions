"""
## Autocorrelation

Autocorrelation measures how a time series correlates with a delayed (lagged) version of itself. It reveals repeating patterns, periodicity, and the degree to which past values predict future values. Autocorrelation at lag k tells you how similar the series is to itself shifted by k time steps.

Given a time series and a maximum lag, compute the autocorrelation for each lag from 0 to max_lag.

### Algorithm

1. Compute the mean and total variance of the series:

$$\bar{x}=\frac{1}{n}\sum_{t=0}^{n-1}x[t],\:\:\gamma_0=\sum_{t=0}^{n-1}(x[t]-\bar{x})^2$$

2. For each lag k, compute the autocovariance and normalize by the total variance:

$$r_k=\frac{\sum_{t=0}^{n-k-1}(x[t]-\bar{x})(x[t+k]-\bar{x})}{\gamma_0}$$

Note that r_0 = 1 always (a series perfectly correlates with itself at lag 0).

### Examples

Input: series = [1, 2, 3, 4, 5], max_lag = 2
Output: [1.0, 0.4, -0.1]

Lag 0 is always 1.0. The linear trend produces positive correlation at lag 1 (0.4) and slight negative correlation at lag 2 (-0.1).

Input: series = [1, -1, 1, -1, 1, -1], max_lag = 2
Output: [1.0, -0.8333, 0.6667]

An alternating series shows strong negative autocorrelation at lag 1 (adjacent values are opposite) and positive autocorrelation at lag 2 (values two steps apart are the same sign).

### Hints
- Hint 1: First compute the mean: sum(series)/n. Then compute the total variance: sum of (x[t] - mean)^2 for all t. For each lag k, sum the products (x[t] - mean) * (x[t+k] - mean) for t from 0 to n-k-1, and divide by the total variance.
- Hint 2: Handle the edge case where variance is 0 (constant series) by returning [1.0] + [0.0] * max_lag. For the general case, use nested loops: outer loop over lags, inner loop to compute the sum of cross-products.

### Requirements

- Compute autocorrelation for each lag from 0 to max_lag inclusive
- Subtract the mean from values before computing covariances
- Normalize by the total variance (autocovariance at lag 0)
- If variance is zero (constant series), return 1.0 for lag 0 and 0.0 for all other lags
- Return a list of floats of length max_lag + 1

### Constraints

- series has at least 2 elements
- 0 <= max_lag < len(series)
- Return a list of floats of length max_lag + 1
- Time limit: 300 ms
"""

def autocorrelation(series, max_lag):
    """
    Compute the autocorrelation of a time series for lags 0 to max_lag.
    """
    n = len(series)
    mean = sum(series) / n

    centered = [x - mean for x in series]
    gamma_0 = sum(x * x for x in centered)

    if gamma_0 == 0:
        return [1.0] + [0.0] * max_lag

    result = []
    for k in range(max_lag + 1):
        cov = 0.0
        for t in range(n - k):
            cov += centered[t] * centered[t + k]
        result.append(cov / gamma_0)

    return result