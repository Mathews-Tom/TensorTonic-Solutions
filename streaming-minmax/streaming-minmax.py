"""
### Streaming Min-Max Normalization

Implement streaming min-max normalization: given multiple batches of data, update per-feature running min and max, and normalize each incoming batch.

Streaming normalization processes data in real-time without knowing the full dataset statistics upfront. Your implementation should maintain running minimum and maximum values for each feature, updating them as new batches arrive, then normalize each batch using the current global statistics.

### Normalization Formula:

$$x'=\frac{x-\text{min}}{\text{max}-\text{min}+\epsilon}$$

### Function Arguments

- `D: int` - Number of features
- `state: dict` - Contains 'min' and 'max' arrays (shape D,)
- `X_batch: array-like, shape (B, D)` - Input batch
- `eps: float` - Small value to avoid division by zero

### Examples

Input: streaming_minmax_init(D=2), streaming_minmax_update(state, [[1,3],[2,1]])
Output: streaming_minmax_init({'min': [inf,inf], 'max': [-inf,-inf]}), streaming_minmax_update([[0,1],[1,0]])

Input: streaming_minmax_init(D=1), streaming_minmax_update(state, [[5],[3]])
Output: streaming_minmax_init({'min': [inf], 'max': [-inf]}), streaming_minmax_update([[1],[0]])

### Hints

- Hint 1: Initialize with `np.full()` for min and max arrays.
- Hint 2: Use `np.minimum()` and `np.maximum()` to update running statistics.
- Hint 3: Use `np.maximum()` to handle constant features safely with eps.

### Requirements

- On init: min=+inf, max=-inf
- On update: state['min']=min(state['min'], batch_min); same for max
- Normalize using the updated min/max (post-update) for consistency
- Handle constant columns (range≈0) via eps
- Pure NumPy implementation

### Constraints

- Handles constant columns (range≈0) via eps
- Batches can arrive in any order/size
- NumPy only; time limit: 300ms
"""

import numpy as np

def streaming_minmax_init(D):
    """
    Initialize state dict with min, max arrays of shape (D,).
    """
    return {
        "min": np.full(D, np.inf, dtype=float),
        "max": np.full(D, -np.inf, dtype=float),
    }


def streaming_minmax_update(state, X_batch, eps=1e-8):
    """
    Update state's min/max with X_batch, return normalized batch.
    """
    X_batch = np.asarray(X_batch, dtype=float)

    # Compute batch statistics
    batch_min = np.min(X_batch, axis=0)
    batch_max = np.max(X_batch, axis=0)

    # Update running statistics
    state["min"] = np.minimum(state["min"], batch_min)
    state["max"] = np.maximum(state["max"], batch_max)

    # Compute safe denominator
    denom = np.maximum(state["max"] - state["min"], eps)

    # Normalize batch using updated statistics
    X_norm = (X_batch - state["min"]) / denom

    return X_norm