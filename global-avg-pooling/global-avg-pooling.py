"""
## Implement Global Average Pooling

Implement Global Average Pooling (GAP) over spatial dimensions for channel-first tensors.

### Shape Transformations

- If input x has shape (C, H, W) → output is shape (C,)
- If input x has shape (N, C, H, W) → output is shape (N, C)

### Mathematical Definition

$$\text{GAP}(x)_c=\frac{1}{HW}\sum_{h=1}^{H}\sum_{w=1}^{W}x_{c,h,w}$$
 
For each channel $c$, average all spatial locations

### Examples

Input: x = np.ones((3, 2, 2))
Output: [1., 1., 1.]
Each channel has all 1s, so average = 1

Input: x = np.array([[[[1,2],[3,4]]]]) (shape (1,1,2,2))
Output: [[2.5]]
(1+2+3+4)/4 = 2.5

### Requirements

- NumPy only. Vectorized (no Python loops over elements)
- Accept (C,H,W) or (N,C,H,W). Raise ValueError otherwise
- Return dtype float (np.float64 is fine), do not modify the input

### Constraints

- H,W ≥ 1; N,C up to a few hundred
- Time ≤ 200 ms; Memory ≤ 64 MB
"""

import numpy as np

def global_avg_pool(x):
    """
    Compute global average pooling over spatial dims.
    Supports (C,H,W) => (C,) and (N,C,H,W) => (N,C).
    """
    x = np.asarray(x, dtype=float)

    if x.ndim == 3:  # (C, H, W)
        return np.mean(x, axis=(1, 2))
    elif x.ndim == 4:  # (N, C, H, W)
        return np.mean(x, axis=(2, 3))
    else:
        raise ValueError("Input must have shape (C,H,W) or (N,C,H,W)")