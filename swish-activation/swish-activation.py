"""
## Implement Swish Activation

Implement the Swish activation function. Swish is a smooth, learnable activation function that often improves performance over ReLU due to smoother gradient flow.

### Swish Formula:

$$\begin{align} \text{Swish}(x)&=x\cdot\sigma(x), \\ \text{where}\; \sigma(x)&=\frac{1}{1+e^{-x}} \end{align}$$

### Function Arguments

`x` - Input (scalar, list, or NumPy array)

### Examples

Input: [0, 1, -1, 3]
Output: [0.0, 0.731, -0.269, 2.857]
Smooth activation with both positive and negative outputs

Input: 0.0
Output: [0.0]
Scalar input returns 1D array with shape (1)

Input: [[1, -1], [2, -2]]
Output: [[0.731, -0.269], [1.762, -0.238]]
Works element-wise on multi-dimensional arrays

### Hints

Hint 1: First implement sigmoid: 1 / (1 + np.exp(-x)), then multiply by x.

Hint 2: For numerical stability, clip extreme values before computing exponential to prevent overflow.

Hint 3: Use np.asarray() to handle different input types consistently.

### Requirements

- Return np.ndarray of floats
- Implement sigmoid yourself (do not use scipy.special.expit)
- Vectorized implementation only (no loops)
- Ensure numerical stability in sigmoid
- Preserve input shape

### Constraints

- Use NumPy only
- Time limit: 300ms; Memory ≤ 64MB
"""

import numpy as np

def swish(x):
    """
    Implement Swish activation function.
    """
    x = np.asarray(x, dtype=float)

    # numerical stability for exp
    x_clip = np.clip(x, -500, 500)

    sigmoid = 1.0 / (1.0 + np.exp(-x_clip))
    return x * sigmoid