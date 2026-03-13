"""
## Compute Entropy for a Node

Given class labels from a decision tree node, compute the entropy using the stable logarithm formula.

Entropy is a fundamental concept from information theory that measures the amount of uncertainty or randomness in a dataset. In decision trees, it's used as a splitting criterion to build trees that maximize information gain.

### Entropy Formula:

$$H(S)=-\sum_{i=1}^{C}p_i\cdot \log_{2}(p_i)$$

Where $p_i$ is the proportion of samples belonging to class $i$, and $C$ is the number of classes. By convention, $0\log_{2}(0)=0$

### Function Arguments

- `y: array-like` - Class labels for samples in the node

### Examples

Input: y=[1,1,1,1]
Output: 0.0

Input: y=[0,1,0,1]
Output: 1.0

###  Hints

- Hint 1: Use `np.unique()` with `return_counts=True` to get class frequencies.
- Hint 2: Filter out zero probabilities before computing logarithms to avoid numerical issues.
- Hint 3: Use `np.log2()` for base-2 logarithms in the entropy formula.

### Requirements

- Return single float value ≥ 0
- Handle empty nodes (return 0 entropy)
- Use stable logarithm computation (avoid log(0))
- Support multi-class problems
- Use base-2 logarithms for interpretability

### Constraints

- Total samples `≤ 1e6`; NumPy only
- Time limit: 100ms; Memory: 64MB
"""

import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    y = np.asarray(y)

    # Handle empty node
    if y.size == 0:
        return 0.0

    # Get class counts
    _, counts = np.unique(y, return_counts=True)

    # Convert to probabilities
    p = counts / counts.sum()

    # Remove zero probabilities for numerical stability
    p = p[p > 0]

    # Compute entropy
    entropy = -np.sum(p * np.log2(p))

    return float(entropy)