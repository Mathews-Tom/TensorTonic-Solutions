"""
## Implement Nadam (Nesterov + Adam)

Implement one update step of Nadam optimizer. Given current parameters, moments, and gradients, return updated parameters and moments using Nesterov-accelerated Adam.

### Step 1: Update First Moment

$$m_t=\beta_1m_{t-1}+(1-\beta_1)g_t$$

### Step 2: Update Second Moment

$$v_t=\beta_2v_t+1(1-\beta_2)g_t^2$$

### Step 3: Nesterov-Adjusted Update

$$w_t=w_{t-1}-\eta\cdot\frac{\beta_1m_t+(1-\beta_1)g_t}{\sqrt{v_t}+\epsilon}$$

Where: w = parameters, m = first moment, v = second moment, g = gradients, η = learning rate, β₁,β₂ = decay rates

Function Arguments
- `w: np.ndarray` - Current parameters (any shape)
- `m: np.ndarray` - First moment estimates (same shape as w)
- `v: np.ndarray` - Second moment estimates (same shape as w)
- `grad: np.ndarray` - Current gradients (same shape as w)
- `lr: float = 0.002` - Learning rate
- `beta1: float = 0.9` - First moment decay rate
- `beta2: float = 0.999` - Second moment decay rate
- `eps: float = 1e-8` - Small constant for numerical stability

### Examples

Input: w=[1.0, -1.0], m=[0.1, -0.1], v=[0.01, 0.01], grad=[0.2, -0.3], lr=0.002
Output: ([0.998, -0.997], [0.11, -0.12], [0.01003, 0.01008])
Nesterov acceleration combines current and updated momentum for parameter update

Input: w=[1.0, 2.0], m=[0.1, 0.2], v=[0.01, 0.04], grad=[0.0, 0.0], lr=0.002
Output: ([0.998, 1.998], [0.09, 0.18], [0.00999, 0.03996])
Zero gradient: moments decay, parameter update uses existing momentum

Input: w=[1.0, 2.0], m=[0.0, 0.0], v=[0.0, 0.0], grad=[0.1, 0.2], lr=0.002
Output: ([0.988, 1.988], [0.01, 0.02], [0.00001, 0.00004])
First step: moments initialized, Nesterov uses both new momentum and gradient

### Hints

Hint 1: Convert inputs to NumPy arrays first. Update moments using exponential moving averages like in Adam.

Hint 2: The Nesterov adjustment uses: beta1 * m_new + (1 - beta1) * grad in the numerator instead of just m_new.

### Requirements

- Return tuple (new_w, new_m, new_v) with same shapes as inputs
- Use the exact update formulas above (no bias correction)
- Vectorized implementation only (no Python loops)
- Handle any array shape (1D, 2D, etc.)

### Constraints

- 0 < beta1, beta2 < 1
- lr > 0, eps > 0
- Libraries: NumPy only
"""

import numpy as np

def nadam_step(w, m, v, grad, lr=0.002, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    Perform one Nadam update step.
    """
    w = np.asarray(w, dtype=float)
    m = np.asarray(m, dtype=float)
    v = np.asarray(v, dtype=float)
    grad = np.asarray(grad, dtype=float)

    # Step 1: update first moment
    new_m = beta1 * m + (1.0 - beta1) * grad

    # Step 2: update second moment
    new_v = beta2 * v + (1.0 - beta2) * (grad ** 2)

    # Step 3: Nesterov-adjusted update
    nesterov_m = beta1 * new_m + (1.0 - beta1) * grad
    new_w = w - lr * nesterov_m / (np.sqrt(new_v) + eps)

    return new_w, new_m, new_v