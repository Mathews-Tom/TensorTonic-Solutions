"""
## Learning Rate Scheduler (Linear Decay)

Implement a linear learning rate schedule with optional warmup.

The learning rate starts at 0, increases linearly to an initial rate (η₀) over the warmup steps, then decays linearly toward a final rate (ηf) until total training steps are completed. After the total steps, the learning rate remains fixed at ηf.

### Mathematical Definition

$$LR(t)=\left\{\begin{matrix}\frac{t\cdot\eta_0}{W} & \text{if}\:\:t<W  \\ \eta_f+(\eta_0-\eta_f)\cdot\frac{T-t}{T-W} & \text{if}\:\:W\le t\le T \\ \eta_f & \text{if}\:\:t>T \\ \end{matrix}\right.$$ 
 
where $t$: current step (0-based), $W$: warmup steps, $T$: total steps, $\eta_0$: initial learning rate, $\eta_f$: final learning rate.

### Examples

Input: step=0, total_steps=100, initial_lr=1e-3, final_lr=0.0, warmup_steps=10
Output: 0.0

Input: step=10, total_steps=100, initial_lr=1e-3, final_lr=0.0, warmup_steps=10
Output: 0.001

Input: step=50, total_steps=100, initial_lr=1e-3, final_lr=0.0, warmup_steps=10
Output: 0.00055

### Hints

- Hint 1: Handle each phase separately with conditional statements based on the current step.
- Hint 2: Use linear interpolation between start and end values for both warmup and decay phases.

### Constraints

- Scalar computation (no vectorization required)
- Must handle zero warmup and steps beyond total_steps
- Return a single float value
- Time limit: 100 ms
"""

def linear_lr(step, total_steps, initial_lr, final_lr=0.0, warmup_steps=0) -> float:
    """
    Linear warmup (0→initial_lr) then linear decay (initial_lr→final_lr).
    Steps are 0-based; clamp at final_lr after total_steps.
    """

    # After training finished
    if step > total_steps:
        return float(final_lr)

    # Warmup phase
    if warmup_steps > 0 and step < warmup_steps:
        return float(step * initial_lr / warmup_steps)

    # If no decay phase (edge case)
    if total_steps == warmup_steps:
        return float(final_lr)

    # Linear decay phase
    if step <= total_steps:
        return float(
            final_lr + (initial_lr - final_lr) * (total_steps - step) / (total_steps - warmup_steps)
        )

    return float(final_lr)