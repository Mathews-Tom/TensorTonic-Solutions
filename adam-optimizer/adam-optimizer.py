import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    # Make it robust to python lists / scalars
    param_arr = np.asarray(param, dtype=float)
    grad_arr  = np.asarray(grad, dtype=float)
    m_arr     = np.asarray(m, dtype=float)
    v_arr     = np.asarray(v, dtype=float)

    # Update biased first/second moments
    m_new = beta1 * m_arr + (1.0 - beta1) * grad_arr
    v_new = beta2 * v_arr + (1.0 - beta2) * (grad_arr ** 2)

    # Bias correction
    m_hat = m_new / (1.0 - (beta1 ** t))
    v_hat = v_new / (1.0 - (beta2 ** t))

    # Parameter update
    param_new = param_arr - lr * m_hat / (np.sqrt(v_hat) + eps)

    # If inputs were plain scalars, return Python floats for convenience
    if np.isscalar(param) and np.isscalar(grad) and np.isscalar(m) and np.isscalar(v):
        return float(param_new), float(m_new), float(v_new)

    return param_new, m_new, v_new