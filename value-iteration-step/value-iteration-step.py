import numpy as np

def value_iteration_step(values, transitions, rewards, gamma):
    """
    Perform one step of value iteration and return updated values.
    """
    V = np.array(values, dtype=np.float64)
    T = np.array(transitions, dtype=np.float64)  # (S, A, S')
    R = np.array(rewards, dtype=np.float64)       # (S, A)

    # Q(s,a) = R(s,a) + gamma * T(s,a,:) · V
    Q = R + gamma * (T @ V)  # (S, A, S') @ (S',) -> (S, A)

    return Q.max(axis=1).tolist()