import numpy as np

def apply_homogeneous_transform(T, points):
    """
    Apply 4x4 homogeneous transform T to 3D point(s).
    """
    T = np.array(T, dtype=float)

    P = np.array(points, dtype=float)
    single_point = (P.ndim == 1)
    if single_point:
        P = P.reshape(1, 3)          # (1, 3)

    # Convert to homogeneous: (N, 4) by appending 1s
    ones = np.ones((P.shape[0], 1), dtype=float)
    Ph = np.hstack([P, ones])        # (N, 4)

    # Apply transform: (N, 4) -> (N, 4)
    Ph_out = (T @ Ph.T).T

    # Extract spatial part
    out = Ph_out[:, :3]

    return out.reshape(3).tolist() if single_point else out.tolist()
