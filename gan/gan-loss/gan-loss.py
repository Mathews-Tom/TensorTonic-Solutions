import numpy as np

def discriminator_loss(real_probs: np.ndarray, fake_probs: np.ndarray) -> float:
    """
    Compute discriminator loss.

    L_D = -E[ log D(x) + log(1 - D(G(z))) ]
    """
    eps = 1e-8

    real_probs = np.clip(real_probs, eps, 1 - eps)
    fake_probs = np.clip(fake_probs, eps, 1 - eps)

    loss = -np.mean(np.log(real_probs) + np.log(1 - fake_probs))
    return float(loss)


def generator_loss(fake_probs: np.ndarray) -> float:
    """
    Compute generator loss.

    Non-saturating form:
    L_G = -E[ log D(G(z)) ]
    """
    eps = 1e-8

    fake_probs = np.clip(fake_probs, eps, 1 - eps)

    loss = -np.mean(np.log(fake_probs))
    return float(loss)