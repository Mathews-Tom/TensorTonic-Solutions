import numpy as np

def train_gan_step(real_data: np.ndarray, generator, discriminator, noise_dim: int) -> dict:
    """
    Perform one training step for GAN.
    Computes discriminator loss and generator loss for the batch.

    Supports:
    - generator as callable(z, output_dim) OR object with .generate(z) / .forward(z)
    - discriminator as callable(x) OR object that stores a callable under common fields/methods,
      and/or requires an extra arg like training=False.
    """
    if not isinstance(real_data, np.ndarray) or real_data.ndim != 2:
        raise ValueError("real_data must be a 2D numpy array (batch_size, data_dim)")
    if not isinstance(noise_dim, int) or noise_dim <= 0:
        raise ValueError("noise_dim must be a positive integer")

    batch_size, data_dim = real_data.shape
    z = np.random.randn(batch_size, noise_dim)

    # ---------------- Generator ----------------
    if callable(generator):
        fake_data = generator(z, data_dim)
    elif hasattr(generator, "generate") and callable(getattr(generator, "generate")):
        # Many mocks: generate(z) -> fake batch
        fake_data = generator.generate(z)
    elif hasattr(generator, "forward") and callable(getattr(generator, "forward")):
        fake_data = generator.forward(z)
    else:
        raise AttributeError("generator must be callable or have a .generate(z) / .forward(z) method")

    # -------------- Discriminator resolver --------------
    def _try_call(fn, x):
        """Try common call signatures."""
        for args in ((x,), (x, False), (x, True), (x, None)):
            try:
                return fn(*args)
            except TypeError:
                continue
        return None

    def _resolve_callable(obj):
        """Find an underlying callable on obj."""
        # If it's directly callable, done.
        if callable(obj):
            return obj

        # Common method names first
        method_names = [
            "discriminator", "discriminate", "predict", "classify", "score",
            "evaluate", "eval", "run", "infer", "inference", "forward",
            "call", "__call__"
        ]
        for name in method_names:
            if hasattr(obj, name):
                fn = getattr(obj, name)
                if callable(fn):
                    return fn

        # Common fields that may hold the callable
        field_names = ["model", "net", "network", "fn", "func", "function", "clf", "estimator"]
        for name in field_names:
            if hasattr(obj, name):
                fn = getattr(obj, name)
                if callable(fn):
                    return fn

        # If it's a dict/tuple/list holding a callable, unwrap
        if isinstance(obj, dict):
            for v in obj.values():
                if callable(v):
                    return v
        if isinstance(obj, (list, tuple)):
            for v in obj:
                if callable(v):
                    return v

        # Last resort: scan attributes for *any* callable
        for name in dir(obj):
            if name.startswith("_"):
                continue
            fn = getattr(obj, name, None)
            if callable(fn):
                return fn

        return None

    def d_probs(x):
        fn = _resolve_callable(discriminator)
        if fn is None:
            raise AttributeError("Could not resolve a callable from discriminator.")
        out = _try_call(fn, x)
        if out is None:
            raise TypeError("Resolved discriminator callable, but could not call it with common signatures.")
        return out

    real_probs = d_probs(real_data)
    fake_probs = d_probs(fake_data)

    # Flatten to (N,)
    real_probs = np.asarray(real_probs).reshape(-1)
    fake_probs = np.asarray(fake_probs).reshape(-1)

    # Clip for numerical stability
    eps = 1e-8
    real_probs = np.clip(real_probs, eps, 1.0 - eps)
    fake_probs = np.clip(fake_probs, eps, 1.0 - eps)

    # Losses
    d_loss = -np.mean(np.log(real_probs) + np.log(1.0 - fake_probs))
    g_loss = -np.mean(np.log(fake_probs))  # non-saturating

    return {"d_loss": float(d_loss), "g_loss": float(g_loss)}