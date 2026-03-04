import numpy as np

def make_vgg_config(variant: str) -> list:
    """
    Return the layer configuration for a VGG variant (vgg11, vgg13, vgg16, vgg19).
    Integers are conv filter counts, 'M' is max-pool.
    """
    v = variant.lower().strip()

    configs = {
        "vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
        "vgg13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
        "vgg16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
        "vgg19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M",
                  512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
    }

    if v not in configs:
        raise ValueError(f"Unknown VGG variant '{variant}'. Choose from: {list(configs.keys())}")

    return configs[v]