import numpy as np
import wandb


class Logger:
    """Accumulate scalar metrics and periodically log their mean."""

    def __init__(self, log_interval):
        self.log_interval = log_interval
        self.data = None

    def update(self, info, step):
        info = flatten_dict(info)
        if self.data is None:
            self.data = {key: [] for key in info}
        
        for key, value in info.items():
            self.data.setdefault(key, []).append(value)

        if step % self.log_interval == 0:
            means = {key: np.mean(value) for key, value in self.data.items()}
            self.log(means, step)
            self.data = None

    def log(self, info, step):
        wandb.log(flatten_dict(info), step=step)


def flatten_dict(data, prefix=None):
    """Flatten nested metric dictionaries for Weights & Biases logging."""

    flattened = {}
    for key, value in data.items():
        full_key = key if prefix is None else f"{prefix}/{key}"
        if isinstance(value, dict):
            flattened.update(flatten_dict(value, prefix=full_key))
        else:
            flattened[full_key] = value
    return flattened
