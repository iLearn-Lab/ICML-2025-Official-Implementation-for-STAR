import torch.nn as nn


class MLPProj(nn.Module):
    """Project low-dimensional observations into the shared embedding space."""

    def __init__(self, input_size, output_size, hidden_size=None, num_layers=1, dropout=0.0):
        super().__init__()
        if num_layers < 1:
            raise ValueError("MLPProj requires num_layers >= 1.")
        if num_layers > 1 and hidden_size is None:
            hidden_size = output_size
        sizes = [input_size] + [hidden_size] * (num_layers - 1) + [output_size]
        layers = []
        for i in range(num_layers - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(sizes[-2], sizes[-1]))
        self.projection = nn.Sequential(*layers)
        self.out_channels = output_size

    def forward(self, data):
        return self.projection(data)
