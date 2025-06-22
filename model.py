import torch.nn as nn
from torch.optim import Adam

class RNAseqClassifier(nn.Module):
    def __init__(self, input_size, layer_config, output_size):
        """
        Parameters:
        - input_size (int): Number of input features (genes)
        - layer_config (list[dict]): List of layer definitions. Each dict should have:
            - 'size' (int): Hidden layer size
            - 'batchnorm' (bool): Whether to apply BatchNorm
            - 'dropout' (float or None): Dropout rate
        - output_size (int): Number of output classes
        """
        super().__init__()

        layers = []
        in_dim = input_size

        for cfg in layer_config:
            out_dim = cfg["size"]
            layers.append(nn.Linear(in_dim, out_dim))

            if cfg.get("batchnorm", False):
                layers.append(nn.BatchNorm1d(out_dim))

            layers.append(nn.ReLU())

            if cfg.get("dropout") is not None:
                layers.append(nn.Dropout(cfg["dropout"]))

            in_dim = out_dim

        # Final classification layer
        layers.append(nn.Linear(in_dim, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def get_model_and_optimizer(
    input_size, 
    output_size, 
    layer_config=None, 
    lr=1e-4, 
    weight_decay=1e-5
):
    if layer_config is None:
        layer_config = [
            {"size": 512, "batchnorm": True, "dropout": 0.5},
            {"size": 256, "batchnorm": True, "dropout": 0.5},
            {"size": 128, "batchnorm": True, "dropout": None}
        ]

    model = RNAseqClassifier(
        input_size=input_size,
        layer_config=layer_config,
        output_size=output_size
    )
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    return model, optimizer
