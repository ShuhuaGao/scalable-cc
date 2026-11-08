"""Policy network architectures shared by training and evaluation."""

from torch import nn


def network(outputs: int) -> nn.Sequential:
    """Create a two-hidden-layer network.

    :param outputs: Number of output units.
    :return: Neural network.
    """
    return nn.Sequential(
        nn.Linear(21, 128),
        nn.GELU(),
        nn.Linear(128, 128),
        nn.GELU(),
        nn.Linear(128, outputs),
    )


def imitation_network() -> nn.Sequential:
    """Create the bounded imitation policy.

    :return: Neural network with sigmoid output.
    """
    return nn.Sequential(*list(network(1).children()), nn.Sigmoid())
