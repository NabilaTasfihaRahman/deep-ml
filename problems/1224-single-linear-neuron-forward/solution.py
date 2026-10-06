import torch
import torch.nn as nn


def single_neuron_forward(x):
    """Forward pass of one fixed linear neuron.

    Args:
        x: torch.Tensor of shape (1, 3).

    Returns:
        Python float, the neuron output.
    """
    # TODO: build nn.Linear(3, 1), set fixed weight/bias under no_grad, return float output
    
    with torch.no_grad():
        weight=[[0.5,-0.2,0.3]]
        bias=0.1
        linear=nn.Linear(3,1)
        linear.weight.copy_(torch.tensor([[0.5, -0.2, 0.3]]))
        linear.bias.copy_(torch.tensor(0.1))
    return float(linear(x))
