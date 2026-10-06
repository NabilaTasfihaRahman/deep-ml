import torch

def relu(t):
    """Element-wise ReLU: max(0, t).

    Args:
        t (torch.Tensor): input tensor

    Returns:
        torch.Tensor: activated tensor
    """
    # TODO: implement with pure torch ops
    output=torch.clamp(t,min=0)
    return torch.tensor(output)

def leaky_relu(t, slope=0.01):
    """Element-wise Leaky ReLU with given negative slope.

    Args:
        t (torch.Tensor): input tensor
        slope (float): slope for negative values

    Returns:
        torch.Tensor: activated tensor
    """
    # TODO: implement with pure torch ops
    new_t=t*slope
    output=torch.where(t>0,t,new_t)
    return output
