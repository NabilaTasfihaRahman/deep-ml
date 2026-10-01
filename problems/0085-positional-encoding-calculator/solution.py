import torch

def pos_encoding(position: int, d_model: int):
    """
    Compute positional encodings for Transformer models.

    Args:
        position: sequence length (number of positions)
        d_model: model dimensionality

    Returns:
        torch.Tensor of shape (position, d_model) with dtype float16,
        or -1 if position == 0 or d_model <= 0.
    """
    if position==0 or d_model<=0:
        return -1
    angle=torch.zeros(position,d_model,dtype=torch.float16)
    for i in range(angle.shape[0]):
        for j in range(angle.shape[1]):
            if j%2 ==0:
                angle[i][j]=torch.sin(torch.tensor((i/10000**(j/d_model))))
            else:
                angle[i][j]=torch.cos(torch.tensor((i/10000**((j-1)/d_model))))
    return angle