import torch

def dropout(x: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    # TODO: implement inverted dropout
    if training:
        
        mask=torch.bernoulli(torch.full_like(x,1-p))
        return (mask*x)/(1-p)
    else:
        return x
