import torch

def linear_backward(grad_output, x, W):
    # TODO: return (grad_input, grad_W, grad_b) for y = x @ W.T + b
    grad_input=torch.matmul(grad_output,W)
    grad_W=torch.matmul(grad_output.T,x)
    grad_b=grad_output.sum(axis=0)
    return (grad_input, grad_W, grad_b)

