import torch

def activation_derivatives(x: float) -> dict[str, float]:
    """
    Compute the derivatives of Sigmoid, Tanh, and ReLU at a given point x
    using PyTorch autograd.
    
    Args:
        x: Input value
        
    Returns:
        Dictionary with keys 'sigmoid', 'tanh', 'relu' and their derivative values
    """
    # Your code here - use autograd!
    # Hint: Create tensors with requires_grad=True, apply activation, call .backward()
    x=torch.tensor(x,requires_grad=True)
    sigmoid=torch.sigmoid(x)
    tanh=torch.tanh(x)
    relu=torch.relu(x)
    sigmoid.backward()
    sigmoid_grad=x.grad.clone()
    x.grad.zero_()
    tanh.backward()
    tanh_grad=x.grad.clone()
    x.grad.zero_()
    relu.backward()
    relu_grad=x.grad.clone()
    x.grad.zero_()
    return {'sigmoid': sigmoid_grad.item(),
    'tanh':tanh_grad.item(),
    'relu':relu_grad.item()}