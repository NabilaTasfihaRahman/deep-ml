import torch
import torch.nn as nn

def train_neuron(features: torch.Tensor, labels: torch.Tensor, initial_weights: torch.Tensor, initial_bias: float, learning_rate: float, epochs: int) -> tuple[list[float], float, list[float]]:
    """
    Simulates a single neuron with sigmoid activation and trains it using
    backpropagation with MSE loss via SGD.

    Args:
        features: Input feature tensor of shape (n_samples, n_features)
        labels: Binary label tensor of shape (n_samples,)
        initial_weights: Initial weight tensor of shape (n_features,)
        initial_bias: Initial bias scalar
        learning_rate: Learning rate for SGD
        epochs: Number of training epochs

    Returns:
        Tuple of (updated_weights, updated_bias, mse_values) all rounded to 4 decimal places
    """
    mse=[]
    n=features.shape[0]
    labels=labels.unsqueeze(1)
    initial_weights=initial_weights.unsqueeze(1)
    for i in range(epochs):
        z=torch.matmul(features,initial_weights)+initial_bias
        mse.append((torch.sum((torch.sigmoid(z)-labels)**2)/n))
        print(z.shape)
        prediction=torch.sigmoid(z)
        bleh=(prediction-labels)*prediction*(1-prediction)
        weight_grad=(2/n)*((torch.matmul(bleh.T,features)).T)
        bias_grad=(2/n)*torch.sum(bleh)
        updated_weight=initial_weights-learning_rate*weight_grad
        updated_bias=initial_bias-learning_rate*bias_grad
        initial_weights=updated_weight
        initial_bias=updated_bias
    return ([round(x,4) for x in updated_weight.flatten().tolist()],round(updated_bias.item(),4),[round(x.item(),4) for x in mse])
