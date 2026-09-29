import torch

def rnn_forward(input_sequence: list, initial_hidden_state: list, Wx: list, Wh: list, b: list) -> torch.Tensor:
    """
    Implements a simple RNN cell forward pass using PyTorch.

    Args:
        input_sequence: List of input vectors for each time step.
        initial_hidden_state: The initial hidden state vector.
        Wx: Weight matrix for input-to-hidden connections.
        Wh: Weight matrix for hidden-to-hidden connections.
        b: Bias vector.

    Returns:
        torch.Tensor: The final hidden state after processing the entire sequence,
                      rounded to four decimal places.
    """
    input_sequence=torch.tensor(input_sequence)
    Wx=torch.tensor(Wx)
    Wh=torch.tensor(Wh)
    b=torch.tensor(b)
    initial_hidden_state=torch.tensor(initial_hidden_state)
    for i in range(input_sequence.shape[0]):
        hidden=torch.tanh(torch.matmul(Wx ,input_sequence[i]) + torch.matmul(Wh ,initial_hidden_state)+ b)
        initial_hidden_state=hidden
    return hidden
    