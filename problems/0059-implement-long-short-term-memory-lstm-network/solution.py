import torch

class LSTM:
    def __init__(self, input_size: int, hidden_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Initialize weights and biases as float64 tensors
        self.Wf = torch.randn(hidden_size, input_size + hidden_size, dtype=torch.float64)
        self.Wi = torch.randn(hidden_size, input_size + hidden_size, dtype=torch.float64)
        self.Wc = torch.randn(hidden_size, input_size + hidden_size, dtype=torch.float64)
        self.Wo = torch.randn(hidden_size, input_size + hidden_size, dtype=torch.float64)

        self.bf = torch.zeros(hidden_size, 1, dtype=torch.float64)
        self.bi = torch.zeros(hidden_size, 1, dtype=torch.float64)
        self.bc = torch.zeros(hidden_size, 1, dtype=torch.float64)
        self.bo = torch.zeros(hidden_size, 1, dtype=torch.float64)

    def forward(self, x: torch.Tensor, initial_hidden_state: torch.Tensor, initial_cell_state: torch.Tensor):
        """
        Processes a sequence of inputs and returns the hidden states,
        final hidden state, and final cell state.

        Args:
            x: Input tensor of shape (seq_len, input_size)
            initial_hidden_state: Initial hidden state of shape (hidden_size, 1)
            initial_cell_state: Initial cell state of shape (hidden_size, 1)

        Returns:
            outputs: Tensor of hidden states at each time step
            h: Final hidden state tensor
            c: Final cell state tensor
        """
        outputs=[]
        for i in range(x.shape[0]):
            forget=torch.sigmoid(torch.matmul(self.Wf,torch.cat([initial_hidden_state,x[i].unsqueeze(1)]))+self.bf)
            input_gate=torch.sigmoid(torch.matmul(self.Wi,torch.cat([initial_hidden_state,x[i].unsqueeze(1)]))+self.bi)
            ct_bar=torch.tanh(torch.matmul(self.Wc,torch.cat([initial_hidden_state,x[i].unsqueeze(1)]))+self.bc)
            c=forget*initial_cell_state+input_gate*ct_bar
            output=torch.sigmoid(torch.matmul(self.Wo,torch.cat([initial_hidden_state,x[i].unsqueeze(1)]))+self.bo)
            h=output*torch.tanh(c)
            outputs.append(h)
            initial_hidden_state=h
            initial_cell_state=c
        outputs=torch.stack(outputs)
        return outputs,h,c

