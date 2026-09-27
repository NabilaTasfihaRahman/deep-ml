import torch

class DropoutLayer:
    def __init__(self, p: float):
        """Initialize the dropout layer.
        
        Attributes to set:
            self.p: the dropout rate
            self.mask: stores the dropout mask (initially None)
        """
        # Your code here
        self.p=p
        self.mask=None

    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Forward pass of the dropout layer.
        
        Generate a new mask on each training forward pass and store it in self.mask.
        """
        # Your code here
        self.mask=torch.bernoulli(torch.full(x.shape,torch.tensor(1-self.p)))
        if training==True:
            x=(x*self.mask)/(1-self.p)
        return x
    def backward(self, grad: torch.Tensor) -> torch.Tensor:
        """Backward pass of the dropout layer.
        
        Use the stored self.mask from the most recent forward pass.
        """
        # Your code here
        grad=(grad*self.mask)/(1-self.p)
        return grad