import torch
import torch.nn.functional as F
def compute_qkv(X: torch.Tensor, W_q: torch.Tensor, W_k: torch.Tensor, W_v: torch.Tensor):
    """
    Compute Query (Q), Key (K), and Value (V) matrices.
    """
    return torch.matmul(X, W_q), torch.matmul(X, W_k), torch.matmul(X, W_v)

def masked_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Compute masked self-attention.
    """
    # Your code here
    seq_len=Q.shape[0]
    mask=torch.triu(torch.full((seq_len,seq_len), -1e9),diagonal=1)
    score=torch.matmul(Q,K.T)/torch.sqrt(torch.tensor(Q.shape[1]))
    masked_Score=score+mask
    
    return torch.matmul(F.softmax(masked_Score,dim=1),V)