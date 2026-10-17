import numpy as np

def transfer_and_attend(X, W_q_lin, W_k_lin, W_v_lin):
    """
    Transfer linear-style weights (shape (d_out, d_in)) into parameter-style
    weights (shape (d_in, d_out)) and compute single-head causal self-attention.

    Args:
        X: input array of shape (seq_len, d_in)
        W_q_lin, W_k_lin, W_v_lin: linear-style weight matrices of shape (d_out, d_in)

    Returns:
        Nested list of shape (seq_len, d_out): the self-attention output.
    """
    # Your code here
    Q=np.matmul(X,np.transpose(W_q_lin))
    K=np.matmul(X,np.transpose(W_k_lin))
    V=np.matmul(X,np.transpose(W_v_lin))
    
    S=np.matmul(Q,np.transpose(K))/np.sqrt(Q.shape[1])
    S[np.triu_indices_from(S,k=1)]=-10**9
    S_new=S - np.max(S, axis=-1, keepdims=True)
    A=np.exp(S_new)/np.sum(np.exp(S_new),axis=1,keepdims=True)
    return np.matmul(A,V)
