import numpy as np

def roll(a: np.ndarray) -> np.ndarray:
    """Circular left shift by one."""
    # Your code here
    out=[]
    n=a.shape[0]
    for i in range(n):
        out.append(a[(i+1)%n])
    return np.array(out)
