import numpy as np

def flip(a: np.ndarray) -> np.ndarray:
    """Reverse 1-D array a without slicing a[::-1]."""
    # Your code here
    out=[]
    n=len(a)
    for i in range(n):
        out.append(a[n-1-i])

    return np.array(out)