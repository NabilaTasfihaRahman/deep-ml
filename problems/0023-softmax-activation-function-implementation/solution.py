import math
import numpy as np
def softmax(scores: list[float]) -> list[float]:
    # Your code here
    denominator=[np.exp(i-np.max(scores)) for i in scores]
    output=[np.exp(i-np.max(scores))/np.sum(denominator, axis=-1) for i in scores]
    return output