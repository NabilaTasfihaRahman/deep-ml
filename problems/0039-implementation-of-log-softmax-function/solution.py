import numpy as np

def log_softmax(scores: list) -> np.ndarray:
	# Your code here
	denominator=np.sum([np.exp(i-np.max(scores)) for i in scores])
	return [i-np.max(scores)-np.log(denominator) for i in scores]