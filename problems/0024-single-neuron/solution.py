import math
import numpy as np
def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
	# Your code here
	output=np.matmul( features,weights)+bias
	
	probabilities=1/(1+np.exp(-output))
	mse=np.mean((np.transpose(probabilities)-labels)**2)
	return probabilities, mse