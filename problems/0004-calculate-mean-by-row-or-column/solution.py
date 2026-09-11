import numpy as np
def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
	if mode=='column':
		return (np.mean(np.array(matrix),axis=0)).tolist()
	else:
		return (np.mean(np.array(matrix),axis=1)).tolist()
