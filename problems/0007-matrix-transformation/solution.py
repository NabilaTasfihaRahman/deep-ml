import numpy as np

def transform_matrix(A: list[list[int|float]], T: list[list[int|float]], S: list[list[int|float]]) -> list[list[int|float]]: 
	bleh=np.matmul(A, S)
	if np.linalg.det(T)==0 or np.linalg.det(S)==0:
		return -1
	else:
		transformed_matrix=np.matmul(np.linalg.inv(T),bleh)
	
		return transformed_matrix