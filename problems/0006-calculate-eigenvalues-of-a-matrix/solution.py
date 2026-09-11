import numpy
def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:
	eigenvalues=numpy.linalg.eigvals(matrix)
	return eigenvalues