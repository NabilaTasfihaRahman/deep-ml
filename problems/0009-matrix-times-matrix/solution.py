import numpy as np
def matrixmul(a:list[list[int|float]],
              b:list[list[int|float]])-> list[list[int|float]]:
	col1=len(a[0])
    row2=len(b)
    
    if col1!=row2:
        return -1
    else:
        return np.matmul(a,b)