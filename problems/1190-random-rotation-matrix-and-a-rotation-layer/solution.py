import numpy as np
def rotation_layer(X, angle):
    rotation_matrix=[[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]
    return np.matmul(X,np.transpose(rotation_matrix))