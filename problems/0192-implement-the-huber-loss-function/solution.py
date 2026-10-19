import numpy as np
def huber_loss(y_true, y_pred, delta=1.0):
	"""
	Compute the Huber Loss between true and predicted values.

	Args:
		y_true (float | list[float]): Ground truth values
		y_pred (float | list[float]): Predicted values
		delta (float): Transition threshold between MSE and MAE behavior

	Returns:
		float: Average Huber loss
	"""
	# Your code here
	if isinstance(y_true,list):
		y_true=list(y_true)
	else:
		arr1=np.array([y_true])
		y_true=arr1.tolist()

	if isinstance(y_pred,list):
		y_pred=list(y_pred)
	else:
		arr2=np.array([y_pred])
		y_pred=arr2.tolist()
		
	Loss=[]
	for i in range(len((y_pred))):
		if abs(y_true[i]-y_pred[i])<=delta:
			loss=0.5*(y_true[i]-y_pred[i])**2
		else:
			loss=delta*(abs(y_true[i]-y_pred[i])-0.5*delta)
		Loss.append(loss)
	return np.mean(Loss)