import numpy as np
def calculate_brightness(img):
	# Write your code here
	row_length=[len(row) for row in img]
	
	if len(img)==0 or max(row_length)!= min(row_length) or np.any(np.array(img) >255):
		return -1
	brightness=np.mean(img)
	return brightness
