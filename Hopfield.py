import numpy as np

x_input = np.array([[1,-1,-1, 1], [-1,1,1,-1]], dtype=np.float).reshape(2,4,1)
col = len(x_input[0])
weights = -1 * len(x_input) * np.identity(col)

for x in x_input:
	weights += x@x.transpose()

print (weights)
