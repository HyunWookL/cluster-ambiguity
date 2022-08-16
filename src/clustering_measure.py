from sklearn.metrics import silhouette_score
import numpy as np
from scipy.spatial.distance import cdist
from numba import njit



# def weighted_complete_distance(X1, X2, grid):
# 	"""
# 	compute the weighted complete distance of the given data
# 	"""
# 	## change tuples in X1 and X2 into list and convert them to numpy
# 	X1 = np.array([list(x) for x in X1], dtype=np.int32)
# 	X2 = np.array([list(x) for x in X2], dtype=np.int32)

# 	cdist


def kl_divergence(X1, X2, grid):
	"""
	compute the KL divergence of the given data
	"""
	## change tuples in X1 and X2 into list and convert them to numpy
	X1_grid = np.zeros(grid.shape)	
	X2_grid = np.zeros(grid.shape)

	for x in X1:
		X1_grid[x[0]][x[1]] = 1
	for x in X2:
		X2_grid[x[0]][x[1]] = 1
	
	return 0


def weighted_silhouette(X1, X2, grid):
	"""
	compute silhouette score of the given data
	"""

	## change tuples in X1 and X2 into list and convert them to numpy
	X1 = np.array([list(x) for x in X1], dtype=np.int32)
	X2 = np.array([list(x) for x in X2], dtype=np.int32)

	for i in range(len(X1)):
		a_val(i, X1, grid)
	for i in range(len(X2)):
		a_val(i, X2, grid)



	
def a_val(idx, X, grid):
	"""
	compute the a value of the given index in the given data
	"""
	if len(X) == 1:
		return 0

	## numba function for computing loop
	def a_val_loop_2(idx, X, grid):
		a_val = 0
		for i in range(len(X)):
			if i == idx:
				continue
			a_val += np.sqrt(np.sum(np.square(X[idx] - X[i]))) * grid[X[i][0]][X[i][1]] 
		return a_val

	a_val = a_val_loop_2(idx, X, grid)

	return(a_val * grid[X[idx][0]][X[idx][1]]) / (len(X) - 1)

