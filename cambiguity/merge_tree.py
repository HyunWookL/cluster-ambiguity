import numpy as np
from numba import njit

## function to construct the arrays
@njit
def construct_value_indices(grid, values, x_idx, y_idx, size):
	for i in range(size):
		for j in range(size):
			values[i * size + j] = grid[i, j]
			x_idx[i * size + j] = i
			y_idx[i * size + j] = j

def construct_merge_tree(grid):
	"""
	construct merge tree based on the given input grid
	"""
	size = len(grid)

	## change grid into arrays holding the grid value and grid indices (x,y)
	values = np.zeros(size ** 2)
	x_idx  = np.zeros(size ** 2)
	y_idx  = np.zeros(size ** 2)
	construct_value_indices(grid, values, x_idx, y_idx, size)

	## sort the arrays (values, x_idx, y_idx) based on the values
	sort_indices = np.argsort(-values)
	values = values[sort_indices]
	x_idx = x_idx[sort_indices]
	y_idx = y_idx[sort_indices]

	## 





	