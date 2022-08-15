from operator import is_
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


def is_adjacent_cells_occupied(node_distribution, x, y):
	"""
	check if the adjacent cells of (x,y) are occupied
	"""
	for i in range(max(x - 1, 0), min(x + 2, len(node_distribution))):
		for j in range(max(y - 1, 0), min(y + 2, len(node_distribution))):
			if i == x and j == y:
				continue
			if node_distribution[i, j] != -1:
				return True
	return False
	

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

	## traverse the sorted arrays and construct the merge tree
	node_distribution = np.full((size, size), -1)
	node_pool         = {}

	curr_F = values[0]
	for i in range(size ** 2):

		## confirm the constructed merge tree candidate 
		if val != curr_F:
			## TODO
			## TODO
			curr_F = val
			
		## update the merge tree candidate
		x, y, val = x_idx[i], y_idx[i], values[i]
		if is_adjacent_cells_occupied(node_distribution, x, y):
			##TODO
			pass
		else:
			##TODO
			pass




		







	