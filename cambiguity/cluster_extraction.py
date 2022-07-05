import numpy as np
from collections import deque

def cluster_extraction(grid, walk_num_ratio = 0.5):
	"""
	extract the cluster from the grid and return as a form of grid
	"""

	## set constant
	grid_size = grid.shape[0]
	walk_num = (grid_size ** 2) * walk_num_ratio

	## normalize the grid
	grid = grid / np.max(grid)

	## set the seed cell
	seed_x = np.random.randint(grid_size)
	seed_y = np.random.randint(grid_size)
	while grid[seed_x, seed_y] == 0:
		seed_x = np.random.randint(grid_size)
		seed_y = np.random.randint(grid_size)

	## set the data structure for the walk 
	traversal_queue = deque([(seed_x, seed_y)])
	visited = set()
	visited.add(seed_x * grid_size + seed_y)

	## run the traversal
	visit_num = 0
	while visit_num < walk_num:
		if not traversal_queue:
			break
		x, y = traversal_queue.popleft()
		for i in range(x - 1, x + 2):
			for j in range(y - 1, y + 2):
				if (i >= 0 and i < grid_size and j >= 0 and j < grid_size) and (i != x or j != y):
					weight = (grid[i, j] + grid[x, y] + (1 - np.abs(grid[i, j] - grid[x, y]))) / 3
					prob   = np.random.rand()
					# print(weight, prob)
					if weight > prob:
						traversal_queue.append((i, j))
						visited.add(i * grid_size + j)
						visit_num += 1

	
	## change visited into grid format (0 or 1)
	visited_grid = np.zeros((grid_size, grid_size), dtype = np.int)
	for ij in visited:
		visited_grid[ij // grid_size, ij % grid_size] = 1

	return visited_grid





