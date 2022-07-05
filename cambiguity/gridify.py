import numpy as np
from numba import njit, prange


def pixelify(data, opacity, radius, pixel_size):
	"""
	convert input 2D data into a pixels based on the density of points
	"""

	pixels = np.zeros((pixel_size, pixel_size))

	## rescale data to fit in the pixels
	x_min = np.min(data[:, 0])
	x_max = np.max(data[:, 0])
	y_min = np.min(data[:, 1])
	y_max = np.max(data[:, 1])

	if (x_max - x_min) > (y_max - y_min):
		y_max = (y_max + y_min) / 2 + (x_max - x_min) / 2
		y_min = (y_max + y_min) / 2 - (x_max - x_min) / 2
	else:
		x_max = (x_max + x_min) / 2 + (y_max - y_min) / 2
		x_min = (x_max + x_min) / 2 - (y_max - y_min) / 2
	
	data[:, 0] = (data[:, 0] - x_min) / (x_max - x_min) * pixel_size
	data[:, 1] = (data[:, 1] - y_min) / (y_max - y_min) * pixel_size

	## function to draw points in the pixels
	@njit
	def draw_points(data, opacity, radius, pixels):
		for datum in data:
			x = int(datum[0])
			y = int(datum[1])
			for i in range(x - radius, x + radius + 1):
				for j in range(y - radius, y + radius + 1):
					if ((i - x) ** 2 + (j - y) ** 2 <= radius ** 2 
							and i >= 0 and i < pixel_size
							and j >= 0 and j < pixel_size
					):
						pixels[i, j] += opacity
						if (pixels[i, j]) > 1:
							pixels[i, j] = 1
	
	## run the function
	draw_points(data, opacity, radius, pixels)

	return pixels


def gridify(data, opacity, radius, grid_size, cell_size):
	"""
	convert input 2D data into a grid based on the density of points
	"""

	## get pixels 
	pixel_size = grid_size * cell_size
	pixels = pixelify(data, opacity, radius, pixel_size)
	
	@njit(parallel=True)
	def fill_grid(pixels, grid, cell_size):
		for i in prange(grid.shape[0]):
			for j in prange(grid.shape[1]):
				grid[i, j] = np.sum(pixels[i * cell_size : (i + 1) * cell_size, j * cell_size : (j + 1) * cell_size])


	## fill the grid based on pixels value
	grid = np.zeros((grid_size, grid_size))
	fill_grid(pixels, grid, cell_size)
	grid = grid / (cell_size ** 2)

	return grid


