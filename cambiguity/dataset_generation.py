from dataclasses import dataclass
import numpy as np
from gridify import gridify
from cluster_extraction import cluster_extraction
import os
from tqdm import tqdm

def generate_random_clusters(
	data, data_name, opacity, radius, pixel_size, grid_size,
	walk_num_ratio, cluster_num, path
):
	"""
	generate grid based on given hyperparameters and
	extract clusters from the grid
	"""
	grid = gridify(data, opacity, radius, pixel_size, grid_size)
	
	identifier = '_'.join([data_name, str(opacity), str(radius), str(pixel_size), str(grid_size)])
	## make directory if not exists
	if not os.path.exists(f"{path}/{identifier}/"):
		os.makedirs(f"{path}/{identifier}/")
	for i in tqdm(range(cluster_num)):
		cluster = cluster_extraction(grid, walk_num_ratio)
		np.save(f"{path}/{identifier}/{i}.npy", cluster)