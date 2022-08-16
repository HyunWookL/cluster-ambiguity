import numpy as np
import importlib

import gridify as gy
importlib.reload(gy)

import construct_merge_tree as cmt
importlib.reload(cmt)


class ClusterAmbiguity():
	"""
	class for computing cluster ambiguity and storing the related infos
	"""

	def __init__(
		self, 
		data, 
		opacity, 
		radius, 
		pixel_size = 1024, 
		grid_size = 64
	):
		"""
		initialize the class; 
		gridify the data and construct a merge tree structure
		"""
		self.data 			= data
		self.opacity 		= opacity
		self.radius 		= radius
		self.pixel_size = pixel_size
		self.grid_size  = grid_size
		self.grid       = gy.gridify(self.data, self.opacity, self.radius, self.pixel_size, self.grid_size)
		self.mt   		  = cmt.construct_merge_tree(self.grid)

	def fit(self):
		"""
		compute the cluster ambiguity
		"""
		## get and traverse the node hierarchy
		self.node_hierarchy = self.mt.get_node_hierarchy()
		print(list(self.node_hierarchy.keys())[0])
		# self.cluster_ambiguity_score = self.traverse_and_compute_cluster_ambiguity(self.node_hierarchy.keys()[0])


	