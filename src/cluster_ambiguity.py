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
		root_node = list(self.node_hierarchy.keys())[0]
		final_score, final_weight, _ = self.traverse_and_accumulate_score(root_node, self.node_hierarchy[root_node])
		self.score = final_score / final_weight

		return self.score

	def traverse_and_accumulate_score(self, node_id, n_hierarchy):
		"""
		recursively traverse the node hierarchy and accumulate the score
		"""
		score = 0
		weight = 0

		## base case (no child)
		if len(n_hierarchy) == 0:
			coords = self.mt.get_node_coord_dict(node_id)
			return score, weight, coords

		## recursive case
		keys = list(n_hierarchy.keys())
		coords_list = []

		### accumulate score and weight
		for key in keys:
			score_, weight_, coords_ = self.traverse_and_accumulate_score(key, n_hierarchy[key])
			score += score_
			weight += weight_
			coords_list.append(coords_)
		

		### compute the score of the current node
		curr_pair_score_sum = 0 
		curr_pair_weight_sum = 0
		for i, _ in enumerate(coords_list):
			for j in range(i + 1, len(coords_list)):
				curr_pair_score  = self.__compute_score(coords_list[i], coords_list[j])
				curr_pair_weight = self.__get_weight([coords_list[i], coords_list[j]])
				curr_pair_score_sum += curr_pair_score * curr_pair_weight
				curr_pair_weight_sum += curr_pair_weight
		curr_node_score = curr_pair_score_sum / curr_pair_weight_sum
		curr_node_weight = self.__get_weight(coords_list)

		### add the score / weight of the current node to the accumulated score
		score  += curr_node_score * curr_node_weight
		weight += curr_node_weight
		coords  = self.__combine_coords(coords_list + [self.mt.get_node_coord_dict(node_id)])

		return score, weight, coords

	def __compute_score(self, coords1, coords2):
		"""
		compute the score of the two given coordinates
		"""
		return 0.5

	def __get_weight(self, coords_list):
		"""
		get the weight of the given coordinates list
		the weight is the number of tuples throughout the coords 
		"""
		weight = 0
		for coord in coords_list:
			for val in coord:
				weight += len(coord[val])
		return weight
	
	def __combine_coords(self, coords_list):
		"""
		combine the coordinates of the given coordinates list
		"""
		new_coords = {}
		for coord in coords_list:
			for key in coord:
				if key not in new_coords:
					new_coords[key] = coord[key]
				else:
					new_coords[key] += coord[key]
		return new_coords

				




	