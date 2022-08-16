import numpy as np
import importlib

import gridify as gy
importlib.reload(gy)

import construct_merge_tree as cmt
importlib.reload(cmt)

import clustering_measure as clm
importlib.reload(clm)


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
				if (len(coords_list[i]) + len(coords_list[j])) < 3:
					continue ## skip the pair if the number of tuples is less than 3 (cannot perform regression)
				curr_pair_score  = self.__compute_score(coords_list[i], coords_list[j])
				curr_pair_weight = self.__get_weight([coords_list[i], coords_list[j]])
				curr_pair_score_sum += curr_pair_score * curr_pair_weight
				curr_pair_weight_sum += curr_pair_weight
		if curr_pair_score_sum > 0:
			curr_node_score = curr_pair_score_sum / curr_pair_weight_sum
			curr_node_weight = self.__get_weight(coords_list)
		else:
			curr_node_score = 0
			curr_node_weight = 0

		### add the score / weight of the current node to the accumulated score
		score  += curr_node_score * curr_node_weight
		weight += curr_node_weight
		coords  = self.__combine_coords(coords_list + [self.mt.get_node_coord_dict(node_id)])

		return score, weight, coords

	def __compute_score(self, coords_1, coords_2):
		"""
		compute the score of the two given coordinates
		"""
		## extract keys
		F_1 = list(coords_1.keys())
		F_2 = list(coords_2.keys())
		F_list = list(set(F_1 + F_2))
		F_list = np.sort(F_list)[::-1]
		m_list = []
		# print("==============")
		# print(F_list)
		# print(coords_1)
		# print(coords_2)


		## compute the measure scores corresponds to F values
		curr_cluster_1 = []
		curr_cluster_2 = []
		for F in F_list: 
			### update the current clusters 1 and 2
			if F in F_1:
				curr_cluster_1 += coords_1[F]
			if F in F_2:
				curr_cluster_2 += coords_2[F]
			if len(curr_cluster_1) == 0 or len(curr_cluster_2) == 0:
				m_list.append(-100)
				continue
			m_list.append(clm.silhouette(curr_cluster_1, curr_cluster_2, self.grid))
		
		print("----------------")
			### compute the score



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

				




	