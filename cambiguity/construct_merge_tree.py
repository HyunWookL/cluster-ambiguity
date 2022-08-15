import numpy as np
from numba import njit
import merge_tree as mtree
import networkx as nx

import importlib
importlib.reload(mtree)

## function to construct the arrays
@njit
def construct_value_indices(grid, values, x_idx, y_idx, size):
	for i in range(size):
		for j in range(size):
			values[i * size + j] = grid[i, j]
			x_idx[i * size + j] = i
			y_idx[i * size + j] = j


def adjacent_cells_composition(node_distribution, x, y):
	"""
	check the compostion of the adjacent cells of (x,y) in the node_distribution
	"""
	composition = set()
	for i in range(max(x - 1, 0), min(x + 2, len(node_distribution))):
		for j in range(max(y - 1, 0), min(y + 2, len(node_distribution))):
			if i == x and j == y:
				continue
			composition.add(node_distribution[i, j])
	return composition

def update_adj_nodes_graph(adj_nodes_graph, composition_list):
	"""
	update the adjacent nodes graph for future combine of the nodes
	"""
	for node_id1 in composition_list:
		for node_id2 in composition_list:
			if node_id1 != node_id2 and not adj_nodes_graph.has_edge(node_id1, node_id2):
				adj_nodes_graph.add_edge(node_id1, node_id2)

def update_merge_tree(mt, node_distribution, adj_nodes_graph, curr_F):
	"""
	update the merge tree based on the given components
	"""
	components = nx.connected_components(adj_nodes_graph)
	for c in components:
		merged_node_id = mt.merge_nodes(c, curr_F)
		for child_id in c:
			node_distribution[node_distribution == child_id] = merged_node_id
	adj_nodes_graph.clear()



	

def construct_merge_tree(grid):
	"""
	construct merge tree based on the given input grid
	returns the constructed merge tree sturcture
	"""
	size = len(grid)

	## change grid into arrays holding the grid value and grid indices (x,y)
	values = np.zeros(size ** 2)
	x_idx  = np.zeros(size ** 2, dtype=np.int32)
	y_idx  = np.zeros(size ** 2, dtype=np.int32)
	construct_value_indices(grid, values, x_idx, y_idx, size)

	## sort the arrays (values, x_idx, y_idx) based on the values
	sort_indices = np.argsort(-values)
	values 			 = values[sort_indices]
	x_idx 			 = x_idx[sort_indices]
	y_idx 			 = y_idx[sort_indices]

	## initialize the data structure for constructing merge tree
	node_distribution = np.full((size, size), -1)
	adj_nodes_graph   = nx.Graph()
	mt 								= mtree.MergeTree(size)

	## traverse the sorted arrays and update merge tree
	curr_F = values[0]
	for i in range(size ** 2):
		x, y, val = x_idx[i], y_idx[i], values[i]

		## confirm the constructed merge tree candidate 
		if val != curr_F:
			update_merge_tree(mt, node_distribution, adj_nodes_graph, curr_F)
			curr_F = val
			
		## update the merge tree candidate
		composition = adjacent_cells_composition(node_distribution, x, y)
		if len(composition) == 1: ## if the composition is unique
			if -1 in composition: ## if the adjacent cells are not occupied by other nodes 
				node_id = mt.initialize_node(curr_F, x, y)
				node_distribution[x, y] = node_id
			else:
				node_id = composition.pop()
				mt.add_cell(node_id, curr_F, x, y)
				node_distribution[x, y] = node_id
		else: ## if the composition is not unique
			composition.discard(-1)
			composition_list = list(composition)
			mt.add_cell(composition_list[0], curr_F, x, y)
			node_distribution[x, y] = composition_list[0]
			update_adj_nodes_graph(adj_nodes_graph, composition_list)
		
	## finialize merge tree
	update_merge_tree(mt, node_distribution, adj_nodes_graph, curr_F)			
	return mt

		
				



		







	