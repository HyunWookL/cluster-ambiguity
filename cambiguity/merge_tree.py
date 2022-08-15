class MergeTree():
	def __init__(self):
		self.nodes = {}
		self.curr_node_id = 0

	def initialize_node(self, val, x, y):
		"""
		initialize node with using node_id as key
		returns the node_id of the initialized node
		"""
		self.nodes[self.curr_node_id] = {"coord": { val: [(x, y)] }, "childs": [] } 
		self.curr_node_id += 1
		return self.curr_node_id - 1


	def add_cell(self, node_id, val, x, y):
		"""
		add coord to the node_id
		raise KeyError if node_id does not exist
		"""
		if node_id not in self.nodes:
			raise KeyError("node_id does not exist within the nodes set")
		if val in self.nodes[node_id]:
			self.nodes[node_id]["coord"][val].append((x, y))
		else:
			self.nodes[node_id]["coord"][val] = [(x, y)]

	def merge_nodes(self, component, val):
		"""
		merge the nodes in component into one node
		if the child node only contains "val" as a key, delte the child node
		instead, add the child node to the parent node while moving the coords linked to "val" to parents
		RETURN the resulting merged node id
		"""
		component_list = list(component)
		merged_node = { 
			"coord": { val: [] },
			"childs": []
		}

		## merge the nodes
		for node_id in component_list:
			if val in self.nodes[node_id]["coord"]:
				merged_node["coord"][val] += self.nodes[node_id]["coord"][val]
				del self.nodes[node_id]["coord"][val]
			if len(self.nodes[node_id]["coord"]) == 0 and len(self.nodes[node_id]["childs"]) == 0:
				del self.nodes[node_id]
			else:
				merged_node["childs"].append(node_id)

		## update the nodes
		if len(merged_node["childs"]) == 1: ## if there exists a single child node, the merge node is just a extension of a child node
			self.nodes[merged_node["childs"][0]]["coord"][val] = merged_node["coord"][val]
			return merged_node["childs"][0]
		else:
			self.nodes[self.curr_node_id] = merged_node
			self.curr_node_id += 1
			return self.curr_node_id - 1
	

	# def return_node_