class MergeTree():
	def __init__(self):
		self.nodes = {}
		self.curr_node_id = 0

	def initialize_node(self, val, x, y):
		"""
		initialize node with using node_id as key
		returns the node_id of the initialized node
		"""
		self.nodes[self.curr_node_id] = {val: [(x, y)]} 
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
			self.nodes[node_id][val].append((x, y))
		else:
			self.nodes[node_id][val] = [(x, y)]