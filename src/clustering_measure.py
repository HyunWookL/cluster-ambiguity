from sklearn.metrics import silhouette_score
import numpy as np

def silhouette(X1, X2, grid):
	## concat X1 and X2 to a single data and labels
	print(np.array(grid).shape)
	print("====")
	for x in X1:
		print(x)
		print(grid[x[0]][x[1]], end= " ")
	print("\n")
	for x in X2:
		print(grid[x[0]][x[1]], end= " ")
	print("\n")
	# X = np.concatenate((X1, X2), axis=0)
	# y = np.concatenate((np.zeros(X1.shape[0]), np.ones(X2.shape[0])), axis=0)
	
