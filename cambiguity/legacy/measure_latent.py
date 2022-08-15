import numpy as np
from pyclustering.cluster.xmeans import xmeans
from sklearn.metrics import silhouette_score

def measure_latent(latent):
	"""
	measure the clusterness of the latent values
	by applying clustering algorithm and applying silhouette coefficient
	"""

	# apply clustering algorithm
	cluster_algorithm = xmeans(latent)
	cluster_algorithm.process()
	cluster_labels_list = cluster_algorithm.get_clusters()

	cluster_labels = np.empty(latent.shape[0])
	for i, cluster_labels_i in enumerate(cluster_labels_list):
		cluster_labels[cluster_labels_i] = i
	

	# apply silhouette coefficient
	score = silhouette_score(latent, cluster_labels)

	return score

	