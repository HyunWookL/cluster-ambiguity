from pyclustering.cluster.xmeans import xmeans 
from sklearn.cluster import DBSCAN
import hdbscan
from sklearn.cluster import Birch
import numpy as np

from sklearn.metrics import adjusted_mutual_info_score, v_measure_score

def get_clustering_diff(clusterer, metric, X):
	labels = []
	for i in range(20):
		label = clusterer(X)
		labels.append(label)
	
	results = []
	for i in range(20):
		for j in range(0, i):
			results.append(metric(labels[i], labels[j]))
	
	return np.mean(results)


def xmeans_clusterer(X):
	kmax = np.random.randint(2, 50)
	tol  = np.random.uniform(0.01, 1.0)
	clusterer_xmeans = xmeans(X, kmax=kmax, tolerance=tol)
	clusterer_xmeans.process()

	clusters = clusterer_xmeans.get_clusters()
	labels = np.zeros(len(X), dtype=int)
	for i, cluster in enumerate(clusters):
		labels[cluster] = i
	
	return labels

def hdbscan_clusterer(X):
	epsilon = np.random.uniform(0.01, 1.0)
	min_samples = np.random.randint(1, 10)
	min_cluster_size = np.random.randint(2, 50)
	clusterer_hdbscan = hdbscan.HDBSCAN(cluster_selection_epsilon=epsilon, min_samples=min_samples, min_cluster_size=min_cluster_size)
	clusterer_hdbscan.fit(X)
	labels = clusterer_hdbscan.labels_
	return labels

def dbscan_clusterer(X):
	epsilon = np.random.uniform(0.01, 1.0)
	min_samples = np.random.randint(1, 10)
	clusterer_dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
	clusterer_dbscan.fit(X)
	labels = clusterer_dbscan.labels_
	return labels

def birch_clusterer(X):
	threshold = np.random.uniform(0.01, 1.0)
	branching_factor = np.random.randint(10, 100)
	clusterer_birch = Birch(threshold=threshold, branching_factor=branching_factor)
	clusterer_birch.fit(X)
	labels = clusterer_birch.labels_
	return labels