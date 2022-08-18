import numpy as np 
from sklearn.mixture import GaussianMixture
from bayes_opt import BayesianOptimization
from kneed import KneeLocator
from scipy.spatial import Delaunay

from helpers import decompose_covariance_matrix

class ClusterAmbiguity():
	"""
	A class for computing cluster ambiguity based on clustme data
	"""

	def __init__(self, n_std=2.448, verbose=0):
		"""
		INPUT:
		- n_std: number of standard deviations to determine the ellipse representing the gaussian
						 default value is sqrt(5.991) = 2.448, where 5.991 is the confidence level at 95% in Chi-square distribution
		- verbose: 0 if no verbose, > 0 if verbose
		"""
		self.n_std = n_std
		self.verbose = (verbose == 0)

	def fit(self, data):
		self.data = data

		## find optimal n_comp
		self.__find_optimal_n_comp()		
		
		## perform gmm with optimal n_comp and extract the infos
		self.gmm = GaussianMixture(n_components=self.optimal_n_comp, covariance_type='full')
		self.gmm.fit(data)
		self.convariances = self.gmm.covariances_
		self.means 				= self.gmm.means_
		self.proba 				= self.gmm.predict_proba(data)

		## extract gaussian infos
		self.__extract_gaussian_info()
		## construct the gabriel graph for future filtering
		self.__construct_gabriel_graph()

	def __find_optimal_n_comp(self):
		## perform gmm from n_comp=1 to n_comp = np.sqrt(len(data)) to find optimal n_comp
		## bic is used for the criteria
		x_list = list(range(1, int(np.sqrt(len(self.data)))))
		y_list = []
		for n_comp in x_list:
			gmm = GaussianMixture(n_components=n_comp, covariance_type='full')
			gmm.fit(self.data)
			bic = gmm.bic(self.data)
			y_list.append(bic)
		
		## find the optimal elbow value based on kneedle algorithm
		## Satopaa, Ville, et al. "Finding a" kneedle" in a haystack: Detecting knee points in system behavior." 2011 31st international conference on distributed computing systems workshops. IEEE, 2011.
		kneedle = KneeLocator(x_list, y_list, curve='convex', direction='decreasing')
		self.optimal_n_comp = kneedle.knee


	def __extract_gaussian_info(self):
		"""
		extract the gaussian info of each gaussian component
		"""
		self.gaussian_info = {}

		## add means
		self.gaussian_info["means"] = self.means.tolist()
		
		## add proba_labels
		self.gaussian_info["proba_labels"] = []
		for i in range(len(self.proba)):
			proba_incremental = [0]
			for j in range(len(self.proba[i])):
				proba_incremental.append(self.proba[i][j] + proba_incremental[j])
			del proba_incremental[0]
			proba_incremental = list(np.around(np.array(proba_incremental), decimals=5))

			pivot = np.random.rand()
			for j in range(len(proba_incremental)):
				if pivot < proba_incremental[j]:
					self.gaussian_info["proba_labels"].append(j)
					break
		
		## add scaling and rotation
		self.gaussian_info["scaling"] = []
		self.gaussian_info["rotation"] = []
		self.gaussian_info["rotation_degree"] = []
		for cov in self.convariances:
			scaling, rotation, rotation_degree = decompose_covariance_matrix(cov)
			self.gaussian_info["scaling"].append(scaling)
			self.gaussian_info["rotation"].append(rotation)
			self.gaussian_info["rotation_degree"].append(rotation_degree)
	
	def __construct_gabriel_graph(self):
		"""
		get the pairs of gaussian components that should be compared to compute the cluster ambiguity
		based on the overlap of ellipse and 
		"""
		tri = Delaunay(self.means)
		gabriel_graph_edges = {}
		for simplex in tri.simplices:
			gabriel_graph_edges[f"{simplex[0]}_{simplex[1]}"] = True
			gabriel_graph_edges[f"{simplex[1]}_{simplex[2]}"] = True
			gabriel_graph_edges[f"{simplex[2]}_{simplex[0]}"] = True

		def check_within_circle(target, v1, v2):
			center = (np.array(v1) + np.array(v2)) / 2
			radius = np.linalg.norm(np.array(v1) - np.array(v2)) / 2
			return np.linalg.norm(target - center) < radius
		
		for simplex in tri.simplices:
			if check_within_circle(self.means[simplex[2]], self.means[simplex[0]], self.means[simplex[1]]):
				gabriel_graph_edges[f"{simplex[0]}_{simplex[1]}"] = False
			if check_within_circle(self.means[simplex[1]], self.means[simplex[0]], self.means[simplex[2]]):
				gabriel_graph_edges[f"{simplex[2]}_{simplex[0]}"] = False
			if check_within_circle(self.means[simplex[0]], self.means[simplex[1]], self.means[simplex[2]]):
				gabriel_graph_edges[f"{simplex[1]}_{simplex[2]}"] = False

		self.gabriel_graph_edges = set()
		for key in gabriel_graph_edges:
			if gabriel_graph_edges[key]:
				self.gabriel_graph_edges.add(key)
		






			


