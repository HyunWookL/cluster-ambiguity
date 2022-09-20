import numpy as np 
from sklearn.mixture import GaussianMixture
from kneed import KneeLocator
from scipy.spatial import Delaunay

import helpers as hp

import importlib

importlib.reload(hp)

import pickle

from scipy.stats import entropy


INPUT_ARR = [
	"rotation_diff", 
	"scaling_diff", 
	"mean_diff", 
	"scaling_size", 
	"scaling_size_diff",
	"mean_diff_scaling_ratio",
	"ellipticity_average",
	"ellipticity_diff",
	"density_diff",
	"density_average",
	"rotation_average",
	"gaussian_mean_vector_angle_diff",
	"gaussian_mean_vector_angle_average",
]


class ClusterAmbiguity():
	"""
	A class for computing cluster ambiguity based on clustme data
	"""

	def __init__(self, corr_thld=0.05, verbose=0, random_state=0, S=3.0, mode="average"):
		"""
		INPUT:
		- corr_thld: the threshold determining the correlation between two clusters
		  - if the correlation is below the threshold, the two clusters are considered not 
			  to be correlated unless they are linked by gabriel graph
		- verbose: 0 if no verbose, > 0 if verbose
		- random_state: input None if dont want to set the random state
		"""
		self.corr_thrl = corr_thld
		self.verbose = (verbose == 0)
		self.random_state = random_state
		self.S = S
		self.mode = mode
		## load regression model
		with open("./regression_model/autosklearn.pkl", "rb") as f:
			self.reg_model = pickle.load(f)

	def fit(self, data):
		np.random.seed(self.random_state)
		self.data = data
		## find optimal n_comp
		self.__find_optimal_n_comp()		
		

		## perform gmm with optimal n_comp and extract the infos
		self.gmm = GaussianMixture(n_components=self.optimal_n_comp, covariance_type='full', random_state=self.random_state)
		self.gmm.fit(data)
		self.convariances = self.gmm.covariances_
		self.means 				= self.gmm.means_
		self.proba 				= self.gmm.predict_proba(data)


		## extract gaussian infos
		self.__extract_gaussian_info()
		## return if n_comp is 1 (with perfect score)
		if(self.optimal_n_comp == 1):
			return 0
		## construct the gabriel graph for future filtering
		self.__construct_gabriel_graph()
		## run the pairwise cluster ambiguity computation
		self.__compute_pairwise_cluster_ambiguity()

		## run final score
		return self.__final_score()
		

	def __find_optimal_n_comp(self):
		## perform gmm from n_comp=1 to n_comp = np.sqrt(len(data)) to find optimal n_comp
		## bic is used for the criteria
		x_list = list(range(1, int(np.sqrt(len(self.data)))))
		y_list = []
		for n_comp in x_list:
			gmm = GaussianMixture(n_components=n_comp, covariance_type='full', random_state=self.random_state)
			gmm.fit(self.data)
			bic = gmm.bic(self.data)
			y_list.append(bic)
		
		## find the optimal elbow value based on kneedle algorithm
		## Satopaa, Ville, et al. "Finding a" kneedle" in a haystack: Detecting knee points in system behavior." 2011 31st international conference on distributed computing systems workshops. IEEE, 2011.
		kneedle = KneeLocator(x_list, y_list, curve='convex', direction='decreasing', S=self.S)
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
			scaling, rotation, rotation_degree = hp.decompose_covariance_matrix(cov)
			self.gaussian_info["scaling"].append(scaling)
			self.gaussian_info["rotation"].append(rotation)
			self.gaussian_info["rotation_degree"].append(rotation_degree)
	
	def __construct_gabriel_graph(self):
		"""
		get the pairs of gaussian components that should be compared to compute the cluster ambiguity
		based on the overlap of ellipse and 
		"""
		if len(self.means) == 2:
			self.gabriel_graph_edges = {"0_1"}
			return

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
		return

	def __compute_pairwise_cluster_ambiguity(self):
		"""
		compute the pairwise cluster ambiguity
		"""
		self.pair_key_list = []
		self.prob_single_score_list = []

		input_variables_arr_list = []
		for i in range(self.optimal_n_comp):
			for j in range(i+1, self.optimal_n_comp):
				new_gaussian_info = hp.normalize_gaussian_info(self.gaussian_info, i, j, self.data)
				input_variables_dict = hp.construct_reg_input_variables(new_gaussian_info, 0, 1)
				input_variables_arr = []
				for var_name in INPUT_ARR:
					input_variables_arr.append(input_variables_dict[var_name])
				input_variables_arr_list.append(input_variables_arr)
				self.pair_key_list.append(f"{i}_{j}")

		self.prob_single_score_list = self.reg_model.predict(input_variables_arr_list)
		
		return

	def __average_ambiguity(self):
		"""
		Compute the ambiguity based on average
		"""
		ambiguity_score_list = np.abs(np.array(self.filtered_prob_single_list) - 0.5) * 2
		self.ambiguity_score = 1 - np.mean(ambiguity_score_list)

		return self.ambiguity_score

	def __entropy_ambiguity(self):
		"""
		Compute the ambiguity based on entropy
		"""
		self.filtered_prob_single_list = np.array(self.filtered_prob_single_list)
		self.filtered_prob_single_list[self.filtered_prob_single_list < 0] = 1e-5
		self.filtered_prob_single_list[self.filtered_prob_single_list > 1] = 1 - 1e-5
		entropy_list = - (
			self.filtered_prob_single_list * np.log2(self.filtered_prob_single_list) +
			(1 - self.filtered_prob_single_list) * np.log2(1 - self.filtered_prob_single_list)
		)

		self.ambiguity_score = np.mean(entropy_list)
		return self.ambiguity_score

		
		


	def __final_score(self):
		"""
		compute the final score
		"""
		self.filtered_prob_single_list = []
		for i, prob_single_score in enumerate(self.prob_single_score_list):
			if prob_single_score > self.corr_thrl or self.pair_key_list[i] in self.gabriel_graph_edges:
				self.filtered_prob_single_list.append(prob_single_score)

		if self.mode == "average":
			return self.__average_ambiguity()
		elif self.mode == "entropy":
			return self.__entropy_ambiguity()



			


