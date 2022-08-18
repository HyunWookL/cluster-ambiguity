import numpy as np 
from sklearn.mixture import GaussianMixture
from bayes_opt import BayesianOptimization
from kneed import KneeLocator

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

		## perform gmm from n_comp=1 to n_comp = np.sqrt(len(data)) to find optimal n_comp
		## bic is used for the criteria
		x_list = list(range(1, int(np.sqrt(len(data)))))
		y_list = []
		for n_comp in x_list:
			gmm = GaussianMixture(n_components=n_comp, covariance_type='full')
			gmm.fit(data)
			bic = gmm.bic(data)
			y_list.append(bic)
		
		## find the optimal elbow value based on kneedle algorithm
		## Satopaa, Ville, et al. "Finding a" kneedle" in a haystack: Detecting knee points in system behavior." 2011 31st international conference on distributed computing systems workshops. IEEE, 2011.
		kneedle = KneeLocator(x_list, y_list, curve='convex', direction='decreasing')
		self.optimal_n_comp = kneedle.knee
		
		## perform gmm with optimal n_comp and extract the infos
		self.gmm = GaussianMixture(n_components=self.optimal_n_comp, covariance_type='full')
		self.gmm.fit(data)
		self.convariances = self.gmm.covariances_
		self.means 				= self.gmm.means_
		self.proba 				= self.gmm.predict_proba(data)

		## extract gaussian infos
		self.__extract_gaussian_info()
		

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




			


