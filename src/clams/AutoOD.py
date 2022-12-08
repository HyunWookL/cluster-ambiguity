"""
Implementation of automatic outlier detection (AutoOD) algorithm.
"""

import numpy as np
import CLAMS as clams
from bayes_opt import BayesianOptimization
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from scipy.spatial.distance import mahalanobis
from pyod.models.abod import ABOD


class AutoOutlierDetection():
	"""
	Class for Automatic Outlier Detection (AutoOD)
	"""

	def __init__(
		self, S=3.0, verbose=0, init_points=8, n_iter=40, detection_algorithms="all",
	):
		self.verbose = verbose
		self.S = S
		self.init_points = init_points
		self.n_iter = n_iter

		detection_algorithms_list = [
			"isof", ## isolation forest
			"lof", ## local outlier factor
			"ocsvm", ## one-class svm
			"abod" ## angle-based outlier detection
		]

		if detection_algorithms == "all":
			self.detection_algorithms = detection_algorithms_list
		else:
			## check whehter the given detetection algorithms are in the list
			for algorithm in detection_algorithms:
				if algorithm not in detection_algorithms_list:
					raise ValueError("The given detection algorithm is not supported.")
			self.detection_algorithms = detection_algorithms
		self.detection_functions = {
			"isof": isolation_forest,
			"lof": local_outlier_factor,
			"ocsvm": one_class_svm,
			"abod": angle_based
		}
		
		self.hyperparameter_range = {
			"isof": {
				"n_estimators": (20, 200),
				"max_samples": (0.1, 1.0),
				"max_features": (0.1, 1.0)
			},
			"lof": {
				"n_neighbors": (5, 50),
			},
			"ocsvm": {
				"kernel": (0, 4),
				"degree": (2, 6),
				"gamma": (0.001, 0.5),
				"nu": (0.001, 1),
				"tol": (1e-5, 1e-2)
			},
			"abod": {
				"n_neighbors": (5, 50),
			}
		}
	
	def fit(self, data):
		"""
		Fit the data using AutoOD.
		"""
		best_method = None
		best_score = -1
		best_params = None
		for algorithm in self.detection_algorithms:
			if self.verbose > 0:
				print("Fitting the data using {} algorithm.".format(algorithm))
			score, params = self.optimize(data, algorithm)
			if score > best_score:
				best_score = score
				best_method = algorithm
				best_params = params
				if best_method in ["isof", "lof"]:
					best_params["contamination"] = "auto"
				elif best_method == "abod":
					best_params["contamination"] = 0.1

		## run the best setting once again
		prediction = self.detection_functions[best_method](data, **best_params)	
		data_wo_outliers = data[prediction == 1]
		if data_wo_outliers.shape[0] == 0:
			raise ValueError("No valid outlier detection executed.")
		ca = clams.ClusterAmbiguity(verbose=self.verbose, S=self.S)
		amb_score = ca.fit(data_wo_outliers)
		return {
			"best_method": best_method,
			"best_params": best_params,
			"ambiguity": amb_score,
			"prediction": prediction
		}
		
			
	def optimize(self, data, method):
		"""
		Run the Bayesian optimization for optmizing OD
		"""

		def __get_loss(**kwargs):
			func, args = self.detection_functions[method], kwargs
			if method in ["isof", "lof"]:
				args["contamination"] = "auto"
			elif method == "abod":
				args["contamination"] = 0.1
			prediction = func(data, **args)
			data_wo_outliers = data[prediction == 1]
			if data_wo_outliers.shape[0] == 0:
				return 0
			ca = clams.ClusterAmbiguity(verbose=self.verbose, S=self.S)
			score = ca.fit(data_wo_outliers) 
			return 1 - score
		
		optimizer = BayesianOptimization(
			f=__get_loss,
			pbounds=self.hyperparameter_range[method],
			random_state=1,
			verbose=self.verbose
		)

		optimizer.maximize(
			init_points=self.init_points,
			n_iter=self.n_iter
		)

		## get the reulsts
		best_params = optimizer.max["params"]
		best_score = optimizer.max["target"]
		return best_score, best_params
		

			
		
def isolation_forest(data, n_estimators, max_samples, contamination, max_features):
	"""
	Isolation Forest
	"""
	n_estimators = int(n_estimators)
	clf = IsolationForest(
		n_estimators=n_estimators,
		max_samples=max_samples,
		contamination=contamination,
		max_features=max_features,
		verbose=0,
		random_state=0,
		n_jobs=-1
	)
	clf.fit(data)
	return clf.predict(data)

def local_outlier_factor(data, n_neighbors, contamination):
	"""
	Local Outlier Factor
	"""
	n_neighbors = int(n_neighbors)
	clf = LocalOutlierFactor(
		n_neighbors=n_neighbors,
		contamination=contamination,
		n_jobs=-1
	)
	return clf.fit_predict(data)

def one_class_svm(data, kernel, degree, gamma, tol, nu):
	"""
	One-Class SVM
	"""
	kernel_list = ["linear", "poly", "rbf", "sigmoid"]
	kernel = kernel_list[int(kernel)]
	degree = int(degree)
	clf = OneClassSVM(
		kernel=kernel,
		degree=degree,
		gamma=gamma,
		tol=tol,
		nu=nu,
		verbose=0,
	)
	clf.fit(data)
	return clf.predict(data)

def mahalanobis_distance(data, contamination):
	"""
	Mahalanobis distance based detection
	"""
	mean = np.mean(data, axis=0)
	cov = np.cov(data.T)
	distances = [mahalanobis(x, mean, cov) for x in data]
	threshold = np.percentile(distances, 100 - contamination * 100)
	prediction = [1 if d < threshold else -1 for d in distances]
	return np.array(prediction)

def angle_based(data, n_neighbors, contamination):
	"""
	Angle-based detection
	"""
	n_neighbors = int(n_neighbors)	
	clf = ABOD(contamination=contamination, n_neighbors=n_neighbors)
	clf.fit(data)
	prediction = clf.predict(data)
	prediction = [1 if p == 0 else -1 for p in prediction]
	return np.array(prediction)