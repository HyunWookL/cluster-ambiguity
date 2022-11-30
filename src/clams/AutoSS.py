"""
Implementation of automatic scatterplot sampling (AutoSS) algorithm.
"""

import numpy as np
import CLAMS as clams
from sampling.Sampler import *
from sampling.SamplingMethods import *
from bayes_opt import BayesianOptimization

class AutoScatterplotSampling():
	"""
	Class for Automatic Scatterplot Sampling (AutoSS)
	"""

	def __init__(
		self, S=3.0, verbose=0, init_points=8, n_iter=40,
		sampling_methods="all"
	):
		self.verbose = verbose
		self.S = S
		self.init_points = init_points
		self.n_iter = n_iter
		self.sampling_methods = sampling_methods

		if sampling_methods == "all":
			self.sampling_methods = [
				"random", 
				"blue_noise",
				"density_biased",
				"outlier_biased_density",
				"multiclass_blue_noise", 
			]
		
		self.sampling_functions = {
			"random": RandomSampling,
			"blue_noise": BlueNoiseSampling,
			"density_biased": DensityBiasedSampling,
			"outlier_biased_density": OutlierBiasedDensityBasedSampling,
			"recursive_subdivision": RecursiveSubdivisionBasedSampling,
			"multiclass_blue_noise": MultiClassBlueNoiseSampling
		}

		self.hyperparameter_range = {
			"random": {
				"sampling_rate": (0.3, 1.0),
			},
			"blue_noise": {
				"sampling_rate": (0.3, 1.0),
			},
			"density_biased": {
				"sampling_rate": (0.3, 1.0),
			},
			"outlier_biased_density": {
				"sampling_rate": (0.3, 1.0),
				"alpha": (0.01, 1),
				"beta": (0.01, 1)
			},
			"multiclass_blue_noise": {
				"sampling_rate": (0.3, 1.0),
			},
		}

	def fit(self, data, label):
		## set sampler
		self.sampler = Sampler()
		self.sampler.set_data(data, label)
		self.candidates = {}
		
		## run optimization for each sampling method
		for method in self.sampling_methods:
			if self.verbose > 0:
				print("Optimizing for {}...".format(method))
			if self.hyperparameter_range[method] is None:
				func, args = self.sampling_functions[method], {}
				self.sampler.set_sampling_method(func, **args)
				sampled_point, sampled_label = self.sampler.get_samples()
				ca = clams.ClusterAmbiguity(verbose=self.verbose, S=self.S)
				score = ca.fit(sampled_point)
				settings = {
					"method": method, 
					"hp": {},
				}
			else:
				sampled_point, sampled_label, settings, score = self.optimize(data, label, method)
			self.candidates[method] = {
				"sampled_point": sampled_point,
				"sampled_label": sampled_label,
				"settings": settings,
				"ambiguity": score
			}
		
		## find the method with the lowest ambiguity among candidates
		best_method = None
		best_score = 10
		for method in self.candidates:
			score = self.candidates[method]["ambiguity"]
			if score < best_score:
				best_score = score
				best_method = method
		
		return self.candidates[best_method]


	def optimize(self, data, label, method):
		"""
		Run the bayesian optimization for optimizing DR embedding
		"""
		def __get_loss(**kwargs):
			"""
			Optimization function
			"""
			func, args = self.sampling_functions[method], kwargs
			self.sampler.set_sampling_method(func, **args)
			sampled_point, sampled_label = self.sampler.get_samples()
			ca = clams.ClusterAmbiguity(verbose=self.verbose, S=self.S)
			score = ca.fit(sampled_point)
			return 1 - score ## note that cluster ambiguity is lower the better
		
		optimizer = BayesianOptimization(
			f=__get_loss,
			pbounds=self.hyperparameter_range[method],
			random_state=0,
			verbose=self.verbose
		)

		optimizer.maximize(
			init_points=self.init_points,
			n_iter=self.n_iter
		)

		## get the results
		max_hyperparameter = optimizer.max["params"]
		self.sampler.set_sampling_method(self.sampling_functions[method], **max_hyperparameter)
		sampled_point, sampled_label = self.sampler.get_samples()
		ca = clams.ClusterAmbiguity(verbose=self.verbose, S=self.S)
		score = ca.fit(sampled_point)
		settings = {
			"method": method,
			"hp": max_hyperparameter
		}
		return sampled_point, sampled_label, settings, score

	