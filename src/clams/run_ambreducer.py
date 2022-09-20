import numpy as np

import umap
from sklearn.manifold import TSNE
from bayes_opt import BayesianOptimization

import AmbReducer as ambr
from snc.snc import SNC 

def find_best_embedding(raw, method="tsne", verbose=2, init_points=8, n_iter=40):
	def get_loss(**kwargs):
		if method == "tsne":
			emb = TSNE(n_components=2, perplexity=kwargs["perplexity"], random_state=0).fit_transform(raw)
		elif method == "umap":
			emb = umap.UMAP(n_components=2, n_neighbors=int(kwargs["n_neighbors"]), min_dist=kwargs["min_dist"], random_state=0).fit_transform(raw)
		metrics = SNC(raw, emb, iteration=300, walk_num_ratio=0.4)
		metrics.fit()
		stead, cohev = metrics.steadiness(), metrics.cohesiveness()
		return (2 * (stead * cohev)) / (stead + cohev)

	optimizer = BayesianOptimization(
		f=get_loss,
		pbounds={ "perplexity": (1, 1000) } if method == "tsne" else { "n_neighbors": (2, 200), "min_dist": (0.0, 1.0) },
		random_state=1,
		verbose=verbose
	)

	optimizer.maximize(init_points=init_points, n_iter=n_iter)

	max_hyperparameter = optimizer.max["params"]
	if method == "tsne":
		emb = TSNE(n_components=2, perplexity=max_hyperparameter["perplexity"], random_state=0).fit_transform(raw)
	elif method == "umap":
		emb = umap.UMAP(n_components=2, n_neighbors=int(max_hyperparameter["n_neighbors"]), min_dist=max_hyperparameter["min_dist"], random_state=0).fit_transform(raw)
	
	return emb, max_hyperparameter

def run(raw, method="tsne", metric="snc", threshold_loss=0.10, S=3.0, verbose=2, init_points=8, n_iter=40):
	emb, max_hyperparameter = find_best_embedding(raw, method=method, verbose=verbose, init_points=init_points, n_iter=n_iter)
	amb = ambr.AmbReducer(method=method, metric=metric, threshold_loss=threshold_loss, S=S, verbose=verbose, init_points=init_points, n_iter=n_iter)
	amb.fit(raw, emb)
	amb.optimize()
	results = amb.get_infos()
	results["init_max_hyperparameter"] = max_hyperparameter
	return results