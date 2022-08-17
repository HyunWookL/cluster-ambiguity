from sklearn.mixture import GaussianMixture
import reader as rd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
import json

"""
Functions for drawing gaussian mixture model estimation result
"""

def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    
    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))

def plot_gmm(gmm, X, labels, label=True, ax=None):
    ax = ax or plt.gca()
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=3, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=3, zorder=2)
    ax.axis('equal')
    
    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)

"""
Functions for extracting gaussian infos from clustme
"""

def decompose_covariance_matrix(covariance_matrix):
	"""
	decompose the covariance matrix into its principal components
	"""
	U, s, Vt = np.linalg.svd(covariance_matrix, full_matrices=True)
	scaling = np.sqrt(s).tolist()
	rotation = np.arccos(U[0, 1])
	rotation_degree = rotation * (180 / np.pi)

	return {
		"scaling": scaling,
		"rotation": rotation,
		"rotation_degree": rotation_degree
	}
	

def extract_single(datum):
	"""
	extract the gaussian mixture labels and params from a single clustme output file
	"""
	gmm = GaussianMixture(n_components=2, covariance_type='full')
	labels = gmm.fit_predict(datum)

	proba = gmm.predict_proba(datum)
	proba_based_label = np.random.rand(len(proba)) < proba[:, 0]
	proba_based_label = proba_based_label.astype(int)

	covariances = gmm.covariances_
	decompose_0 = decompose_covariance_matrix(covariances[0])
	decompose_1 = decompose_covariance_matrix(covariances[1])

	return gmm, {
		"means": gmm.means_.tolist(),
		"covariances": covariances.tolist(),
		"labels": labels.tolist(),
		"proba_labels": proba_based_label.tolist(),
		"weights": gmm.weights_.tolist(),
		"scaling": [decompose_0["scaling"], decompose_1["scaling"]],
		"rotation": [decompose_0["rotation"], decompose_1["rotation"]],
		"rotation_degree": [decompose_0["rotation_degree"], decompose_1["rotation_degree"]]
	}

def scale_datum(datum):
	"""
	scale the datum to the range of [0, 1]
	"""
	min_x = min(datum[:, 0])
	max_x = max(datum[:, 0])
	min_y = min(datum[:, 1])
	max_y = max(datum[:, 1])
	range_x = max_x - min_x
	range_y = max_y - min_y
	min_for_scale = min_x if range_x > range_y else min_y
	max_for_scale = max_x if range_x > range_y else max_y
	scaled_datum = (datum - min_for_scale) / (max_for_scale - min_for_scale)
	return scaled_datum


def extract(is_draw=False):
	"""
	extract the gaussian info and store it with original data in json format
	"""
	data = rd.read_clustme_data()
	for i, datum in enumerate(data):
		datum["data"] = scale_datum(datum["data"])
		gmm, gaussian_info = extract_single(datum["data"])
		if is_draw:
			plot_gmm(gmm, datum["data"], gaussian_info["proba_labels"])
			plt.savefig("./plot_individual/" + str(i) + ".png")
			# plt.savefig("./plot_individual_with_score/" + str(datum["prob_single"]) + ".png")
			plt.clf()
		datum["gaussian_info"] = gaussian_info
		datum["data"] = datum["data"].tolist()
		with open("./extracted/" + str(i) + ".json", "w") as f:
			json.dump(datum, f)

extract(True)