from sklearn.mixture import GaussianMixture
import reader as rd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

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


def extract_single(datum):
	"""
	extract the gaussian mixture labels and params from a single clustme output file
	"""
	gmm = GaussianMixture(n_components=2, covariance_type='full')
	labels = gmm.fit_predict(datum)
	proba = gmm.predict_proba(datum)
	proba_based_label = np.random.rand(len(proba)) < proba[:, 0]
	proba_based_label = proba_based_label.astype(int)
	return gmm, {
		"means": gmm.means_,
		"covariances": gmm.covariances_,
		"labels": labels,
		"proba": proba_based_label
	}

def extract(is_draw=False):
	data = rd.read_clustme_data()
	for i, datum in enumerate(data[:1000:20]):
		gmm, gaussian_info = extract_single(datum["data"])
		if is_draw:
			plot_gmm(gmm, datum["data"], gaussian_info["proba"])
			plt.savefig("./plot_individual/clustme_" + str(i) + ".png")
			plt.clf()
		
extract()