import numpy as np 
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
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
        draw_ellipse(pos, covar, ax=ax, alpha=w * w_factor)
				
    return ax


def plot_gmm_graph(gmm, X, labels, label=True, means=None, edges=None ,ax=None):
		ax = plot_gmm(gmm, X, labels, label, ax)

		for edge in edges:
			edge = edge.split("_")
			datum = np.array([means[int(edge[0])], means[int(edge[1])]])
			ax.plot(*datum.T, c="black")
