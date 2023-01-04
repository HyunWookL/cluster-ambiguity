
import numpy as np
import os, json
from tqdm import tqdm
from sklearn.cluster import KMeans

with open("./measures/dataset_list.json", "r") as f:
	dataset_list = np.array(json.load(f))

gmm_vec = np.load("./measures/gmm_vec.npy")
scagnostics_vec = np.load("./measures/scagnostics_vec.npy")

gmm_min = 2
gmm_max = np.max(gmm_vec)

scatterplot_per_n_comp = 12

sampled_datasets = []
for n_comp in range(gmm_min, gmm_max + 1):
	dataset_with_n_comp = dataset_list[gmm_vec == n_comp]
	scagnostics_vec_with_n_comp = scagnostics_vec[gmm_vec == n_comp]

	if (len(dataset_with_n_comp) < scatterplot_per_n_comp):
		sampled_datasets += dataset_with_n_comp.tolist()
		continue

	## run kMeans
	kmeans = KMeans(n_clusters=scatterplot_per_n_comp, random_state=0).fit(scagnostics_vec_with_n_comp)
	centers = kmeans.cluster_centers_
	centers_label = []
	for center in centers:
		centers_label.append(np.argmin(np.linalg.norm(scagnostics_vec_with_n_comp - center, axis=1)))

	sampled_datasets += dataset_with_n_comp[centers_label].tolist()

print(f"Lenght of sampled datasets: {len(sampled_datasets)}")

with open("./sampling/sampled_datasets.json", "w") as f:
	json.dump(sampled_datasets, f)

