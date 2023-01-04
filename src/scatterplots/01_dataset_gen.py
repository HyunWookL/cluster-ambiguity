import umap 
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding, MDS
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
import umato
import os
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

DATA_PATH = "../labeled-datasets/npy/"
datasets_list = os.listdir(DATA_PATH)
datasets = []
for dataset_name in datasets_list:
	if dataset_name == ".gitignore":
		continue
	dataset = np.load(DATA_PATH + dataset_name + "/data.npy")
	if dataset.shape[0] > 5000:
		## downsample
		dataset = dataset[np.random.choice(dataset.shape[0], 5000, replace=False)]
	dataset = StandardScaler().fit_transform(dataset)
	datasets.append(dataset)

## generate embeddings
for i, dataset in tqdm(enumerate(datasets)):
	if dataset == ".gitignore":
		continue
	## umap embeddings
	for j in range(20):
		if os.path.exists(f"./scatterplots/{datasets_list[i]}_umap_{j}.npy"):
			continue
		try:
			n_neighbors = np.random.randint(2, 200)
			min_dist = np.random.uniform(0.01, 1.0)
			umap_embedding = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist).fit_transform(dataset)
			np.save(f"./scatterplots/{datasets_list[i]}_umap_{j}.npy", umap_embedding)
		except:
			pass
	## tsne embeddings
	for j in range(20):
		if os.path.exists(f"./scatterplots/{datasets_list[i]}_tsne_{j}.npy"):
			continue
		try:
			tsne_embedding = TSNE(n_components=2, perplexity=np.random.randint(2, 500)).fit_transform(dataset)
			np.save(f"./scatterplots/{datasets_list[i]}_tsne_{j}.npy", tsne_embedding)
		except:
			pass
	## densmap embeddings
	for j in range(20):
		if os.path.exists(f"./scatterplots/{datasets_list[i]}_densmap_{j}.npy"):
			continue
		try:
			n_neighbors = np.random.randint(2, 200)
			min_dist = np.random.uniform(0.01, 1.0)
			densmap_embedding = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, densmap=True).fit_transform(dataset)
			np.save(f"./scatterplots/{datasets_list[i]}_densmap_{j}.npy", densmap_embedding)
		except:
			pass
	## ISOMAP embeddings
	for j in range(20):
		if os.path.exists(f"./scatterplots/{datasets_list[i]}_isomap_{j}.npy"):
			continue
		try:
			isomap_embedding = Isomap(n_neighbors=np.random.randint(2, 200)).fit_transform(dataset)
			np.save(f"./scatterplots/{datasets_list[i]}_isomap_{j}.npy", isomap_embedding)
		except:
			pass
	## LLE embeddings
	for j in range(20):
		if os.path.exists(f"./scatterplots/{datasets_list[i]}_lle_{j}.npy"):
			continue
		try:
			lle_embedding = LocallyLinearEmbedding(n_neighbors=np.random.randint(2, 200)).fit_transform(dataset)
			np.save(f"./scatterplots/{datasets_list[i]}_lle_{j}.npy", lle_embedding)
		except:
			pass
	## MDS embeddings
	if os.path.exists(f"./scatterplots/{datasets_list[i]}_mds.npy"):
		continue
	mds_embedding = MDS().fit_transform(dataset)
	np.save(f"./scatterplots/{datasets_list[i]}_mds.npy", mds_embedding)
	## PCA embeddings
	if os.path.exists(f"./scatterplots/{datasets_list[i]}_pca.npy"):
		continue
	pca_embedding = PCA(n_components=2).fit_transform(dataset)
	np.save(f"./scatterplots/{datasets_list[i]}_pca.npy", pca_embedding)
	## Random embeddings
	for j in range(20):
		if os.path.exists(f"./scatterplots/{datasets_list[i]}_rp_{j}.npy"):
			continue
		try:
			rp_embedding = GaussianRandomProjection(n_components=2).fit_transform(dataset)
			np.save(f"./scatterplots/{datasets_list[i]}_rp_{j}.npy", rp_embedding)
		except:
			pass



	