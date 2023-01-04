
import numpy as np
import os, json
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from kneed import KneeLocator

dataset_list = os.listdir("./scatterplots")
dataset_list.remove(".gitignore")

with open("./measures/dataset_list.json", "w") as f:
	json.dump(dataset_list, f)

gmm_vec = []

if not os.path.exists("./measures/gmm_vec.npy"):
	for i, scatterplot_file in tqdm(enumerate(dataset_list)):
		splot = np.load(f"./scatterplots/{scatterplot_file}")
		bics = []
		xlist = list(range(1, 30))
		for i in xlist:
			gmm = GaussianMixture(n_components=i)
			gmm.fit(splot)
			bics.append(gmm.bic(splot))
		kneedle = KneeLocator(xlist, bics, curve='convex', direction="decreasing")
		gmm_vec.append(kneedle.knee)
	print(gmm_vec)
	np.save("./measures/gmm_vec.npy", np.array(gmm_vec))

else:
	gmm_vec = np.load("./measures/gmm_vec.npy")




## save results
# np.save("./scagnostics/result.npy", scagnostics_vec)
# with open("./scagnostics/scagnostics_list.json", "w") as f:
# 	json.dump(scagnostics_list, f)
# with open("./scagnostics/dataset_list.json", "w") as f:
# 	json.dump(dataset_list, f)





	

	
