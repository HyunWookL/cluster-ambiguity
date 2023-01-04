
import numpy as np
import os, json
from pyscagnostics import scagnostics
from tqdm import tqdm

dataset_list = os.listdir("./scatterplots")
dataset_list.remove(".gitignore")


scagnostics_list = [
	"Outlying",
	"Skewed",
	"Clumpy",
	"Sparse",
	"Striated",
	"Convex",
	"Skinny",
	"Stringy",
	"Monotonic",
]

scagnostics_vec = np.zeros((len(dataset_list), len(scagnostics_list)))

if not os.path.exists("./measures/scagnostics_vec.npy"):
	
	for i, scatterplot_file in enumerate(dataset_list):
		print(f"{i}/{len(dataset_list)}")
		splot = np.load(f"./scatterplots/{scatterplot_file}")
		x = splot[:, 0]
		y = splot[:, 1]
		scags, _ = scagnostics(x, y)
		for j, scag in enumerate(scagnostics_list):
			scagnostics_vec[i, j] = scags[scag]
	np.save("./measures/scagnostics_vec.npy", scagnostics_vec)
else:
	scagnostics_vec = np.load("./measures/scagnostics_vec.npy")


## save results
# np.save("./scagnostics/result.npy", scagnostics_vec)
# with open("./scagnostics/scagnostics_list.json", "w") as f:
# 	json.dump(scagnostics_list, f)
# with open("./scagnostics/dataset_list.json", "w") as f:
# 	json.dump(dataset_list, f)





	

	
