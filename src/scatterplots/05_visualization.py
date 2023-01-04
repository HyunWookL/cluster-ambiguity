import json 

import numpy as np
import matplotlib.pyplot as plt



with open("./sampling/sampled_datasets.json", "r") as f:
	sampled_datasets = json.load(f)

num_row = len(sampled_datasets) // 10 + 1
num_col = 10 

fig, axs = plt.subplots(num_row, num_col, figsize=(3 * num_col, 3 * num_row))

for i, dataset in enumerate(sampled_datasets):
	splot = np.load(f"./scatterplots/{dataset}")
	axs[i // num_col, i % num_col].scatter(splot[:, 0], splot[:, 1], s=1)
	## remove ticks
	axs[i // num_col, i % num_col].set_xticks([])
	axs[i // num_col, i % num_col].set_yticks([])

plt.savefig("./sampling/sampled_scatterplots.png", dpi=500)

