import importlib
import numpy as np

import AutoSS as autoss

import matplotlib.pyplot as plt
import os, json
import seaborn as sns

### read datasets

data_name = "mnist"
identifier = "npz"
init_points = 1
n_iter = 2

zipfile = np.load(f"./data_sampling/{data_name}.{identifier}", allow_pickle=True)
data, labels = zipfile["positions"], zipfile["labels"]



auss = autoss.AutoScatterplotSampling(verbose=2, init_points=20, n_iter=60)
results = auss.fit(data, labels)

emb = results["sampled_point"]
ll  = results["sampled_label"]


fig, ax = plt.subplots(1, 2, figsize=(10, 5))

ax[0].scatter(data[:, 0], data[:, 1], c=labels, s=3, cmap="tab10")
ax[1].scatter(emb[:,0], emb[:,1], c=ll, s=3, cmap="tab10")

results["sampled_point"] = results["sampled_point"].tolist()
results["sampled_label"] = results["sampled_label"].tolist()

results["original_point"] = data.tolist()
results["original_label"] = labels.tolist() 

plt.savefig(f"./autoss_results/plots/{data_name}_{init_points}_{n_iter}.png", dpi=300)
plt.savefig(f"./autoss_results/plots/{data_name}_{init_points}_{n_iter}.pdf", dpi=300)

with open(f"./autoss_results/results/{data_name}_{init_points}_{n_iter}.json", "w") as f:
	json.dump(results, f)