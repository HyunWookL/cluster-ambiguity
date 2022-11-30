import importlib
import numpy as np

import AutoSS as autoss

import matplotlib.pyplot as plt
import os, json
import seaborn as sns

### read datasets

zipfile = np.load("./data_sampling/crowdsourced_mapping.npz", allow_pickle=True)
data, labels = zipfile["positions"], zipfile["labels"]

auss = autoss.AutoScatterplotSampling(verbose=2, init_points=1, n_iter=2)
results = auss.fit(data, labels)

emb = results["sampled_point"]
ll  = results["sampled_label"]

fig, ax = plt.subplots(1, 2, figsize=(10, 5))

ax[0].scatter(data[:, 0], data[:, 1], c=labels, s=3, cmap="tab10")
ax[1].scatter(emb[:,0], emb[:,1], c=ll, s=3, cmap="tab10")

plt.savefig("./autoss_results/plots/crowdsourced_mapping.png", dpi=300)
plt.savefig("./autoss_results/plots/crowdsourced_mapping.pdf", dpi=300)

with open("./autoss_results/results/crowdsourced_mapping.json", "w") as f:
		json.dump(results, f)
