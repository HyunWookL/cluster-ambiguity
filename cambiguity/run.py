import importlib
import numpy as np
from tensorflow.keras import layers, losses
import tensorflow as tf

import autoencoder as ae


def dataset_path(name, opacity, radius, pixel_size, grid_size):
	return "_".join([name, str(opacity), str(radius), str(pixel_size), str(grid_size)])


dataset_size = 10000
pixel_size = 1024
grid_size = 64
radius = 3
opacity = 0.2
path = f"./random_clusters/{dataset_path('digitstsne', opacity, radius, pixel_size, grid_size)}"

train_set = np.empty(shape=(dataset_size, grid_size, grid_size))
for i in range(dataset_size):
	curr = np.load(f"{path}/{str(i)}.npy")
	train_set[i, :, :] = curr[:, :]
    
autoencoder = ae.ClusterAutoEncoder(grid_size=64, latent_dim=2)
autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
autoencoder.fit(train_set, train_set, epochs=10, shuffle=True,)