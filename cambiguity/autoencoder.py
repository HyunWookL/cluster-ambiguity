from tensorflow.keras import Model
from tensorflow.keras import layers, losses
import tensorflow as tf
import tensorflow_addons as tfa

class ClusterAutoEncoder(Model):
	def __init__(self, grid_size, latent_dim):
		super(ClusterAutoEncoder, self).__init__()
		self.grid_size = grid_size
		self.encoder = tf.keras.Sequential([
			layers.InputLayer(input_shape=(grid_size, grid_size, 1)),
			layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', strides=2),
			layers.Conv2D(filters=16, kernel_size=3, padding='same', activation='relu', strides=2),
			layers.Conv2D(filters=8, kernel_size=3, padding='same', activation='relu', strides=2),
			layers.Conv2D(filters=4, kernel_size=3, padding='same', activation='relu', strides=2),
			layers.Flatten(),
			layers.Dense(units=64, activation='relu'),
			layers.Dense(units=32, activation='relu'),
			layers.Dense(units=latent_dim, activation='relu')
		])

		final_max_pool_size = grid_size // (2 ** 4) #4
		final_max_pool_filter = 4

		self.decoder = tf.keras.Sequential([
			layers.Dense(units=32, activation='relu'),
			layers.Dense(units= final_max_pool_size * final_max_pool_size * final_max_pool_filter, activation='relu'),
			layers.Reshape(target_shape=(final_max_pool_size, final_max_pool_size, final_max_pool_filter)),
			layers.Conv2DTranspose(filters=8, kernel_size=3, padding='same', activation='relu', strides=2),
			layers.Conv2DTranspose(filters=16, kernel_size=3, padding='same', activation='relu', strides=2),
			layers.Conv2DTranspose(filters=32, kernel_size=3, padding='same', activation='relu', strides=2),
			layers.Conv2DTranspose(filters=1, kernel_size=3, padding='same', activation='relu', strides=2),
			# layers.Conv2DTranspose(filters=16, kernel_size=3, padding='same', activation='relu', strides=2),
			# layers.Conv2DTranspose(filters=32, kernel_size=3, padding='same', activation='relu', strides=2),
		])
	
	def call(self, x):
		encoded = self.encoder(x)
		decoded = self.decoder(encoded)
		return decoded

# clusterautoencoder = ClusterAutoEncoder(grid_size=64)

# clusterautoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

# clusterautoencoder.build(input_shape=(None, 64, 64, 1))
# print(clusterautoencoder.layers[0].summary())
# print(clusterautoencoder.layers[1].summary())