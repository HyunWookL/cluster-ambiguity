import numpy as np

def decompose_covariance_matrix(covariance_matrix):
	"""
	decompose the covariance matrix into its principal components
	"""
	U, s, Vt = np.linalg.svd(covariance_matrix, full_matrices=True)
	scaling = np.sqrt(s).tolist()
	rotation = np.arccos(U[0, 1])
	rotation_degree = rotation * (180 / np.pi)

	return scaling, rotation, rotation_degree


def construct_reg_input_variables(gaussian_info, idx_0, idx_1):

	## extract rotation info
	rotation_diff = gaussian_info["rotation"][idx_0] - gaussian_info["rotation"][idx_1]
	rotation_average = (gaussian_info["rotation"][idx_0] + gaussian_info["rotation"][idx_1]) / 2

	## extract scaling info
	scaling_x_diff = gaussian_info["scaling"][idx_0][0] - gaussian_info["scaling"][idx_1][0]
	scaling_y_diff = gaussian_info["scaling"][idx_0][1] - gaussian_info["scaling"][idx_1][1]
	scaling_diff = np.sqrt(scaling_x_diff**2 + scaling_y_diff**2)

	## extract mean info
	mean_x_diff = gaussian_info["means"][idx_0][0] - gaussian_info["means"][idx_1][0]
	mean_y_diff = gaussian_info["means"][idx_0][1] - gaussian_info["means"][idx_1][1]
	mean_diff = np.sqrt(mean_x_diff**2 + mean_y_diff**2)

	## extract mean vector angle info
	mean_vector_angle = np.arctan2(mean_y_diff, mean_x_diff) + np.pi / 2
	gaussian_mean_vector_angle_0 = np.abs(mean_vector_angle - gaussian_info["rotation"][idx_0])
	gaussian_mean_vector_angle_1 = np.abs(mean_vector_angle - gaussian_info["rotation"][idx_1])
	gaussian_mean_vector_angle_0 = np.min([gaussian_mean_vector_angle_0, np.pi - gaussian_mean_vector_angle_0])
	gaussian_mean_vector_angle_1 = np.min([gaussian_mean_vector_angle_1, np.pi - gaussian_mean_vector_angle_1])

	gaussian_mean_vector_angle_diff = np.abs(gaussian_mean_vector_angle_0 - gaussian_mean_vector_angle_1)
	gaussian_mean_vector_angle_average = (gaussian_mean_vector_angle_0 + gaussian_mean_vector_angle_1) / 2

	## extract the ratio of mean and scaling
	scaling_size = np.linalg.norm(gaussian_info["scaling"][idx_0]) + np.linalg.norm(gaussian_info["scaling"][idx_1])
	scaling_size_diff = np.abs(np.linalg.norm(gaussian_info["scaling"][idx_0]) - np.linalg.norm(gaussian_info["scaling"][idx_1]))
	mean_diff_scaling_ratio = mean_diff / scaling_size

	## extract the ellipticity difference and mean
	ellipsivity_0 = np.max(gaussian_info["scaling"][idx_0]) / np.min(gaussian_info["scaling"][idx_0])
	ellipsivity_1 = np.max(gaussian_info["scaling"][idx_1]) / np.min(gaussian_info["scaling"][idx_1])
	ellipticity_diff = np.abs(ellipsivity_0 - ellipsivity_1)
	ellipticity_average = (ellipsivity_0 + ellipsivity_1) / 2

	## extract the density difference and mean
	labels = np.array(gaussian_info["proba_labels"])
	density_0 = np.count_nonzero(labels == idx_0) / (gaussian_info["scaling"][idx_0][0] * gaussian_info["scaling"][idx_1][1])
	density_1 = np.count_nonzero(labels == idx_1) / (gaussian_info["scaling"][idx_0][0] * gaussian_info["scaling"][idx_1][1])
	density_diff = np.abs(density_0 - density_1)
	density_average = (density_0 + density_1) / 2

	return {
		"rotation_diff": rotation_diff,
		"rotation_average": rotation_average,
		"scaling_diff": scaling_diff,
		"mean_diff": mean_diff,
		"scaling_size": scaling_size,
		"scaling_size_diff": scaling_size_diff,
		"mean_diff_scaling_ratio": mean_diff_scaling_ratio,
		"ellipticity_diff": ellipticity_diff,
		"ellipticity_average": ellipticity_average,
		"density_diff": density_diff,
		"density_average": density_average,
		"gaussian_mean_vector_angle_diff": gaussian_mean_vector_angle_diff,
		"gaussian_mean_vector_angle_average": gaussian_mean_vector_angle_average,
	}