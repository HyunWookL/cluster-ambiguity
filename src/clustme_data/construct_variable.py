"""
Predicting Prob Single based on the extracted gaussian info

input variables: rotation_diff, density_diff, scaling_diff, mean_diff
target variable: prob_single
"""

import pandas as pd 
import numpy as np
import os, json
from helpers import construct_reg_input_variables

def construct_variables():
	"""
	construct variables based on clustme data
	"""
	rotation_diff_list = []
	rotation_average_list = []
	# weight_diff_list = []
	scaling_diff_list = []
	scaling_x_diff_list = []
	scaling_y_diff_list = []
	mean_diff_list = []
	gaussian_mean_vector_angle_diff_list = []
	gaussian_mean_vector_angle_average_list = []
	prob_single_list = []

	scaling_size_list = []
	scaling_size_diff_list = []
	mean_diff_scaling_ratio_list = []

	ellipticity_diff_list = []
	ellipticity_average_list = []

	density_diff_list = []
	density_average_list = []
	for idx in os.listdir("./extracted"):
		if(idx == ".gitignore"):
			continue
		with open("./extracted/" + idx, "r") as f:
			datum = json.load(f)
			prob_single = datum["prob_single"]
			gaussian_info = datum["gaussian_info"]

			input_variables = construct_reg_input_variables(gaussian_info, 0, 1)

			rotation_diff_list.append(input_variables["rotation_diff"])
			rotation_average_list.append(input_variables["rotation_average"])
			scaling_diff_list.append(input_variables["scaling_diff"])
			mean_diff_list.append(input_variables["mean_diff"])
			scaling_size_list.append(input_variables["scaling_size"])
			scaling_size_diff_list.append(input_variables["scaling_size_diff"])
			mean_diff_scaling_ratio_list.append(input_variables["mean_diff_scaling_ratio"])
			ellipticity_diff_list.append(input_variables["ellipticity_diff"])
			ellipticity_average_list.append(input_variables["ellipticity_average"])
			density_diff_list.append(input_variables["density_diff"])
			density_average_list.append(input_variables["density_average"])
			gaussian_mean_vector_angle_diff_list.append(input_variables["gaussian_mean_vector_angle_diff"])
			gaussian_mean_vector_angle_average_list.append(input_variables["gaussian_mean_vector_angle_average"])


	
			prob_single_list.append(prob_single)
	
	df = pd.DataFrame({
		"rotation_diff": rotation_diff_list,
		"rotation_average": rotation_average_list,
		"scaling_diff": scaling_diff_list,
		"mean_diff": mean_diff_list,
		"prob_single": prob_single_list,
		"scaling_size": scaling_size_list,
		"scaling_size_diff": scaling_size_diff_list,
		"mean_diff_scaling_ratio": mean_diff_scaling_ratio_list,
		"ellipticity_diff": ellipticity_diff_list,
		"ellipticity_average": ellipticity_average_list,
		"density_diff": density_diff_list,
		"density_average": density_average_list,
		"gaussian_mean_vector_angle_diff": gaussian_mean_vector_angle_diff_list,
		"gaussian_mean_vector_angle_average": gaussian_mean_vector_angle_average_list
	})


	df.to_csv("./variables/variables.csv", index=False)

construct_variables()




			# ## extract rotation difference			
			# rotation_diff = gaussian_info["rotation"][0] - gaussian_info["rotation"][1]
			# rotation_average = (gaussian_info["rotation"][0] + gaussian_info["rotation"][1]) / 2

			# ## extract density difference
			# # weight_diff = np.abs(gaussian_info["weights"][0] - gaussian_info["weights"][1]) / (gaussian_info["weights"][0] + gaussian_info["weights"][1])

			# ## extract scaling difference			
			# scaling_x_diff = gaussian_info["scaling"][0][0] - gaussian_info["scaling"][1][0]
			# scaling_y_diff = gaussian_info["scaling"][0][1] - gaussian_info["scaling"][1][1]
			# scaling_diff = np.sqrt(scaling_x_diff**2 + scaling_y_diff**2)


			# ## extract means difference			
			# mean_x_diff = gaussian_info["means"][0][0] - gaussian_info["means"][1][0]
			# mean_y_diff = gaussian_info["means"][0][1] - gaussian_info["means"][1][1]
			# mean_diff = np.sqrt(mean_x_diff**2 + mean_y_diff**2)


			# mean_vector_angle = np.arctan2(mean_y_diff, mean_x_diff) + np.pi / 2
			# gaussian_mean_vector_angle_0 = np.abs(mean_vector_angle - gaussian_info["rotation"][0])
			# gaussian_mean_vector_angle_1 = np.abs(mean_vector_angle - gaussian_info["rotation"][1])
			# gaussian_mean_vector_angle_0 = np.min([gaussian_mean_vector_angle_0, np.pi - gaussian_mean_vector_angle_0])
			# gaussian_mean_vector_angle_1 = np.min([gaussian_mean_vector_angle_1, np.pi - gaussian_mean_vector_angle_1])
			
			# gaussian_mean_vector_angle_diff = np.abs(gaussian_mean_vector_angle_0 - gaussian_mean_vector_angle_1)
			# gaussian_mean_vector_angle_average = (gaussian_mean_vector_angle_0 + gaussian_mean_vector_angle_1) / 2



			# ## extract the ratio of mean and scaling
			# scaling_size = np.linalg.norm(gaussian_info["scaling"][0]) + np.linalg.norm(gaussian_info["scaling"][1])
			# scaling_size_diff = np.abs(np.linalg.norm(gaussian_info["scaling"][0]) - np.linalg.norm(gaussian_info["scaling"][1]))
			# mean_diff_scaling_ratio = mean_diff / scaling_size

			# ## extract the ellipticity difference and mean
			# ellipsivity_0 = np.max(gaussian_info["scaling"][0]) / np.min(gaussian_info["scaling"][0])
			# ellipsivity_1 = np.max(gaussian_info["scaling"][1]) / np.min(gaussian_info["scaling"][1])
			# ellipticity_diff = np.abs(ellipsivity_0 - ellipsivity_1)
			# ellipticity_average = (ellipsivity_0 + ellipsivity_1) / 2

			# ## extract the density difference and mean
			# labels = np.array(gaussian_info["proba_labels"])
			# density_0 = (len(labels) - np.count_nonzero(labels)) / (gaussian_info["scaling"][0][0] * gaussian_info["scaling"][0][1])
			# density_1 =  np.count_nonzero(labels) / (gaussian_info["scaling"][1][0] * gaussian_info["scaling"][1][1])
			# density_diff = np.abs(density_0 - density_1)
			# density_average = (density_0 + density_1) / 2
