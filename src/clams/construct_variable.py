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
	rotation_sine_list = []
	scaling_diff_list = []
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
	for idx in os.listdir("./extracted_revised"):
		if(idx == ".gitignore"):
			continue
		with open("./extracted/" + idx, "r") as f:
			datum = json.load(f)
			prob_single = datum["prob_single"]
			gaussian_info = datum["gaussian_info"]

			input_variables = construct_reg_input_variables(gaussian_info, 0, 1)

			rotation_diff_list.append(input_variables["rotation_diff"])
			rotation_average_list.append(input_variables["rotation_average"])
			rotation_sine_list.append(input_variables["rotation_sine"])
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
		"rotation_sine": rotation_sine_list,
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

