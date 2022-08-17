"""
Predicting Prob Single based on the extracted gaussian info

input variables: rotation_diff, density_diff, scaling_diff, mean_diff
target variable: prob_single
"""

import pandas as pd 
import numpy as np
import os, json

def construct_variables():
	"""
	construct variables based on clustme data
	"""
	rotation_diff_list = []
	density_diff_list = []
	scaling_diff_list = []
	scaling_x_diff_list = []
	scaling_y_diff_list = []
	mean_diff_list = []
	prob_single_list = []
	for idx in os.listdir("./extracted"):
		if(idx == ".gitignore"):
			continue
		with open("./extracted/" + idx, "r") as f:
			datum = json.load(f)
			prob_single = datum["prob_single"]
			gaussian_info = datum["gaussian_info"]

			## extract rotation difference			
			rotation_diff = gaussian_info["rotation"][0] - gaussian_info["rotation"][1]

			## extract density difference
			density_diff = (gaussian_info["weights"][0] - gaussian_info["weights"][1]) / (gaussian_info["weights"][0] + gaussian_info["weights"][1])

			## extract scaling difference			
			scaling_x_diff = gaussian_info["scaling"][0][0] - gaussian_info["scaling"][1][0]
			scaling_y_diff = gaussian_info["scaling"][0][1] - gaussian_info["scaling"][1][1]
			scaling_diff = np.sqrt(scaling_x_diff**2 + scaling_y_diff**2)

			## extract means difference			
			mean_x_diff = gaussian_info["means"][0][0] - gaussian_info["means"][1][0]
			mean_y_diff = gaussian_info["means"][0][1] - gaussian_info["means"][1][1]
			mean_diff = np.sqrt(mean_x_diff**2 + mean_y_diff**2)

			rotation_diff_list.append(rotation_diff)
			density_diff_list.append(density_diff)
			scaling_diff_list.append(scaling_diff)
			scaling_x_diff_list.append(scaling_x_diff)
			scaling_y_diff_list.append(scaling_y_diff)
			mean_diff_list.append(mean_diff)
	
			prob_single_list.append(prob_single)
	
	df = pd.DataFrame({
		"rotation_diff": rotation_diff_list,
		"density_diff": density_diff_list,
		"scaling_diff": scaling_diff_list,
		"scaling_x_diff": scaling_x_diff_list,
		"scaling_y_diff": scaling_y_diff_list,
		"mean_diff": mean_diff_list,
		"prob_single": prob_single_list
	})

	df.to_csv("./variables/variables.csv", index=False)

construct_variables()