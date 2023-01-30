import pandas as pd
import autosklearn.regression
import matplotlib.pyplot as plt
import copy
import pickle
 


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler



def read_variables(input_arr):
	"""
	read variables from clustme data
	returns input variables and target variable as numpy arrays
	"""
	df = pd.read_csv("./variables/variables.csv")
	## split df into input variables and target variable
	input_variables = df.get(input_arr).to_numpy()
	target_variable = df.get(["prob_single"]).to_numpy()

	# scaler = StandardScaler()
	# input_variables = scaler.fit_transform(input_variables)

	return input_variables, target_variable

def plot(t, t_pred, score, title):
	plt.scatter(t, t_pred)
	plt.ylabel("Predicted")
	plt.xlabel("Actual")
	plt.title("Score: " + str(score.mean()))
	plt.savefig("./plot_prediction/" + title + ".png")
	plt.clf()


def run_linear_regression(input_arr):
	"""
	Train a linear regression model on the input variables 
	perform cross-validation on the target variable
	"""
	i, t = read_variables(input_arr)
	reg = LinearRegression()
	t_pred = cross_val_predict(reg, i, t, cv=5)
	score = cross_val_score(reg, i, t, cv=5)
	plot(t, t_pred, score, "linear_regression")

def run_polynomial_regression(input_arr, degree):
	"""
	Train a polynomial regression model on the input variables 
	perform cross-validation on the target variable
	"""
	i, t = read_variables(input_arr)
	poly = PolynomialFeatures(degree=degree)
	i_poly = poly.fit_transform(i)
	reg = LinearRegression()
	t_pred = cross_val_predict(reg, i_poly, t, cv=5)
	score = cross_val_score(reg, i_poly, t, cv=5)
	plot(t, t_pred, score, "polynomial_regression_" + str(degree))

def run_mlp_regression(input_arr):
	"""
	Train a neural network regression model on the input variables 
	perform cross-validation on the target variable
	"""
	i, t = read_variables(input_arr)
	reg = MLPRegressor(hidden_layer_sizes=(100,100, 100,), max_iter=1000)
	t_pred = cross_val_predict(reg, i, t, cv=5)
	score = cross_val_score(reg, i, t, cv=5)
	plot(t, t_pred, score, "mlp_regression")

def run_random_forest_regression(input_arr):
	"""
	Train a random forest regression model on the input variables 
	perform cross-validation on the target variable
	"""
	i, t = read_variables(input_arr)
	reg = RandomForestRegressor(n_estimators=100)
	t_pred = cross_val_predict(reg, i, t, cv=5)
	score = cross_val_score(reg, i, t, cv=5)
	plot(t, t_pred, score, "random_forest_regression")

def run_svr_regression(input_arr):
	"""
	Train a support vector regression model on the input variables 
	perform cross-validation on the target variable
	"""
	i, t = read_variables(input_arr)
	reg = SVR(kernel="rbf")
	t_pred = cross_val_predict(reg, i, t, cv=5)
	score = cross_val_score(reg, i, t, cv=5)
	plot(t, t_pred, score, "svr_regression")

def run_knn_regression(input_arr):
	"""
	Train a k-nearest neighbors regression model on the input variables 
	perform cross-validation on the target variable
	"""
	i, t = read_variables(input_arr)
	reg = KNeighborsRegressor(n_neighbors=10)
	t_pred = cross_val_predict(reg, i, t, cv=5)
	score = cross_val_score(reg, i, t, cv=5)
	plot(t, t_pred, score, "knn_regression")

def run_extra_trees_regression(input_arr):
	"""
	Train a extra trees regression model on the input variables 
	perform cross-validation on the target variable
	"""
	i, t = read_variables(input_arr)
	reg = ExtraTreesRegressor(n_estimators=100)
	t_pred = cross_val_predict(reg, i, t, cv=5)
	score = cross_val_score(reg, i, t, cv=5)
	plot(t, t_pred, score, "extra_trees_regression")


def run_autosklearn_based_regression(input_arr):
	"""
	Train a autosklearn based regression model on the input variables 
	perform cross-validation on the target variable
	"""
	i, t = read_variables(input_arr)
	reg = autosklearn.regression.AutoSklearnRegressor(
		time_left_for_this_task=600,                              
		per_run_time_limit=30,
		memory_limit=None,
		resampling_strategy='cv',
		resampling_strategy_arguments={'folds': 5})
	reg.fit(i, t)


	with open("./regression_model/autosklearn.pkl", "wb") as f:
		pickle.dump(reg, f)

	with open("./regression_model/autosklearn.pkl", "rb") as f:
		reg_load = pickle.load(f)

	t_pred = reg_load.predict(i)
	score = reg_load.score(i, t)
	plot(t, t_pred, score, "autosklearn_regression")

	return score




with_scaling_arr = [
	"rotation_diff", 
	"scaling_diff", 
	"mean_diff", 
	"scaling_size", 
	"scaling_size_diff",
	"mean_diff_scaling_ratio",
	"ellipticity_average",
	"ellipticity_diff",
	"density_diff",
	"density_average",
	"rotation_average",
	"gaussian_mean_vector_angle_diff",
	"gaussian_mean_vector_angle_average",
	"rotation_sine"
]

# find_best_variable_combination(with_scaling_arr)




# with_scaling_arr = ["rotation_diff", "density_diff", "scaling_diff", "mean_diff"]

# run_linear_regression(with_scaling_arr)
# run_polynomial_regression(with_scaling_arr, 2)
# run_mlp_regression(with_scaling_arr)
# run_random_forest_regression(with_scaling_arr)
# run_svr_regression(with_scaling_arr)
# run_knn_regression(with_scaling_arr)
# run_extra_trees_regression(with_scaling_arr)

print(run_autosklearn_based_regression(with_scaling_arr))
