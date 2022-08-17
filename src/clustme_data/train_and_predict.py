import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler


import matplotlib.pyplot as plt

def read_variables(input_arr):
	"""
	read variables from clustme data
	returns input variables and target variable as numpy arrays
	"""
	df = pd.read_csv("./variables/variables.csv")
	## split df into input variables and target variable
	input_variables = df.get(input_arr).to_numpy()
	target_variable = df.get(["prob_single"]).to_numpy()

	scaler = StandardScaler()
	input_variables = scaler.fit_transform(input_variables)

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
	reg = KNeighborsRegressor(n_neighbors=5)
	t_pred = cross_val_predict(reg, i, t, cv=5)
	score = cross_val_score(reg, i, t, cv=5)
	plot(t, t_pred, score, "knn_regression")



with_scaling_arr = [
	"rotation_diff", 
	"weight_diff", 
	"scaling_diff", 
	"mean_diff", 
	"scaling_size", 
	"mean_diff_scaling_ratio",
	"ellipticity_average",
	"ellipticity_diff",
	"density_diff",
	"density_average",
]

with_scaling_arr = [
	"rotation_diff", 
	"weight_diff", 
	"scaling_diff", 
	"mean_diff", 
	"scaling_size", 
	"mean_diff_scaling_ratio",
	"ellipticity_average",
	"ellipticity_diff",
	"density_diff",
	"density_average",
]
# with_scaling_arr = ["rotation_diff", "density_diff", "scaling_diff", "mean_diff_scaling_ratio"]

run_linear_regression(with_scaling_arr)
run_polynomial_regression(with_scaling_arr, 2)
run_mlp_regression(with_scaling_arr)
run_random_forest_regression(with_scaling_arr)
run_svr_regression(with_scaling_arr)

run_knn_regression(with_scaling_arr)