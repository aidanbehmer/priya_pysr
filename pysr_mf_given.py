import time

import numpy as np
import matplotlib.pyplot as plt
from typing import List
import pysr
import h5py
import sympy
import numpy as np
from matplotlib import pyplot as plt
from pysr import PySRRegressor
from sklearn.model_selection import train_test_split

# Configure plot defaults
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["grid.color"] = "#666666"

####### Set Input Arguments ########
# This is where you set the args to your function.
# Parameter name
param_name = "Ap"
# take z = 3.6
z = 3.6
# random seed for reproducibility
random_state = 42
# PySR settings
niterations = 20
maxsize = 20
maxdepth = 10
# Plotting
quantile_low = 0.25  # quantile for parameter value to fix
quantile_high = 0.75  # quantile for parameter value to fix
####################################

param_dict = {
    "dtau0": 0,
    "tau0": 1,
    "ns": 2,
    "Ap": 3,
    "herei": 4,
    "heref": 5,
    "alphaq": 6,
    "hub": 7,
    "omegamh2": 8,
    "hireionz": 9,
    "bhfeedback": 10,
}
param_idx = param_dict[param_name]  # index of the parameter in the params array

# TODO: Probably also be careful about the filepath~
with h5py.File(
    "../InferenceMultiFidelity/1pvar/lf_{}_npoints50_datacorrFalse.hdf5".format(param_name), "r"
) as file:
    print(file.keys())

    flux_vectors_low = file["flux_vectors"][:]
    kfkms_low = file["kfkms"][:]
    # kfmpc = file["kfmpc"][:]
    zout = file["zout"][:]
    resolution_low=np.full((1750,1),0)

    params_low = file["params"][:]
#kfkms.shape, flux_vectors.shape, zout.shape, params.shape

with h5py.File(
    "../InferenceMultiFidelity/1pvar/hf_{}_npoints50_datacorrFalse.hdf5".format(param_name), "r"
) as file:
    print(file.keys())

    flux_vectors_hi = file["flux_vectors"][:]
    kfkms_hi = file["kfkms"][:]
    # kfmpc = file["kfmpc"][:]
    zout_hi = file["zout"][:]
    resolution_hi=np.full((1750,1),1)
    params_hi = file["params"][:]

zindex = np.where(zout == z)[0][0]  # index of z = 5

# take z=3.6, and flatten the flux vectors, such that the dim=1 is p1d values per k and parameter
flux_vectors_z_low = flux_vectors_low[:, zindex, :]
flux_vectors_z_low = flux_vectors_z_low.flatten()[:, np.newaxis]  # add a new axis to make it 2D

flux_vectors_z_hi = flux_vectors_hi[:, zindex, :]
flux_vectors_z_hi = flux_vectors_z_hi.flatten()[:, np.newaxis]  # add a new axis to make it 2D

# do the same for kfkms
kfkms_z_low = kfkms_low[:, zindex, :]
kfkms_z_low = kfkms_z_low.flatten()[:, np.newaxis]  # add a new axis to make it 2D

params_values_low = params_low[:, param_idx]
# repeat this for the number of kfkms
params_values_low = np.repeat(params_values_low[:, np.newaxis], kfkms_low.shape[2], axis=1)
params_values_low = params_values_low.flatten()[:, np.newaxis]  # add a new axis to make it 2D

# Shapes: (1750, 1)
X_param = params_values_low
X_k = kfkms_z_low
y = flux_vectors_z_low
assert(y.shape == (1750, 1))
# Concatenate inputs to form design matrix
X = np.hstack([X_param, X_k])  # shape: (1750, 2)

X_1 = np.hstack([X_param, X_k,resolution_low])  # shape: (1750, 2)
assert(X.shape== (1750, 2))

params_values_hi = params_low[:, param_idx]
# repeat this for the number of kfkms
params_values_hi = np.repeat(params_values_hi[:, np.newaxis], kfkms_low.shape[2], axis=1)
params_values_hi = params_values_hi.flatten()[:, np.newaxis]  # add a new axis to make it 2D

# Shapes: (1750, 1)
X_param_hi = params_values_hi

y_hi = flux_vectors_z_hi
X2=np.hstack([X_param_hi, X_k])
X_2=np.hstack([X_param_hi, X_k,resolution_hi])  # shape: (1750, 2)
X_act=np.vstack([X_1, X_2])  # shape: (3500, 2)
Y_act=np.vstack([y, y_hi])  # shape: (3500, 1)
assert(X_act.shape== (3500, 3))
assert(Y_act.shape== (3500, 1))

from pysr import PySRRegressor

model = PySRRegressor(
    model_selection="best",
    niterations=niterations,  # increase for better expressions
    binary_operators=["+", "*", "-", "/", "^"],
    unary_operators=[
        "sin", "cos", "exp", "log", "square", "sqrt","inv(x) = 1/x" #removed sqrt because of degeneracy between it and "^"
    ],
    extra_sympy_mappings={"inv": lambda x: 1 / x},
    loss="loss(x, y) = (x - y)^2",
    maxsize=maxsize,
    maxdepth=maxdepth,
    verbosity=1,
    random_state=random_state,
)

model.fit(X_act, Y_act)

# print(model)
# TODO: Find a way to save the best model to a file~
model.equations_  # to see all candidate expressions


# TODO: Separate the plotting code into another function
# Choose a parameter value to fix
param_fixed_low = np.quantile(params_low[:, param_idx], quantile_low)
param_fixed_hi = np.quantile(params_hi[:, param_idx], quantile_low)
# Get unique k values from your data
k_values_low = np.unique(X_1[:, 1])

# Get unique k values from your data
k_values_hi = np.unique(X_2[:, 1])

# Construct input (param_fixed, k) pairs
X_pred_low = np.column_stack([
    np.full_like(k_values_low, fill_value=param_fixed_low),
    k_values_low,
    np.zeros_like(k_values_low) #NEW
])
X_pred_hi = np.column_stack([
    np.full_like(k_values_hi, fill_value=param_fixed_hi),
    k_values_hi,
    np.ones_like(k_values_hi) #NEW
])
#X_pred_hi = np.column_stack([
 #   np.full_like(k_values, fill_value=param_fixed_hi),
  #  k_values
#])

# Get model prediction
flux_pred_low = model.predict(X_pred_low)
flux_pred_hi = model.predict(X_pred_hi)
#flux_pred = model.predict(X_pred_hi)

# To compare with ground truth, find the closest match in your original param list
# (Assumes you used params[:,2] to construct your X)
param_list = np.unique(X[:, 0])
idx_closest = np.argmin(np.abs(param_list - param_fixed_low))
idx_closest_hi = np.argmin(np.abs(param_list - param_fixed_hi)) #change
param_closest = param_list[idx_closest]
param_closest_hi = param_list[idx_closest_hi] #change

# Now find all rows in X with that param value
mask = X[:, 0] == param_closest
#mask_hi = X2[:, 0] == param_closest
mask_hi = X2[:, 0] == param_closest_hi #change
k_true = X[mask, 1]
flux_true_low = y[mask, 0]  # shape: (N,)

k_true_hi = X2[mask_hi, 1]
flux_true_hi = y_hi[mask_hi, 0]

# Sort both by k to ensure aligned plotting
sort_idx_true = np.argsort(k_true)
sort_idx_true_hi = np.argsort(k_true_hi)
sort_idx_pred_low = np.argsort(k_values_low)
sort_idx_pred_hi = np.argsort(k_values_hi)

# Plot
plt.figure(figsize=(6, 4))
plt.plot(k_true[sort_idx_true], flux_true_low[sort_idx_true], label="True Flux (Low)", lw=2, color="C5")
plt.plot(k_true_hi[sort_idx_true_hi], flux_true_hi[sort_idx_true_hi], label="True Flux (High)", lw=2, color="C6")
plt.plot(k_values_low[sort_idx_pred_low], flux_pred_low[sort_idx_pred_low], label="PySR Prediction (Low)", lw=2, linestyle="--", color="C7")
plt.plot(k_values_hi[sort_idx_pred_hi], flux_pred_hi[sort_idx_pred_hi], label="PySR Prediction (High)", lw=2, linestyle="--", color="C8")




#ANOTHER PARAM
# TODO: Probably make this into a for loop instead of copy-pasting~

# Choose a parameter value to fix
param_fixed_low = np.quantile(params_low[:, param_idx], quantile_high)
param_fixed_hi = np.quantile(params_hi[:, param_idx], quantile_high)
# Get unique k values from your data
k_values_low = np.unique(X_1[:, 1])

# Get unique k values from your data
k_values_hi = np.unique(X_2[:, 1])

# Construct input (param_fixed, k) pairs
X_pred_low = np.column_stack([
    np.full_like(k_values_low, fill_value=param_fixed_low),
    k_values_low,
    np.zeros_like(k_values_low) #NEW
])
X_pred_hi = np.column_stack([
    np.full_like(k_values_hi, fill_value=param_fixed_hi),
    k_values_hi,
    np.ones_like(k_values_hi) #NEW
])
#X_pred_hi = np.column_stack([
 #   np.full_like(k_values, fill_value=param_fixed_hi),
  #  k_values
#])

# Get model prediction
flux_pred_low = model.predict(X_pred_low)
flux_pred_hi = model.predict(X_pred_hi)
#flux_pred = model.predict(X_pred_hi)

# To compare with ground truth, find the closest match in your original param list
# (Assumes you used params[:,2] to construct your X)
param_list = np.unique(X[:, 0])
idx_closest = np.argmin(np.abs(param_list - param_fixed_low))
idx_closest_hi = np.argmin(np.abs(param_list - param_fixed_hi)) #change
param_closest = param_list[idx_closest]
param_closest_hi = param_list[idx_closest_hi] #change

# Now find all rows in X with that param value
mask = X[:, 0] == param_closest
#mask_hi = X2[:, 0] == param_closest
mask_hi = X2[:, 0] == param_closest_hi #change
k_true = X[mask, 1]
flux_true_low = y[mask, 0]  # shape: (N,)

k_true_hi = X2[mask_hi, 1]
flux_true_hi = y_hi[mask_hi, 0]

# Sort both by k to ensure aligned plotting
sort_idx_true = np.argsort(k_true)
sort_idx_true_hi = np.argsort(k_true_hi)
sort_idx_pred_low = np.argsort(k_values_low)
sort_idx_pred_hi = np.argsort(k_values_hi)

# Plot
#plt.figure(figsize=(6, 4))
plt.plot(k_true[sort_idx_true], flux_true_low[sort_idx_true], label="True Flux (Low)", lw=2, color="C1")
plt.plot(k_true_hi[sort_idx_true_hi], flux_true_hi[sort_idx_true_hi], label="True Flux (High)", lw=2, color="C2")
plt.plot(k_values_low[sort_idx_pred_low], flux_pred_low[sort_idx_pred_low], label="PySR Prediction (Low)", lw=2, linestyle="--", color="C3")
plt.plot(k_values_hi[sort_idx_pred_hi], flux_pred_hi[sort_idx_pred_hi], label="PySR Prediction (High)", lw=2, linestyle="--", color="C4")


print(model.get_best())
#print(f"Parameter fixed: {param_fixed_low:.2f} (low fidelity), {param_fixed_hi:.2f} (high fidelity)")
# TODO: Save the figure to a file too~ you can use plt.savefig("filename.png")
plt.xlabel("k [s/km]")
plt.ylabel("P1D(k)")
plt.title(f"z = {z:.2f}, param ≈ {param_fixed_low:.2f}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()