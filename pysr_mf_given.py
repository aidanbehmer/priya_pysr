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
from pysr_model import create_model
from plotting import plot_predictions


# Configure plot defaults
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["grid.color"] = "#666666"

####### Set Input Arguments ########
# This is where you set the args to your function.
# Parameter name
param_name = "ns"  
# take z = 3.6
z = 3.6

# Plotting
quantile_low = 0.16  # quantile for parameter value to fix
quantile_high = 0.84  # quantile for parameter value to fix
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
    resolution_low=np.full((1750,1),0.4)

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
    resolution_hi=np.full((1750,1),0.8)
    params_hi = file["params"][:]

zindex = np.where(zout == z)[0][0]  # index of z = 5

# take z=3.6, and flatten the flux vectors, such that the dim=1 is p1d values per k and parameter
flux_vectors_z_low = flux_vectors_low[:, zindex, :]
# TODO: Check this later: I want the normalized to mean as function of k
mean_flux_low = np.mean(flux_vectors_z_low, axis=0)
std_flux_low = np.std(flux_vectors_z_low, axis=0)
flux_vectors_z_low = (flux_vectors_z_low - mean_flux_low) / std_flux_low  # normalize to mean
#use the mean and std variables later when reverting back to original scale
#make this a function instead of in here
########################################################################
flux_vectors_z_low = flux_vectors_z_low.flatten()[:, np.newaxis]  # add a new axis to make it 2D

flux_vectors_z_hi = flux_vectors_hi[:, zindex, :]
# TODO: Check this later: I want the normalized to mean as function of k
# mean_flux_hi = np.mean(flux_vectors_z_hi, axis=0)
flux_vectors_z_hi = (flux_vectors_z_hi - mean_flux_low) / std_flux_low  # normalize to mean of low fidelity
########################################################################
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


X_max=(np.max(X_param,axis=0))
X_min=(np.min(X_param,axis=0))
X_param_normalized=(X_param-X_min)/(X_max-X_min)
#save the max and min for use in reverting back to original scale
#make this a function as well

X_k_max=np.max(X_k,axis=0)
X_k_min=np.min(X_k,axis=0)
X_k_normalized=(X_k-X_k_min)/(X_k_max-X_k_min)

X = np.hstack([X_param_normalized, X_k_normalized])  # shape: (1750, 2)
X_1 = np.hstack([X_param_normalized, X_k_normalized,resolution_low])  # shape: (1750, 3)
assert(X.shape== (1750, 2))

params_values_hi = params_low[:, param_idx]
# repeat this for the number of kfkms
params_values_hi = np.repeat(params_values_hi[:, np.newaxis], kfkms_low.shape[2], axis=1)
params_values_hi = params_values_hi.flatten()[:, np.newaxis]  # add a new axis to make it 2D

# Shapes: (1750, 1)
X_param_hi = params_values_hi

y_hi = flux_vectors_z_hi

# #normalization of y
# y_low_mean=np.mean(y, axis=0)
# y_low_std=np.std(y, axis=0)
# y_low_normalized=(y-y_low_mean)/y_low_std

# y_hi_mean=np.mean(y_hi, axis=0)
# y_hi_std=np.std(y_hi, axis=0)
# y_hi_normalized=(y_hi-y_low_mean)/y_low_std

#stacking
X_hi_max=np.max(X_param_hi,axis=0)
X_hi_min=np.min(X_param_hi,axis=0)
X_param_hi_normalized=(X_param_hi-X_hi_min)/(X_hi_max-X_hi_min)

X2=np.hstack([X_param_hi_normalized, X_k_normalized]) #non resolution
X_2=np.hstack([X_param_hi_normalized, X_k_normalized,resolution_hi])  # shape: (1750, 3)

#normalization of x
#X_1_normalized=X_1/(np.max(X_1,axis=0)-np.min(X_1,axis=0))
#X_2_normalized=X_2/(np.max(X_2,axis=0)-np.min(X_2,axis=0))
#THROWS ERROR, I BELIEVE BECAUSE OF DIVISION BY 0

#end stacking
X_act=np.vstack([X_1, X_2])  # shape: (3500, 3)
Y_act=np.vstack([y, y_hi])  # shape: (3500, 1)

assert(X_act.shape== (3500, 3))
assert(Y_act.shape== (3500, 1))


#now going into the different files
from pysr import PySRRegressor

model = create_model(niterations=20, maxsize=20, maxdepth=10, random_state=42)
model.fit(X_act, Y_act)

# print(model)
# TODO: Find a way to save the best model to a file~
model.equations_  # to see all candidate expressions

# TODO: patch the param_low and high to normalized versions
params_low_normalized=(params_low-np.min(X_param,axis=0))/(np.max(X_param,axis=0)-np.min(X_param,axis=0))
params_hi_normalized=(params_hi-np.min(X_param_hi,axis=0))/(np.max(X_param_hi,axis=0)-np.min(X_param_hi,axis=0))

#mean_flux_low,std_flux_low
#X_max,X_min, X_hi_max,X_hi_min

#plot_predictions(params_low,params_hi,quantiles,X_hi,X_low,y_hi,y_low,model,z,param_idx,X,X2)
plot_predictions(params_low_normalized,params_hi_normalized,[0.16],X_2,X_1,y_hi,y,model,z,param_idx,X,X2,param_name,mean_flux_low,std_flux_low,X_max,X_min, X_hi_max,X_hi_min,X_k_max,X_k_min)

print(model.get_best())
#print(f"Parameter fixed: {param_fixed_low:.2f} (low fidelity), {param_fixed_hi:.2f} (high fidelity)")
