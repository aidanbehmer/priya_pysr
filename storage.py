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
