import numpy as np
import matplotlib.pyplot as plt
import datetime

#add additional variables to map back inputs and outputs to original scale

def denormalize_y(y_normalized, mean_flux_low, std_flux_low):
    return y_normalized * std_flux_low + mean_flux_low

def denormalize_X_param(X_param_normalized, X_max, X_min):
    return X_param_normalized * (X_max - X_min) + X_min

def denormalize_X_k(X_k_normalized, X_k_max, X_k_min):
    return X_k_normalized * (X_k_max - X_k_min) + X_k_min

def plot_predictions(params_low,params_hi,quantiles,X_hi,X_low,y_hi,y_low,model,z,param_idx,X,X2, param,mean_flux_low,std_flux_low,X_max,X_min, X_hi_max,X_hi_min,X_k_max,X_k_min,label_low=0.4,label_hi=0.8):
    ratios_all_low=[]
    ratios_all_hi=[]
    for i in range(len(quantiles)):
       


        param_fixed_low = np.quantile(params_low[:, param_idx], quantiles[i])
        param_fixed_hi = np.quantile(params_hi[:, param_idx], quantiles[i])
        # Get unique k values from your data
        k_values_low = np.unique(X_low[:, 1])

        # Get unique k values from your data
        k_values_hi = np.unique(X_hi[:, 1])

        # Construct input (param_fixed, k) pairs
        X_pred_low = np.column_stack([
            np.full_like(k_values_low, fill_value=param_fixed_low),
            k_values_low,
            np.ones_like(k_values_low) * label_low #NEW
        ])
        X_pred_hi = np.column_stack([
            np.full_like(k_values_hi, fill_value=param_fixed_hi),
            k_values_hi,
            np.ones_like(k_values_hi) * label_hi  #NEW
        ])


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
        flux_true_low = y_low[mask, 0]  # shape: (N,)

        k_true_hi = X2[mask_hi, 1]
        flux_true_hi = y_hi[mask_hi, 0]

        #Sort both by k to ensure aligned plotting
        sort_idx_true = np.argsort(k_true)
        sort_idx_true_hi = np.argsort(k_true_hi)
        sort_idx_pred_low = np.argsort(k_values_low)
        sort_idx_pred_hi = np.argsort(k_values_hi)

        #apply denormalization for y and x here
        flux_true_low = denormalize_y(flux_true_low, mean_flux_low, std_flux_low)
        flux_true_hi = denormalize_y(flux_true_hi, mean_flux_low, std_flux_low) 
        flux_pred_low = denormalize_y(flux_pred_low, mean_flux_low, std_flux_low)
        flux_pred_hi = denormalize_y(flux_pred_hi, mean_flux_low, std_flux_low)

        #denormalize x param
        X_param_denorm_low = denormalize_X_param(X_low[:,0], X_max, X_min)
        X_param_denorm_hi = denormalize_X_param(X_hi[:,0], X_hi_max, X_hi_min)

        #denormalize x k
        k_true = denormalize_X_k(k_true, X_k_max, X_k_min)
        k_true_hi = denormalize_X_k(k_true_hi, X_k_max, X_k_min)
        k_values_low = denormalize_X_k(k_values_low, X_k_max, X_k_min)
        k_values_hi = denormalize_X_k(k_values_hi, X_k_max, X_k_min)

        # Calculate ratio and its inverse for low resolution
        ratio_low = (flux_pred_low[sort_idx_pred_low]) / flux_true_low[sort_idx_true]
        inverse_ratio_low = 1.0 - ratio_low

        # Calculate ratio and its inverse for high resolution
        ratio_hi = flux_pred_hi[sort_idx_pred_hi] / flux_true_hi[sort_idx_true_hi]
        inverse_ratio_hi = 1.0 - ratio_hi
        ratios_all_low.append(inverse_ratio_low)
        ratios_all_hi.append(inverse_ratio_hi)

        diff_low = flux_pred_low[sort_idx_true] - flux_true_low[sort_idx_pred_low]
        diff_hi = flux_pred_hi[sort_idx_true_hi] - flux_true_hi[sort_idx_pred_hi]

        mse_low = np.mean(diff_low**2)
        mse_hi = np.mean(diff_hi**2)
        relative_error_low = np.mean(np.abs(diff_low / flux_true_low[sort_idx_pred_low])) * 100
        relative_error_hi = np.mean(np.abs(diff_hi / flux_true_hi[sort_idx_pred_hi])) * 100


        mean_diff_low = flux_true_hi[sort_idx_true_hi] - np.mean(flux_true_hi[sort_idx_true_hi])
        mean_diff_hi = flux_true_hi[sort_idx_true_hi] - np.mean(flux_true_hi[sort_idx_true_hi])

        r2_low = 1 - (np.sum(diff_low**2) / np.sum(mean_diff_low**2))
        r2_hi = 1 - (np.sum(diff_hi**2) / np.sum(mean_diff_hi**2))
        print("MSE Low: "+str(mse_low))
        print("MSE High: "+str(mse_hi))
        print("Relative Error Low: "+str(relative_error_low))
        print("Relative Error High: "+str(relative_error_hi))
        print("R2 Low: "+str(r2_low))
        print("R2 High: "+str(r2_hi))

        # Plot
        if i==0:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
            fig.subplots_adjust(hspace=0.05)
        """
        plt.plot(k_values_low[sort_idx_pred_low], inverse_ratio_low, label="1-(Pred/True) Low", lw=2, color="C"+str(4*i+5))
        plt.plot(k_values_hi[sort_idx_pred_hi], inverse_ratio_hi, label="1-(Pred/True) High", lw=2, color="C"+str(4*i+6))

        plt.plot(k_true[sort_idx_true], flux_true_low[sort_idx_true], label="True Flux (Low)", lw=2, color="C"+str(4*i+1))
        plt.plot(k_true_hi[sort_idx_true_hi], flux_true_hi[sort_idx_true_hi], label="True Flux (High)", lw=2, color="C"+str(4*i+2))
        plt.plot(k_values_low[sort_idx_pred_low], flux_pred_low[sort_idx_pred_low], label="PySR Prediction (Low)", lw=2, linestyle="--", color="C"+str(4*i+1))
        plt.plot(k_values_hi[sort_idx_pred_hi], flux_pred_hi[sort_idx_pred_hi], label="PySR Prediction (High)", lw=2, linestyle="--", color="C"+str(4*i+2))
        """
        ax1.plot(k_true[sort_idx_true], flux_true_low[sort_idx_true], label="True Flux (Low)", lw=2, color="C"+str(4*i+1))
        ax1.plot(k_true_hi[sort_idx_true_hi], flux_true_hi[sort_idx_true_hi], label="True Flux (High)", lw=2, color="C"+str(4*i+2))
        ax1.plot(k_values_low[sort_idx_pred_low], flux_pred_low[sort_idx_pred_low], label="PySR Prediction (Low)", lw=2, linestyle="--", color="C"+str(4*i+1))
        ax1.plot(k_values_hi[sort_idx_pred_hi], flux_pred_hi[sort_idx_pred_hi], label="PySR Prediction (High)", lw=2, linestyle="--", color="C"+str(4*i+2))

        ax1.set_ylabel("P1D(k)")
        ax1.set_title(f"z = {z:.2f}, param ≈ {param_fixed_low:.2f}")
        ax1.legend()
        ax1.grid(True)

        # --- Bottom plot: Ratio or inverse ratio ---
        ax2.plot(k_values_low[sort_idx_pred_low], inverse_ratio_low, label="1-(Pred/True) Low", lw=2, color="C"+str(4*i+5))
        ax2.plot(k_values_hi[sort_idx_pred_hi], inverse_ratio_hi, label="1-(Pred/True) High", lw=2, color="C"+str(4*i+6))

        ax2.axhline(0.0, color='gray', linestyle=':', lw=1)  # baseline at 0
        ax2.set_xlabel("k [s/km]")
        ax2.set_ylabel("1 - Ratio")
        ax2.grid(True)
        ax2.legend(fontsize="small", loc='upper right')

    param_fixed_low = denormalize_X_param(np.array([param_fixed_low]), X_max, X_min)[0]
    """
    plt.xlabel("k [s/km]")
    plt.ylabel("P1D(k)")
    plt.title(f"z = {z:.2f}, param ≈ {param_fixed_low:.2f}")
    plt.legend()
    plt.grid(True)
    """
    plt.tight_layout()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"pysr_graphs_{z}_{param}.pdf")
    plt.show()


    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), sharex=True)#gridspec_kw={'height_ratios': [3, 1]}
    ax1.fill_between(k_values_low[sort_idx_pred_low],ratios_all_low[0], ratios_all_low[2], alpha=0.2, label="Ratio (Pred/True) Low", color="C"+str(4*i+5))
    ax2.fill_between(k_values_hi[sort_idx_pred_hi],ratios_all_hi[0], ratios_all_hi[2], alpha=0.2, label="Ratio (Pred/True) High", color="C"+str(4*i+6))
    ax1.plot(k_values_low[sort_idx_pred_low], ratios_all_low[0], color="r",linestyle="-.", lw=1)#label="Ratio (Pred/True) Low 16th percentile 
    ax2.plot(k_values_hi[sort_idx_pred_hi], ratios_all_hi[0], color="m", linestyle="-.",lw=1)#label="Ratio (Pred/True) High 16th percentile"
    ax1.plot(k_values_low[sort_idx_pred_low], ratios_all_low[1],color="r" ,lw=1)#label="Ratio (Pred/True) Low 50th percentile" 
    ax2.plot(k_values_hi[sort_idx_pred_hi], ratios_all_hi[1], color="m", lw=1)#label="Ratio (Pred/True) High 50th percentile"
    ax1.plot(k_values_low[sort_idx_pred_low], ratios_all_low[2], color="r",linestyle="--", lw=1)#label="Ratio (Pred/True) Low 84th percentile"
    ax2.plot(k_values_hi[sort_idx_pred_hi], ratios_all_hi[2], color="m",linestyle="--", lw=1)#label="Ratio (Pred/True) High 84th percentile"
    ax1.legend()
    ax2.legend()
    plt.show()
    """


#need: params_lo, params_hi, quantiles (array), param_idx, X_hi (should be including the resolutiuon val), X_low, y_lo,y_hi, model, z, 