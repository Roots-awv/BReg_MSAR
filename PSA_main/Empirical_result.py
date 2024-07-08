import numpy as np
import pandas as pd
import arviz as az
import forestplot as fp
from matplotlib import pyplot as plt
from scipy import stats

def post_mode(samples):
    kde = stats.gaussian_kde(samples)
    x = np.linspace(np.min(samples), np.max(samples), 1000)
    return x[np.argmax(kde(x))]

############################################################
# Loading the results
model = [
  "ridge0",
  "regHS1",
  "regHS1tau4",
  "regHSstudent",
  "blasso1",
  "absslass199",
  "regHS",
  "ridge4",
  "regHSstudent2",
  "absslassu99",
  "absslass499"
]

###################### "PSA_SAMdata_plus/"+
file_ridge0 = model[0]+"122509_4_4000vi.nc"
fit_az_ridge0 = az.from_netcdf(file_ridge0)
file_regHS1 = model[1]+"122509_4_4000vi.nc"
fit_az_regHS1 = az.from_netcdf(file_regHS1)
file_regHS2 = model[2]+"122509_4_4000vi.nc"
fit_az_regHS2 = az.from_netcdf(file_regHS2)
file_regHS3 = model[3]+"122509_4_4000vi.nc"
fit_az_regHS3 = az.from_netcdf(file_regHS3)
file_blasso = model[4]+"122509_4_4000vi.nc"
fit_az_blasso = az.from_netcdf(file_blasso)
file_absslass = model[5]+"122509_4_4000vi.nc"
fit_az_absslass = az.from_netcdf(file_absslass)
file_regHS = model[6]+"122509_4_4000vi.nc"
fit_az_regHS = az.from_netcdf(file_regHS)
file_ridge4 = model[7]+"122509_4_4000vi.nc"
fit_az_ridge4 = az.from_netcdf(file_ridge4)
file_regHS4 = model[8]+"122509_4_4000vi.nc"
fit_az_regHS4 = az.from_netcdf(file_regHS4)
file_absslassou = model[9]+"122509_4_4000vi.nc"
fit_az_absslassou = az.from_netcdf(file_absslassou)
file_absslasso4 = model[10]+"122509_4_4000vi.nc"
fit_az_absslasso4 = az.from_netcdf(file_absslasso4)

#############################################################
### Designs of figures
## ridge vs regHS light
plt.style.use('ggplot')

# Create the figure and axis
fig, ax = plt.subplots(figsize=(12, 10))  # Modify the width and height as needed

# Create the grid of subplots
axes = az.plot_forest(
    [fit_az_ridge0, fit_az_regHS1, fit_az_regHS3],
    model_names=['Ridge-0','reg. HS-1','reg. HS-2'],
    kind="forestplot",
    var_names=["beta_"],
    combined=True,
    hdi_prob=0.95,
    colors=["tab:pink","royalblue","blueviolet"],
    ax=ax
)
axes[0].set_yticklabels(["$\\beta_{" + str(p + 1) + "}$" for p in range(27)[::-1]], fontsize='large')
axes[0].set_xlabel("Estimates", fontsize='x-large')
axes[0].set_ylabel("Coefficients", fontsize='x-large')
axes[0].set_title("")

# Adjust layout to prevent overlap
plt.axvline(0, color="grey", ls="--")
plt.tight_layout() # rect=[1,1,1,1]
plt.savefig('Graphics_PSA-i.jpg')

###############################################
## aBSS-Lasso

# Create the figure and axis
fig, ax = plt.subplots(figsize=(12, 10))  # Modify the width and height as needed

# Create the forest plot
axes = az.plot_forest(
    [fit_az_absslass, fit_az_absslassou,file_absslasso4],
    model_names=['aBSS-Lasso', 'aBSS-Lasso-1','aBSS-Lasso-2'],
    kind="forestplot",
    var_names=["beta_"],
    combined=True,
    hdi_prob=0.95,
    colors=["tab:green","darkgoldenrod","darkgreen"],
    ax=ax
)

ax.set_yticklabels(["$\\beta_{" + str(p + 1) + "}$" for p in range(27)[::-1]], fontsize='large')
ax.set_xlabel("Estimates", fontsize='x-large')
ax.set_ylabel("Coefficients", fontsize='x-large')
ax.set_title("")

# Adjust layout to prevent overlap
plt.axvline(0, color="grey", ls="--")
plt.tight_layout()  # Adjust rect parameter if needed
plt.savefig('Graphics_PSA-ii.jpg')

###############################################
## ridge vs reg. HS vs B-Lasso

# Create the figure and axis
fig, ax = plt.subplots(figsize=(12, 10))  # Modify the width and height as needed

# Create the forest plot
axes = az.plot_forest(
    [fit_az_ridge4, fit_az_regHS,file_blasso],
    model_names=['Ridge-1', 'reg. HS-0','B-Lasso'],
    kind="forestplot",
    var_names=["beta_"],
    combined=True,
    hdi_prob=0.95,
    colors=["tab:red","steelblue","orange"],
    ax=ax
)

ax.set_yticklabels(["$\\beta_{" + str(p + 1) + "}$" for p in range(27)[::-1]], fontsize='large')
ax.set_xlabel("Estimates", fontsize='x-large')
ax.set_ylabel("Coefficients", fontsize='x-large')
ax.set_title("")

# Adjust layout to prevent overlap
plt.axvline(0, color="grey", ls="--")
plt.tight_layout()  # Adjust rect parameter if needed
plt.savefig('Graphics_PSA-iii.jpg')

###############################################
## Selected

# Create the figure and axis
fig, ax = plt.subplots(figsize=(12, 10))  # Modify the width and height as needed

# Create the forest plot
axes = az.plot_forest(
    [fit_az_ridge0, fit_az_regHS1, fit_az_regHS2, fit_az_regHS3],
    model_names=['Ridge-0','reg. HS-1','reg. HS-2','reg. HS-3'],
    kind="forestplot",
    var_names=["beta_x","beta_evt","beta_y","beta_xy"],
    combined=True,
    hdi_prob=0.95,
    colors=["tab:pink","teal","royalblue","blueviolet"],
    ax=ax
)

beta_x = ["$\\beta_{x" + str(p + 1) + "}$" for p in range(6)[::-1]]
beta_y = ["$\\beta_{y" + str(p + 1) + "}$" for p in range(3)[::-1]]
beta_xy = ["$\\beta_{x1y" + str(p + 1) + "}$" for p in range(3)[::-1]]
beta_select = beta_xy + beta_y + beta_x

ax.set_yticklabels(beta_select, fontsize='large')
ax.set_xlabel("Estimates", fontsize='x-large')
ax.set_ylabel("Coefficients", fontsize='x-large')
ax.set_title("")

# Adjust layout to prevent overlap
plt.axvline(0, color="grey", ls="--")
plt.tight_layout()  # Adjust rect parameter if needed
plt.savefig('Graphics_PSA-draft.jpg')

###############################################
## Selected

# Create the figure and axis
fig, axes = plt.subplots(2, 1, figsize=(12, 10))  # Modify the width and height as needed

# y-axis labels
beta_x = ["$\\beta_{x" + str(p + 1) + "}$" for p in range(6)[::-1]]
beta_y = ["$\\beta_{y" + str(p + 1) + "}$" for p in range(3)[::-1]]
beta_xy = ["$\\beta_{xy" + str(p + 1) + "}$" for p in range(3)[::-1]]
beta_small = beta_xy + beta_x

# Define colors and model names
colors = ["tab:pink", "teal", "dodgerblue", "royalblue", "blueviolet"]
model_names = ['Ridge-0', 'reg. HS-1', 'reg. HS-2', 'reg. HS-3', 'reg. HS-4']

# Large effects forest plot
az.plot_forest(
    [fit_az_ridge0, fit_az_regHS1, fit_az_regHS2, fit_az_regHS3, fit_az_regHS4],
    model_names=model_names,
    kind="forestplot",
    var_names=["beta_y"],
    combined=True,
    hdi_prob=0.95,
    colors=colors,
    ax=axes[0]
)

axes[0].set_yticklabels(beta_y, fontsize='large')
axes[0].set_xlabel("Estimates", fontsize='x-large')
axes[0].set_ylabel("Large effects", fontsize='x-large')
axes[0].set_title("")
axes[0].set_xlim([-15, 5])
axes[0].axvline(0, color="grey", linestyle="--")

# Small effects forest plot
az.plot_forest(
    [fit_az_ridge0, fit_az_regHS1, fit_az_regHS2, fit_az_regHS3, fit_az_regHS4],
    model_names=model_names,
    kind="forestplot",
    var_names=["beta_x", "beta_evt", "beta_xy"],
    combined=True,
    hdi_prob=0.95,
    colors=colors,
    ax=axes[1]
)

axes[1].set_yticklabels(beta_small, fontsize='large')
axes[1].set_xlabel("Estimates", fontsize='x-large')
axes[1].set_ylabel("Small effects", fontsize='x-large')
axes[1].set_title("")
axes[1].set_xlim([-15, 5])
axes[1].axvline(0, color="grey", linestyle="--")

# Adjust layout to prevent overlap
plt.tight_layout(rect=[1, 1, 1, 1])  # Adjust rect parameter to leave space for the legend
plt.savefig('Graphics_PSA-iv.jpg')

########################################################
### Grayscale Figures
########################################################
## Prepare DataFrame
all_methods = ['Ridge-0', 'reg. HS-1', 'reg. HS-2', 'reg. HS-3', 
           'B-Lasso', 'aBSS-Lasso-1', 'reg. HS-0' ,'Ridge-1',
           'reg. HS-3', 'aBSS-Lasso-0', 'aBSS-Lasso-2']

var_names = 'beta_' 
loc= [0,1,6,7,8,9,26]
coefficients = range(1, len(loc) + 1)
methods = ['Ridge-0', 'reg. HS-1', 'reg. HS-2']
model = ["ridge0",  "regHS1",  "regHSstudent2"]

postmodes = {}
postmeans = {}
lower_dist = {}
upper_dist = {}
for m_ in range(len(model)):
  file_ = model[m_]+"122509_4_4000vi.nc"
  fit_az_ = az.from_netcdf(file_)
  results_fit_az = az.summary(fit_az_, var_names=var_names,stat_funcs={'mode':post_mode},hdi_prob=0.95)
  results_fit_df = pd.DataFrame(results_fit_az)
  # Compute the absolute differences for your actual data
  results_fit_df['abs_diff_mean_hdi_2.5%'] = (results_fit_df['mean'] - results_fit_df['hdi_2.5%']).abs()
  results_fit_df['abs_diff_mean_hdi_97.5%'] = (results_fit_df['hdi_97.5%'] - results_fit_df['mean']).abs()
  results_fit_loc = results_fit_df[['mode','mean','abs_diff_mean_hdi_2.5%','abs_diff_mean_hdi_97.5%']].iloc[loc]
  postmodes[methods[m_]] = results_fit_loc['mode'].to_numpy()
  postmeans[methods[m_]] = results_fit_loc['mean'].to_numpy()
  lower_dist[methods[m_]] = results_fit_loc['abs_diff_mean_hdi_2.5%'].to_numpy()
  upper_dist[methods[m_]] = results_fit_loc['abs_diff_mean_hdi_97.5%'].to_numpy()

#################
# Plotting posterior mean estimates w/ 95% CI
plt.figure(figsize=(12, 10))

# Define markers and grayscale colors
markers = ['s', 'x', '+']  # Different markers for each method
colors = ['black', 'dimgray', 'dimgray']  # Different shades of gray

# Define larger spacing between boxplots
spacing = 0.15  # Increase spacing between boxplots

# Plotting boxplots for each method across parameters
for idx, method in enumerate(methods):
    mean = postmeans[method]
    lower = lower_dist[method]
    upper = upper_dist[method]
    
    # Adjust y-position for each method
    x_pos = np.array(coefficients) + idx * spacing
        
    # Plot box and whisker plot with error bars
    plt.errorbar(x_pos, mean, yerr=[lower,upper], 
                 fmt=markers[idx], color=colors[idx], label=method, 
                 markersize = 12, markeredgecolor= 'black', capsize=5)

# Customize labels and title
plt.xticks(np.array(coefficients) + (len(methods) - 1) * spacing / 2, [f"$\\beta_{{{i+1}}}$" for i in loc], fontsize='large')
plt.ylabel('Estimates', fontsize='x-large')
plt.xlabel('Coefficients', fontsize='x-large')
plt.title('Comparison of Regularization Methods Estimates', fontsize='xx-large')
plt.axhline(0, color='darkgray', linestyle='--')
plt.ylim([-15, 5])
plt.legend(loc='upper left', fontsize='xx-large')
plt.grid(True)

# Show plot
plt.tight_layout()
#plt.show()
plt.savefig('Graphics_PSA-vii.jpg')

#################
# Plotting posterior mean estimates w/ 95% CI
plt.figure(figsize=(12, 10))

# Define markers and grayscale colors
markers = ['s', 'x', '+']  # Different markers for each method
colors = ['black', 'darkgray', 'darkgray']  # Different shades of gray

# Define larger spacing between boxplots
spacing = 0.1  # Increase spacing between boxplots

# Plotting boxplots for each method across parameters
for idx, method in enumerate(methods):
    mean = postmeans[method]
    lower = lower_dist[method]
    upper = upper_dist[method]
    
    # Adjust y-position for each method
    y_pos = np.array(coefficients) + idx * spacing
        
    # Plot box and whisker plot with error bars
    plt.errorbar(mean, y_pos, xerr=[lower,upper], 
                 fmt=markers[idx], color=colors[idx], label=method, 
                 markersize = 10, markeredgecolor= 'black', capsize=5)

# Customize labels and title
plt.yticks(np.array(coefficients) + (len(methods) - 1) * spacing / 2, [f"$\\beta_{{{i+1}}}$" for i in loc], fontsize='large')
plt.xlabel('Estimates', fontsize='x-large')
plt.ylabel('Coefficients', fontsize='x-large')
plt.title('Comparison of Regularization Methods Estimates', fontsize='xx-large')
plt.axvline(0, color='lightgray', linestyle='--')
plt.legend(loc='upper left', fontsize='xx-large')
plt.grid(True)

# Show plot
plt.tight_layout()
#plt.show()
plt.savefig('Graphics_PSA-viii.jpg')

np.exp(2.034)
np.exp(2.468)

#################################
# Plotting posterior mean vs. posterior modes
plt.figure(figsize=(12, 10))

# Define markers and grayscale colors
markers = ['s',  'o','D']  # Different markers for each method
colors = ['k',  'k','k']  # Different shades of gray

# Define larger spacing between boxplots
spacing = 0.2  # Increase spacing between boxplots

# Plotting means and modes for each method across parameters
for idx, method in enumerate(methods):
    mean = postmeans[method]
    mode = postmodes[method]
    
    # Adjust y-position for each method
    x_pos = np.array(coefficients) + idx * spacing

    # Plot means
    plt.plot(x_pos,mean, marker=markers[idx], color=colors[idx], linestyle='None', 
             markersize=15, markeredgecolor='black', label=f'{method} Mean')

    # Plot modes
    plt.plot(x_pos,mode, marker=markers[idx], color=colors[idx], linestyle='None',
             markersize=15, markeredgecolor='black', markerfacecolor='white', label=f'{method} Mode')

# Customize labels and title
plt.xticks(np.array(coefficients) + (len(methods) - 1) * spacing / 2, [f"$\\beta_{{{i+1}}}$" for i in loc], fontsize='large')
plt.ylabel('Estimates', fontsize='x-large')
plt.xlabel('Coefficients', fontsize='x-large')
plt.title('Comparison of Posterior Means and Modes', fontsize='xx-large')
plt.legend(loc='upper left', fontsize='large')
plt.grid(True)
plt.ylim([-5.25, 2.25])
# Show plot
plt.tight_layout()
plt.savefig('Graphics_PSA-ix.jpg')
#plt.show()

#################
# Plotting posterior mean estimates w/ 95% CI
plt.figure(figsize=(12, 10))

# Define markers and grayscale colors
markers = ['s', 'x', '+']  # Different markers for each method
colors = ['black', 'darkgray', 'darkgray']  # Different shades of gray

# Define larger spacing between boxplots
spacing = 0.1  # Increase spacing between boxplots

# Plotting boxplots for each method across parameters
for idx, method in enumerate(methods):
    mean = postmeans[method]
    lower = lower_dist[method]
    upper = upper_dist[method]
    
    # Adjust y-position for each method
    y_pos = np.array(coefficients) + idx * spacing
        
    # Plot box and whisker plot with error bars
    plt.errorbar(mean, y_pos, xerr=[lower,upper], 
                 fmt=markers[idx], color=colors[idx], label=method, 
                 markersize = 10, markeredgecolor= 'black', capsize=5)

# Customize labels and title
plt.yticks(np.array(coefficients) + (len(methods) - 1) * spacing / 2, [f"$\\beta_{{{i+1}}}$" for i in loc], fontsize='large')
plt.xlabel('Estimates', fontsize='x-large')
plt.ylabel('Coefficients', fontsize='x-large')
plt.title('Comparison of Regularization Methods Estimates', fontsize='xx-large')
plt.axvline(0, color='lightgray', linestyle='--')
plt.legend(loc='upper left', fontsize='xx-large')
plt.grid(True)

# Show plot
plt.tight_layout()
#plt.show()
plt.savefig('Graphics_PSA-viii.jpg')

##########################
# Plotting posterior mean vs. posterior modes
plt.figure(figsize=(12, 10))

# Define markers and grayscale colors
markers = ['s',  'o','D']  # Different markers for each method
colors = ['tab:pink',  'darkblue','darkviolet']  # Different shades of gray

# Define larger spacing between boxplots
spacing = 0.2  # Increase spacing between boxplots

# Plotting means and modes for each method across parameters
for idx, method in enumerate(methods):
    mean = postmeans[method]
    mode = postmodes[method]
    
    # Adjust y-position for each method
    x_pos = np.array(coefficients) + idx * spacing

    # Plot means
    plt.plot(x_pos,mean, marker=markers[idx], color=colors[idx], linestyle='None', 
             markersize=15, markeredgecolor='black', label=f'{method} Mean')

    # Plot modes
    plt.plot(x_pos,mode, marker=markers[idx], color=colors[idx], linestyle='None',
             markersize=15, markeredgecolor='black', markerfacecolor='white', label=f'{method} Mode')

# Customize labels and title
plt.xticks(np.array(coefficients) + (len(methods) - 1) * spacing / 2, [f"$\\beta_{{{i+1}}}$" for i in loc], fontsize='large')
plt.ylabel('Estimates', fontsize='x-large')
plt.xlabel('Coefficients', fontsize='x-large')
plt.title('Comparison of Posterior Means and Modes', fontsize='xx-large')
plt.legend(loc='upper left', fontsize='large')
plt.grid(True)
plt.ylim([-5.25, 2.25])
# Show plot
plt.tight_layout()
plt.savefig('Graphics_PSA-ix.jpg')
#plt.show()

#################################
### Colored scale
#################################
## Prepare DataFrame
all_methods = ['Ridge-0', 'reg. HS-1', 'reg. HS-2', 'reg. HS-3', 
           'B-Lasso', 'aBSS-Lasso-1', 'reg. HS-0' ,'Ridge-1',
           'reg. HS-3', 'aBSS-Lasso-0', 'aBSS-Lasso-2']

var_names = 'beta_' 
loc= [0,1,6,7,8,9,26]
coefficients = range(1, len(loc) + 1)
methods = ['reg HS-0', 'B-Lasso', 'aBSS-Lasso-0']
model = ["regHS",  "blasso1",  "absslassu99"]

postmodes = {}
postmeans = {}
lower_dist = {}
upper_dist = {}
for m_ in range(len(model)):
  file_ = model[m_]+"122509_4_4000vi.nc"
  fit_az_ = az.from_netcdf(file_)
  results_fit_az = az.summary(fit_az_, var_names=var_names,stat_funcs={'mode':post_mode},hdi_prob=0.95)
  results_fit_df = pd.DataFrame(results_fit_az)
  # Compute the absolute differences for your actual data
  results_fit_df['abs_diff_mean_hdi_2.5%'] = (results_fit_df['mean'] - results_fit_df['hdi_2.5%']).abs()
  results_fit_df['abs_diff_mean_hdi_97.5%'] = (results_fit_df['hdi_97.5%'] - results_fit_df['mean']).abs()
  results_fit_loc = results_fit_df[['mode','mean','abs_diff_mean_hdi_2.5%','abs_diff_mean_hdi_97.5%']].iloc[loc]
  postmodes[methods[m_]] = results_fit_loc['mode'].to_numpy()
  postmeans[methods[m_]] = results_fit_loc['mean'].to_numpy()
  lower_dist[methods[m_]] = results_fit_loc['abs_diff_mean_hdi_2.5%'].to_numpy()
  upper_dist[methods[m_]] = results_fit_loc['abs_diff_mean_hdi_97.5%'].to_numpy()

# Plotting posterior mean vs. posterior modes
plt.figure(figsize=(12, 10))

# Define markers and grayscale colors
markers = ['s', 'o','D']  # Different markers for each method
colors = ['darkblue', 'orange', 'darkgreen']  # Different shades of gray

# Define larger spacing between boxplots
spacing = 0.2  # Increase spacing between boxplots

# Plotting means and modes for each method across parameters
for idx, method in enumerate(methods):
    mean = postmeans[method]
    mode = postmodes[method]
    
    # Adjust y-position for each method
    x_pos = np.array(coefficients) + idx * spacing

    # Plot means
    plt.plot(x_pos,mean, marker=markers[idx], color=colors[idx], linestyle='None', 
             markersize=15, markeredgecolor='black', label=f'{method} Mean')

    # Plot modes
    plt.plot(x_pos,mode, marker=markers[idx], color=colors[idx], linestyle='None',
             markersize=15, markeredgecolor='black', markerfacecolor='white', label=f'{method} Mode')

# Customize labels and title
plt.xticks(np.array(coefficients) + (len(methods) - 1) * spacing / 2, [f"$\\beta_{{{i+1}}}$" for i in loc], fontsize='large')
plt.ylabel('Estimates', fontsize='x-large')
plt.xlabel('Coefficients', fontsize='x-large')
plt.title('Comparison of Posterior Means and Modes', fontsize='xx-large')
plt.legend(loc='lower left', fontsize='large')
plt.grid(True)
plt.ylim([-50, 2.5])
# Show plot
plt.tight_layout()
plt.savefig('Graphics_PSA-xi.jpg')

#######################################
#### Mode as point estimate
## Prepare DataFrame
all_methods = ['Ridge-0', 'reg. HS-1', 'reg. HS-2', 'reg. HS-3', 
           'B-Lasso', 'aBSS-Lasso-1', 'reg. HS-0' ,'Ridge-1',
           'reg. HS-3', 'aBSS-Lasso-0', 'aBSS-Lasso-2']

var_names = 'beta_' 
loc= [0,1,6,7,8,9,26]
coefficients = range(1, len(loc) + 1)
methods = ['Ridge-0', 'reg. HS-0', 'reg. HS-2']
model = ["ridge0",  "regHS",  "regHSstudent2"]

postmodes = {}
lower_dist = {}
upper_dist = {}
for m_ in range(len(model)):
  file_ = model[m_]+"122509_4_4000vi.nc"
  fit_az_ = az.from_netcdf(file_)
  results_fit_az = az.summary(fit_az_, var_names=var_names,stat_funcs={'mode':post_mode},hdi_prob=0.95)
  results_fit_df = pd.DataFrame(results_fit_az)
  # Compute the absolute differences for your actual data
  results_fit_df['abs_diff_mode_hdi_2.5%'] = (results_fit_df['mode'] - results_fit_df['hdi_2.5%']).abs()
  results_fit_df['abs_diff_mode_hdi_97.5%'] = (results_fit_df['hdi_97.5%'] - results_fit_df['mode']).abs()
  results_fit_loc = results_fit_df[['mode','abs_diff_mode_hdi_2.5%','abs_diff_mode_hdi_97.5%']].iloc[loc]
  postmodes[methods[m_]] = results_fit_loc['mode'].to_numpy()
  lower_dist[methods[m_]] = results_fit_loc['abs_diff_mode_hdi_2.5%'].to_numpy()
  upper_dist[methods[m_]] = results_fit_loc['abs_diff_mode_hdi_97.5%'].to_numpy()

#################
# Plotting posterior mean estimates w/ 95% CI
plt.figure(figsize=(12, 10))

# Define markers and grayscale colors
markers = ['1','+', 'x']  # Different markers for each method
colors = ['tab:pink', 'steelblue', 'darkblue']  # Different shades of gray

# Define larger spacing between boxplots
spacing = 0.1  # Increase spacing between boxplots

# Plotting boxplots for each method across parameters
for idx, method in enumerate(methods):
    mode = postmodes[method]
    lower = lower_dist[method]
    upper = upper_dist[method]
    
    # Adjust y-position for each method
    y_pos = np.array(coefficients) + idx * spacing
        
    # Plot box and whisker plot with error bars
    plt.errorbar(mode, y_pos, xerr=[lower,upper], 
                 fmt=markers[idx], color=colors[idx], label=method, 
                 markersize = 10, markeredgecolor= 'black', capsize=5)

# Customize labels and title
plt.yticks(np.array(coefficients) + (len(methods) - 1) * spacing / 2, [f"$\\beta_{{{i+1}}}$" for i in loc], fontsize='large')
plt.xlabel('Estimates', fontsize='x-large')
plt.ylabel('Coefficients', fontsize='x-large')
plt.title('Comparison of Regularization Methods Estimates', fontsize='xx-large')
plt.axvline(0, color='lightgray', linestyle='--')
plt.legend(loc='upper left', fontsize='xx-large')
plt.grid(True)

# Show plot
plt.tight_layout()
#plt.show()
plt.savefig('Graphics_PSA-xii.jpg')

#################
### Prepare DataFrame
#all_methods = ['Ridge-0', 'reg. HS-1', 'reg. HS-2', 'reg. HS-3', 
#           'B-Lasso', 'aBSS-Lasso-1', 'reg. HS-0' ,'Ridge-1',
#           'reg. HS-3', 'aBSS-Lasso-0', 'aBSS-Lasso-2']
#
#var_names = 'beta_' 
#coefficients = range(1, 28)
#methods = ['Ridge-0', 'reg. HS-2']
#model = ["ridge0",  "regHSstudent2"]
#
#postmodes = {}
#postmeans = {}
#lower_dist = {}
#upper_dist = {}
#for m_ in range(len(model)):
#  file_ = model[m_]+"122509_4_4000vi.nc"
#  fit_az_ = az.from_netcdf(file_)
#  results_fit_az = az.summary(fit_az_, var_names=var_names,stat_funcs={'mode':post_mode},hdi_prob=0.95)
#  results_fit_df = pd.DataFrame(results_fit_az)
#  # Compute the absolute differences for your actual data
#  results_fit_df['abs_diff_mean_hdi_2.5%'] = (results_fit_df['mean'] - results_fit_df['hdi_2.5%']).abs()
#  results_fit_df['abs_diff_mean_hdi_97.5%'] = (results_fit_df['hdi_97.5%'] - results_fit_df['mean']).abs()
#  results_fit_loc = results_fit_df[['mode','mean','abs_diff_mean_hdi_2.5%','abs_diff_mean_hdi_97.5%']]
#  postmodes[methods[m_]] = results_fit_loc['mode'].to_numpy()
#  postmeans[methods[m_]] = results_fit_loc['mean'].to_numpy()
#  lower_dist[methods[m_]] = results_fit_loc['abs_diff_mean_hdi_2.5%'].to_numpy()
#  upper_dist[methods[m_]] = results_fit_loc['abs_diff_mean_hdi_97.5%'].to_numpy()
#
## Plotting posterior mean vs. posterior modes
#plt.figure(figsize=(12, 10))
#
## Define markers and grayscale colors
#markers = ['s',  'o']  # Different markers for each method
#colors = ['k',  'k']  # Different shades of gray
#
## Define larger spacing between boxplots
#spacing = 0.25  # Increase spacing between boxplots
#
## Plotting means and modes for each method across parameters
#for idx, method in enumerate(methods):
#    mean = postmeans[method]
#    mode = postmodes[method]
#    
#    # Adjust y-position for each method
#    y_pos = np.array(coefficients) + idx * spacing
#
#    # Plot means
#    plt.plot(mean, y_pos, marker=markers[idx], color=colors[idx], linestyle='None', markersize=10, markeredgecolor='black', label=f'{method} Mean')
#
#    # Plot modes
#    plt.plot(mode, y_pos, marker=markers[idx], color=colors[idx], linestyle='None', markersize=10, markeredgecolor='black', markerfacecolor='white', label=f'{method} Mode')
#
## Customize labels and title
#plt.yticks(np.array(coefficients) + (len(methods) - 1) * spacing / 2, [f"$\\beta_{{{i+1}}}$" for i in loc], fontsize='large')
#plt.xlabel('Estimates', fontsize='x-large')
#plt.ylabel('Coefficients', fontsize='x-large')
#plt.title('Comparison of Posterior Means and Modes', fontsize='xx-large')
#plt.axvline(0, color='lightgray', linestyle='--')
#plt.legend(loc='upper left', fontsize='large')
#plt.grid(True)
#
## Show plot
#plt.tight_layout()
##plt.savefig('Graphics_PSA-vi.jpg')
#plt.show()

#################################### 
'''
Saving resultts in csv files 
'''
####################################
var_names = ['loady','sigma_y',
            'alpha_','B_',
            'tau_e','L_e',
            'loadX','sigma_X',
            'sigma_u',
            'beta_0','beta_'] 

func_dict = {
  "2.5%":lambda x: np.percentile(x,2.5),
  "97.5%":lambda x: np.percentile(x,97.5),
}

for m_ in model:
  file_ = m_+"122509_4_4000vi.nc"
  fit_az_ = az.from_netcdf(file_)
  results_fit_az = az.summary(fit_az_,
      var_names=var_names,
      stat_funcs=func_dict,
      hdi_prob=0.95)
  results_fit_df = pd.DataFrame(results_fit_az)
  results_fit_df.to_csv(m_+"122509_4_4000vi.csv")

################ 
# Computing the posterior mode using KDE (Kernel density estimation)
model_names = [
  "Ridge-0",
  "reg. HS-1",
  "regHS1tau4",
  "regHSstudent",
  "B-Lasso",
  "aBSS-Lasso-1",
  "reg. HS-0",
  "Ridge-1",
  "reg. HS-2",
  "aBSS-Lasso-0",
  "aBSS-Lasso-2"
]

for m_ in range(len(model)):
  file_ = model[m_]+"122509_4_4000vi.nc"
  fit_az_ = az.from_netcdf(file_)
  results_fit_az = az.summary(fit_az_,
      stat_focus='mean',
      var_names=['beta_0','beta_x','beta_evt','beta_y',
                 'beta_xy','beta_evty1','beta_evty2','beta_evty3'],
      stat_funcs={'mode': post_mode},
      hdi_prob=0.95)
  results_fit_df = pd.DataFrame(results_fit_az)
  results_fit_df.to_csv("stat_beta_"+file_+".csv")
  post_MvM=results_fit_df[['mean','mode']]
  # Create a MultiIndex for columns
  new_columns = pd.MultiIndex.from_product([['Post. '+model_names[m_]], post_MvM.columns])
  post_MvM.columns = new_columns
  post_MvM.to_csv("postmode_beta_"+model_names[m_]+".csv")
  
###### Posterior plot
#az.plot_posterior(fit_az_ridge0,var_names=['beta_'],point_estimate='mode',hdi_prob=0.95,multimodal=True)
#plt.show()
