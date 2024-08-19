import numpy as np
from rpy2.robjects import r
import arviz as az
import pandas as pd
from cmdstanpy.model import CmdStanModel
#cmdstanpy.install_cmdstan()
import os

os.environ['STAN_NUM_THREADS'] = '16'

methods = 'regHS'

###### Collecting the data
r.source("revision2_dataprep1.R")

y1_data = np.array(r['y1_data'])
y2_data = np.array(r['y2_data'])
x2_evt_data = np.array(r['x2_evt_data'])

###### Defining the shape of the data
n_dim,t_dim,j_dim= y1_data.shape

#### Splitting missing and observed data 
## Withing level data
# Missing data
y_miss = np.argwhere(np.isnan(y1_data))
y_miss1 = y_miss[:,0]
y_miss2 = y_miss[:,1]
y_miss3 = y_miss[:,2]

y1_data[np.isnan(y1_data)] = -99

## Between level data
x_miss = np.argwhere(np.isnan(y2_data.T))
x_miss1 = x_miss[0,0]
x_miss2 = x_miss[1,0]
x_miss3 = x_miss[2,0]
y2_data[np.isnan(y2_data)] = -99

#### Estimation with Stan

# Prepare data dictionary for stan
NDLCSEM_data = {
  'N':n_dim, 'T':t_dim, 
  'y':y1_data, 
  'N_ymiss':len(y_miss),
  'y_miss1':y_miss1+1, 
  'y_miss2':y_miss2+1,
  'y_miss3':y_miss3+1, 
  'x_miss1':x_miss1+1, 
  'x_miss2':x_miss2+1,
  'x_miss3':x_miss3+1,
  'X':y2_data.T,
  'sigma_s':(np.pi/np.sqrt(3))/np.sqrt(n_dim*t_dim),
  'X_evt':x2_evt_data.T
} 

# Launch stan
cmd_stanmodel = CmdStanModel(
  stan_file='ildmsvar_'+methods+'.stan',
  cpp_options={'STAN_THREADS': 'TRUE'})
vi_init = cmd_stanmodel.variational(
    data=NDLCSEM_data, 
    require_converged=False,
    draws=200,grad_samples = 50,
    seed=123,inits=0.0)

# create a dictionary with init values 
n_chains,n_samples,n_warmups=[4,2000,2000]
init_dict = {var: samples.mean(axis=0) for var, samples in vi_init.stan_variables(mean=False).items()}
init_chains= [init_dict for _ in range(n_chains)]

# NUTS-HMC sampling
post_sample = cmd_stanmodel.sample(
  data=NDLCSEM_data,chains=n_chains,iter_warmup=n_warmups,iter_sampling=n_samples,
  inits=init_chains,seed=60424,show_progress=True) 

# convert pystan to arviz
fit_az = az.from_cmdstanpy(
      posterior=post_sample,observed_data={'y':NDLCSEM_data['y'],'X':NDLCSEM_data['X']},
      coords={
        "chain": np.arange(n_chains),"draw": np.arange(n_samples)},
      dims={
      'loady': ["chain","draw","var1"],
      'sigma_y': ["chain","draw","var2"],
      'alpha_': ["chain","draw","var4"],
      'B_': ["chain","draw","var5","var6"],
      'tau_e': ["chain","draw","var7"],
      'L_e': ["chain","draw","var8","var9"],
      'loadX': ["chain","draw","var10"],
      'sigma_X': ["chain","draw"],
      'sigma_u': ["chain","draw"],
      'beta_0': ["chain","draw"],
      'beta_x': ["chain","draw"],
      'beta_y': ["chain","draw","var12"],
      'beta_xy': ["chain","draw","var13"]})

fit_az.to_netcdf(
  methods+
  str(n_dim)+str(t_dim)+str(j_dim)+'_'+
  str(n_chains)+'_'+str(n_samples+n_warmups)+"vi.nc")

#print(post_sample.diagnose())


