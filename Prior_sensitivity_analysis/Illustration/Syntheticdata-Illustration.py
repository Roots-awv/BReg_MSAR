import numpy as np
import numpy.random as rnd
import pandas as pd
import quantecon as qe
from numpy.linalg import inv

# The syntax contains two parts: the first part consists in generating the data and
# the second involves estimation with cmdstanpy.
# To generate the syntetic data we built the data_generation() function. Missing data  
# were included to mimic the situation we dealt with the empirical data.

##########################################################################################
############################## Synthetic data ############################################
##########################################################################################

rnd.seed(171122)
##### Functions
def inv_logit(x):
  return 1/(1+np.exp(-x))

def data_generation(N_dim,T_dim,K_dim):
  ########### Dimension
  #N_dim = 122
  #T_dim = 50
  #K_dim = 3 # latent factor 
  J_dim = 3 # items
  J_evt = 5
  I_K = np.eye(K_dim) 

  ##### Parameters
  ## Covariance matrix
  phi_var = 0.25
  sigmaetaY = 0.25
  sigmaY = sigmaX = 0.25

  ## Coefficients
  beta_0 = 4
  beta_x = 0.5 * np.ones(J_evt+1)
  beta_y = -0.5 * np.ones(K_dim)
  beta_yevt = -0.5 * np.ones(J_evt*K_dim)
  beta_yx = 0.5 * np.ones(K_dim)
  B = np.array([[0.7,0,0],
                 [0.2,0.5,0],
                 [0.2,0.2,0.5]])
  B = np.vstack((B,np.zeros((K_dim,K_dim))))
  mu = np.array([[2,3,3],
                 [0,0,0]])
  Omega_K = sigmaetaY*I_K # np.array([[1,0.5,0.5],[0.5,1,0.5],[0.5,0.5,1]])   

  ## Loading factor
  lambdaX = np.ones((J_dim))
  lambdaY = np.kron(I_K,np.ones((J_dim,1)))  

  ###### Between level 
  etaX = rnd.normal(0,phi_var,(N_dim))  

  ## Measurment model
  X = np.zeros((J_dim,N_dim))
  zetaX = rnd.normal(0,sigmaX,(N_dim,J_dim))
  for j in range(J_dim):
    X[j] = lambdaX[j] * etaX + zetaX[:,j]
  X_evt = rnd.normal(0,1,(N_dim,J_evt))

  ###### Within level
  Y = np.zeros((K_dim*J_dim,N_dim,T_dim),dtype= None)
  epsilonY = np.zeros((K_dim*J_dim,N_dim,T_dim),dtype= None)
  eta_y = np.zeros((K_dim,N_dim,T_dim))
  trans_Matrix = np.zeros((N_dim,T_dim,2,2)) 
  possible_states = ("0","1") 
  states_seq = np.zeros((N_dim,T_dim),dtype=str)
  hidden_state = np.zeros((N_dim,T_dim),dtype=int)
  mc_object = np.zeros((N_dim,T_dim),dtype=object)  
  eta_p = np.zeros((N_dim,T_dim)) 
  
  for i in range(N_dim):
    #### t = 0
    # states transition
    states_ini = '0'

    # Transition probability
    trans_Matrix[i,0,0,0] = inv_logit(beta_0) #+eta_s[i] 
    trans_Matrix[i,0,0,1] = 1- trans_Matrix[i,0,0,0]
    trans_Matrix[i,0,1,0] = 0.05 #np.full(N_dim,0.05) 
    trans_Matrix[i,0,1,1] = 1 - trans_Matrix[i,0,1,0] 
    mc_object[i,0] = qe.MarkovChain(P = trans_Matrix[i,0,:,:], state_values=possible_states)

    # Sequence of hidden states
    update = mc_object[i,0].simulate(ts_length= 2, init=states_ini)        
    states_seq[i,0] = update[1]
    hidden_state[i,0]=int(states_seq[i,0])

    # Structural relationships 
    S_e = inv(I_K-B[hidden_state[i,0]]@B[hidden_state[i,0]].T)@Omega_K
    eta_y[:,i,0] = rnd.multivariate_normal(mu[hidden_state[i,0]], S_e)

    # Measurement model
    epsilonY[:,i,0] = rnd.multivariate_normal(np.zeros(K_dim*J_dim),sigmaY*np.eye(K_dim*J_dim))
    Y[:,i,0] = lambdaY @ eta_y[:,i,0] + epsilonY[:,i,0]

    #### t > 0
    for t in range(1,T_dim):  

      # Transition probability
      eta_ti = beta_x[0]*etaX[i]+ X_evt[i] @ beta_x[1:]
      eta_tv_x = eta_y[:,i,t-1]@beta_y + etaX[i]*eta_y[:,i,t-1]@beta_yx
      eta_tv_evt=np.outer(X_evt[i],eta_y[:,i,t-1]) #@beta_yevt
      eta_p[i,t] = eta_ti + eta_tv_x + eta_tv_evt.flatten() @ beta_yevt
    
      trans_Matrix[i,t,0,0] = inv_logit(beta_0+eta_p[i,t]) #-phi_t[t-1]+eta_s[i] 
      trans_Matrix[i,t,0,1] = 1- trans_Matrix[i,t,0,0]
      trans_Matrix[i,t,1,0] = 0.05 #np.full(N_dim,0.05) 
      trans_Matrix[i,t,1,1] = 1 - trans_Matrix[i,t,1,0] 

      # Creating a Markov Chain object
      mc_object[i,t] = qe.MarkovChain(P = trans_Matrix[i,t,:,:], state_values=possible_states)

      # Sequence of hidden states
      update = mc_object[i,t].simulate(ts_length= 2, init=str(states_seq[i,t-1]))        
      states_seq[i,t] = update[1]
      hidden_state[i,t]=int(states_seq[i,t])

      #Autoregressive model
      eta_y[:,i,t] = rnd.multivariate_normal(
        mu[hidden_state[i,t]]+B[hidden_state[i,t]] @ (eta_y[:,i,t-1]-mu[hidden_state[i,t]]),Omega_K) 
      epsilonY[:,i,t] = rnd.normal(0,sigmaY,(K_dim*J_dim))
      Y[:,i,t] = lambdaY @ eta_y[:,i,t] + epsilonY[:,i,t]
      
      # Missing data
      Y[Y < 0.99] = X[0,1] = X[0,3] = X[0,5] = np.nan

  return {'N':N_dim, 'T':T_dim, 'Y':Y, 'X':X,'X_evt':X_evt} 

#### Simulation
n_dim,t_dim,k_dim=[122,50,3]
data=data_generation(n_dim,t_dim,k_dim)
print(n_dim,t_dim,k_dim)

Y_data=np.stack([data['Y'][0],data['Y'][1],data['Y'][2],
                data['Y'][3],data['Y'][4],data['Y'][5],
                data['Y'][6],data['Y'][7],data['Y'][8]],axis=2)

X_data = data["X"]
X_evt = data["X_evt"]

#### Splitting missing and observed data 
# Since Stan does not accept nan value, we need to replace them with placeholders (-99 in our case).
# Meanwhile, we also identify where those nan value are located. Once we obtain their coordinate, we 
# introduce them into Stan and assign to these placeholders priors (the same distribution as the observed 
# data).  

## Withing level data
# Missing data
y_miss = np.argwhere(np.isnan(Y_data))
y_miss1 = y_miss[:,0]
y_miss2 = y_miss[:,1]
y_miss3 = y_miss[:,2]

Y_data[np.isnan(Y_data)] = -99

## Between level data
x_miss = np.argwhere(np.isnan(X_data.T))
x_miss1 = x_miss[0,0]
x_miss2 = x_miss[1,0]
x_miss3 = x_miss[2,0]

X_data[np.isnan(X_data)] = -99

##########################################################################################
############################## NUTS-HMC Estimation with cmdstan ##########################
##########################################################################################
from cmdstanpy.model import CmdStanModel
import arviz as az
import os

# Now, we estimate the models using the NUTS-HMC implemented in cmdstan. 
# Note that we originally aim to compare regularization methods.
# To do so, we specify the "methods" variable to be the name of the 
# regularization methods we want to use.

methods = 'Ridge-0' #['Ridge-0','Ridge-1',
                    # 'B-lasso',
                    # 'ABSS-lasso-0','ABSS-lasso-1','ABSS-lasso-2',
                    # 'regHS-0','regHS-1','regHS-2']

## Prepare data dictionary for stan
# Here we gather the all information aout the data we want to transfer into stan
# Note that we need to specify the scale parameter for the reg. HS priors 
# that we denote by sigma_s. Following the description in table 5 in the manuscript,  
# the NDLCSEM_data dictionary need to include:
#            - 'sigma_s':np.pi/np.sqrt(3*n_dim*t_dim) for reg. HS-0 and reg. HS-1
#            - 'sigma_s':2/np.sqrt(n_dim*t_dim) for reg. HS-2

NDLCSEM_data = {
  'N':n_dim, 'T':t_dim, 
  'y':Y_data, 
  'N_ymiss':len(y_miss),
  'y_miss1':y_miss1+1, 
  'y_miss2':y_miss2+1,
  'y_miss3':y_miss3+1, 
  'x_miss1':x_miss1+1, 
  'x_miss2':x_miss2+1,
  'x_miss3':x_miss3+1,
  'X':X_data.T,
  'X_evt':X_evt
}  

# Launch stan
cmd_stanmodel = CmdStanModel(
  stan_file=methods+'.stan')
vi_init = cmd_stanmodel.variational(
    data=NDLCSEM_data, 
    require_converged=False,
    draws=200,grad_samples = 50,
    seed=123,inits=0.0)

# create a dictionary with init values 
n_chains,n_samples,n_warmups=[4,2000,2000]
init_dict = {var: samples.mean(axis=0) for var, samples in vi_init.stan_variables(mean=False).items()}
init_chains= [init_dict for _ in range(n_chains)]

# NUTS-HMC sampling with NUTS-HMC
post_sample = cmd_stanmodel.sample(
  data=NDLCSEM_data,chains=n_chains,iter_warmup=n_warmups,iter_sampling=n_samples,
  inits=init_chains,seed=60424,show_progress=True) 

# convert cmdstanpy to arviz 
fit_az = az.from_cmdstanpy(
      posterior=post_sample,observed_data={'y':NDLCSEM_data['y'],'X':NDLCSEM_data['X']},
      log_likelihood="Lkd_MSAR",
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
      'beta_': ["chain","draw","var12"]})

# Variable names
var_names = ['loady','sigma_y',
            'alpha_','B_',
            'tau_e','L_e',
            'loadX','sigma_X',
            'sigma_u',
            'beta_0','beta_'] 

# additional functions
func_dict = {
  "mode": lambda x: np.max(x),        
  "5%":lambda x: np.percentile(x,5),  
  "95%":lambda x: np.percentile(x,95),
}

# Results
results_fit_az = az.summary(fit_az, var_names=var_names,stat_funcs=func_dict)
results_fit_df = pd.DataFrame(results_fit_az)
results_fit_df.to_csv(methods+" results.csv")

