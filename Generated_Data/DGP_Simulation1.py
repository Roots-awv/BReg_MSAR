import numpy as np
import numpy.random as rnd
import pandas as pd
import quantecon as qe
from matplotlib import pyplot as plt
from math import sqrt

############################## SIMULATION ##############################
rnd.seed(171122)
##### Functions
def inv_logit(x):
  return 1/(1+np.exp(-x))

def data_generation(N_dim,T_dim,P_dim,non_zero,var_effect):
  ##### Dimension
  # N_dim = 100
  # T_dim = 50
  # P_dim = 25
  #### Parameters
  #### L2
  #### Covariance matrix
  # var_effect =1
  # non_zero = 0.25
  phi_corr = 0.3 
  phi_var = 1
  sigma_logit = np.pi/sqrt(3) 
  Phi_corrX =  np.ones((P_dim,P_dim)) *phi_corr - np.eye(P_dim) * (phi_corr-phi_var) 
  Phi_X = Phi_corrX *var_effect
  # Coefficients
  var_num = np.round(P_dim*non_zero)
  all_beta = [0.5,-0.5] * np.int(var_num)
  beta_eta = all_beta[:np.int(var_num)]
  beta_eta.extend([0]*(P_dim-np.int(var_num)))
  beta_eta = np.array(beta_eta)   
  beta_eta
  ## L1
  sigma_eps = 0.25
  rho = np.array([0,0.8])
  mu = np.array([0,3])
  ###### Between level
  X = rnd.multivariate_normal(np.zeros(P_dim),Phi_X,N_dim)
  # (np.transpose(X) @ X)/N_dim  
  ## The variance of X @ beta eta_i is set to be equal to 30% of the variance of eta_i :
  sigma_u = 0.125 # sqrt(sigma_logit*(0.25/(1-0.25))-(beta_eta.T @ Phi_X @ beta_eta))
  eta_i =X @ beta_eta + rnd.normal(0,sigma_u,N_dim)
  Rsq = (beta_eta.T @ Phi_X @ beta_eta+sigma_u**2)/(beta_eta.T @ Phi_X @ beta_eta+sigma_u**2+sigma_logit**2)
  Rsq
  #(sigma_u**2 + np.var(X @ beta_eta))/(sigma_u**2 + np.var(X @ beta_eta)+sigma_logit)  
  ###### States (chains) and transition probabilities
  beta_0 = 3
  P00 = np.exp((eta_i+beta_0))/(1+np.exp(eta_i+beta_0)) 
  P01 = 1- P00
  np.std(np.log(P00/P01)) - sigma_logit
  #np.exp(0.13326979)
  # Here, I assumed that once individuals move from 0 to 1 they would 
  # have a low probability of coming back i-e to move from 1 to 0
  P10 = np.full(N_dim,0.05) 
  P11 = 1 - P10
  ## Creates transition matrix 
  # initialization
  possible_states = ("0", "1") # For example "1" if a student have an intention to quit
                               #             "0" if a student have an intention to stay
  trans_Matrix = np.zeros((N_dim,2,2))
  trans_proba = np.zeros((N_dim,2))
  mc_object = np.zeros(N_dim, dtype=object)
  states_seq = np.zeros(N_dim, dtype=object)
  # loop
  for i in range(N_dim) :
      # Transition matrix 
      trans_Matrix[i] = np.array([[P00[i],P01[i]], [P10[i],P11[i]]])
      trans_proba[i] = np.diag(trans_Matrix[i])
      # Creating a Markov Chain object
      mc_object[i] = qe.MarkovChain(P = trans_Matrix[i], state_values=possible_states)
      # Sequence of hidden states
      states_seq[i] = mc_object[i].simulate(ts_length= T_dim , init='0')
  states_seq[0].shape
  # Converting hidden state to integers
  hidden_state = np.zeros((N_dim,T_dim))
  for i in range(N_dim):
    for t in range(T_dim):
      hidden_state[i,t]=int(states_seq[i][t])
  hidden_state= hidden_state.astype(int)
  hidden_state[:,1].mean()
  ##### Within-level / Markov switching Autoregressive  
  y = np.zeros((N_dim,T_dim))
  for i in range(N_dim):
      y[i,0] = mu[hidden_state[i,0]]+ rnd.normal(0,(sigma_eps)/(1-rho[hidden_state[i,0]]**2))
      for t in range(1,T_dim):
          y[i,t] = rnd.normal(mu[hidden_state[i,t]] + 
              rho[hidden_state[i,t]] * (y[i,t-1]-mu[hidden_state[i,t]]) ,sigma_eps) 
  ##### Data list
  param = np.hstack((mu[1],rho[1],sigma_eps,beta_eta,beta_0)) #,sigma_eta
  return {'N':N_dim, 'T':T_dim, 'P':P_dim,'Theta':param, 'y':y.T, 'X':X,'Rsq':Rsq} 

for n_dim in [50,75,100]:                # Conditions on the nb. of individuals   
  for t_dim in [10,25,50]:               # Conditions on the length of time pts.
      for p_dim in [5,15,25]:            # Conditions on nb. of level 2 covariates
        for nz_ in [0.25,0.5]:          # Conditions on the level 2 covariates correlations  
          for rep in range(200):       # Replications     
            #### Simulation
            var_ = 1
            data=data_generation(n_dim,t_dim,p_dim,nz_,var_)
            print('N'+str(n_dim)+'_T'+str(t_dim)+'_P'+str(p_dim)+'_nz'+str(nz_),data['Rsq'])
            #### Data frame conversion
            Theta = pd.DataFrame(data['Theta'],columns=['Theta'])
            MSAR1_Y_df =pd.DataFrame(data['y'])
            MSAR1_Xdf = pd.DataFrame(data['X'],columns=['X'+str(p+1) for p in range(data['P'])])
            #### Contition and Folders
            nz_c=np.int(nz_*100)
            var_c=np.int(var_*100)
            folder='../FINAL_RESULTS_S1/'+'N'+str(n_dim)+'_T'+str(t_dim)+'_P'+str(p_dim)+'_nz'+str(nz_c)+'_var'+str(var_c)#+'/'
            ### Save files in folder for each conditions
            Theta.to_csv(folder+'Theta_bl.csv', index=False)
            MSAR1_Y_df.to_csv(folder+'Y_repnum'+str(rep+1)+'.csv', index=False)
            MSAR1_Xdf.to_csv(folder +'X_repnum'+str(rep+1)+'.csv', index=False)


