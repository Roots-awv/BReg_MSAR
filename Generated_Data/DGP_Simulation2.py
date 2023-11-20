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

def data_generation(N_dim,T_dim,P_dim,phi_corr,var_effect):
  ##### Dimension
  #N_dim = 100
  #T_dim = 70
  #P_dim = 7
  ##### Parameters
  ## L2
  ## Covariance matrix
  #var_effect = 1
  #phi_corr = 0.5 
  phi_var = 1
  Phi_corrX =  np.ones((P_dim,P_dim)) *phi_corr - np.eye(P_dim) * (phi_corr-phi_var) 
  Phi_X = Phi_corrX *var_effect
  sigma_logit = np.pi/sqrt(3) 
  ## Coefficients
  #if (P_dim % 2) == 1: # if 1 odd number  
  #  var_num = np.int(1 + P_dim/2)
  #else:                # if 0 even number 
  var_num = np.int(P_dim/2)
  x_beta = [0.5,-0.5] * var_num
  x_beta = x_beta[:var_num]
  x_beta.extend([0]*(P_dim-var_num))
  beta_x = np.array(x_beta)
  beta_x
  # beta_0 = 2.5
  beta_z = 0.5
  z_beta = [0]*var_num
  z_b = ([0.5,-0.5]* (P_dim-var_num))
  z_b = z_b[:(P_dim-var_num)]
  z_beta.extend(z_b)
  beta_xz = np.array(z_beta)#(beta_xz[:P_dim]) 
  beta_xz
  ## L1
  sigma_eps = 0.25
  rho = np.array([0,0.8])
  gamma = 0.5
  mu = np.array([0,3])
  ###### Covariates
  #Phi_W = np.eye(P_dim + T_dim) 
  #Phi_W[:P_dim, :P_dim] = Phi_X
  #Phi_W[P_dim:, :P_dim]=Phi_W[:P_dim, P_dim:]=0
  #W = rnd.multivariate_normal(np.zeros(P_dim+ T_dim),Phi_W,N_dim)  
  #W[:,2].mean()
  #X = W[:,:P_dim]
  #X[:,1].mean()
  #Z = W[:,P_dim:]
  #Z[:,1].mean()
  X = rnd.multivariate_normal(np.zeros(P_dim),Phi_X,N_dim)
  Z = rnd.normal(0,var_effect,(N_dim,T_dim))  #phi_xz=[]
  cov_XZ = np.cov(X.T, Z.T)
  #  mat_cov = np.cov(X[:,0],Z[:,t])
  #  mat_cov = mat_cov[0,1]
  #  phi_xz.append(mat_cov)
  #phi_txz = - np.array(phi_xz)
  ###### States (chains) and transition probabilities
  # initialization 
  eta_fei = X @ beta_x  
  eta_fes = X @ beta_xz    
  cov_fesZ = np.cov(eta_fes.T, Z.T)
  # Transition matrix 
  beta_0 = 3 #- (eta_fes @ Z).mean()
  beta_0
  #(cov_XZ[P_dim:, :P_dim] @ beta_xz).sum()
  cov_fesZ[P_dim:, :P_dim].sum()
  (eta_fes @ Z).mean() - eta_fes.mean()**2 * Z.mean()**2
  sigma_ri = 0.125 # Var_fe * ((1-Rsq)/Rsq) - sigma_logit**2
  sigma_rs = 0
  eps_ri = rnd.normal(0,sigma_ri,(N_dim))
  eps_rs = rnd.normal(0,sigma_rs,(N_dim))

  trans_Matrix = np.zeros((N_dim,T_dim,2,2)) 
  possible_states = ("0", "1") # For example "1" if a student have an intention to quit
                                 #             "0" if a student have an intention to stay
  states_seq = np.zeros((N_dim,T_dim),dtype=object)
  mc_object = np.zeros((N_dim,T_dim),dtype=object)  
  eta_p = np.zeros((N_dim,T_dim)) 
  for t in range(T_dim):
    for i in range(N_dim):
      eta_p[i,t] = eta_fei[i] + Z[i,t] * beta_z + Z[i,t] * eta_fes[i] + beta_0
      trans_Matrix[i,t,0,0] = inv_logit(eta_p[i,t] + eps_ri[i] + Z[i,t] * eps_rs[i]) 
      trans_Matrix[i,t,0,1] = 1- trans_Matrix[i,t,0,0]
      # Here, I assumed that once individuals move from 0 to 1 they would 
      # have a low probability of coming back i-e to move from 1 to 0
      trans_Matrix[i,t,1,0] = 0.05#np.full(N_dim,0.05) 
      trans_Matrix[i,t,1,1] = 1 - trans_Matrix[i,t,1,0] 
  #var_eta= sigma_ri**2+ beta_x.T @ Phi_X @ beta_x + beta_z**2 * var_effect + (cov_XZ[P_dim:, :P_dim] @ beta_xz).mean()  
  Rsq=(eta_p.var()+sigma_ri**2+sigma_rs**2)/(eta_p.var()+sigma_ri**2+sigma_rs**2+sigma_logit**2)
  Rsq
  ## Creates transition matrix 
  # loop
  for i in range(N_dim) :
    states_seq[i,0] = '0'
    for t in range(1,T_dim):
      # Creating a Markov Chain object
      mc_object[i,t] = qe.MarkovChain(P = trans_Matrix[i,t,:,:], state_values=possible_states)
      # Sequence of hidden states
      update = mc_object[i,t].simulate(ts_length= 2, init=str(states_seq[i,t-1]))        
      states_seq[i,t] = update[1]
  
  # Converting hidden state to integers
  hidden_state = np.zeros((N_dim,T_dim))
  for i in range(N_dim):
    for t in range(T_dim):
      hidden_state[i,t]=int(states_seq[i,t])
  hidden_state = hidden_state.astype(int) 
  
  ##### Within-level / Markov switching Autoregressive  
  y = np.zeros((N_dim,T_dim))
  for i in range(N_dim):
      y[i,0] = mu[hidden_state[i,0]]+ rnd.normal(gamma*Z[i,0],(sigma_eps)/sqrt(1-rho[hidden_state[i,0]]**2))
      for t in range(1,T_dim):
          y[i,t] = rnd.normal(mu[hidden_state[i,t]] + 
              rho[hidden_state[i,t]] * (y[i,t-1]-mu[hidden_state[i,t]]) +gamma*Z[i,t],sigma_eps) 
  ##### Data list
  param = np.hstack((mu[1],rho[1],gamma,sigma_eps,beta_0,beta_x,beta_z,beta_xz,sigma_ri)) #,sigma_eta
  return {'N':N_dim, 'T':T_dim, 'P':P_dim,'Theta':param, 'y':y.T, 'X':X, 'Z':Z.T,'Rsq':Rsq,'beta_0':beta_0} 

##### Conditions

for t_dim in [30,40,50,60,70]:  # Conditions on the length of time pts.
  for rep in range(200):        # Replications     
    #rep  = 1 t_dim =50
    #### Simulation
    n_dim,p_dim,corr_,var_=[100,7,0.3,1]
    data=data_generation(n_dim,t_dim,p_dim,corr_,var_)
    [data['Rsq'],data['beta_0']]
    #### Data frame conversion
    Theta = pd.DataFrame(data['Theta'],columns=['Theta'])
    MSAR1_Y_df =pd.DataFrame(data['y'])
    MSAR1_Xdf = pd.DataFrame(data['X'],columns=['X'+str(p+1) for p in range(data['P'])])
    MSAR1_Z_df =pd.DataFrame(data['Z'])
    #### Contition and Folders
    corr_c=np.int(corr_*100)
    var_c=np.int(var_*100)
    folder='N'+str(n_dim)+'_T'+str(t_dim)+'_P'+str(p_dim)+'_corr'+str(corr_c)+'_var'+str(var_c)+'/'
    ### Save files in folder for each conditions
    Theta.to_csv(folder+'Theta_tvtp.csv', index=False)
    MSAR1_Y_df.to_csv(folder+'Y_repnum'+str(rep+1)+'tvtp.csv', index=False)
    MSAR1_Xdf.to_csv(folder +'X_repnum'+str(rep+1)+'tvtp.csv', index=False)
    MSAR1_Z_df.to_csv(folder+'Z_repnum'+str(rep+1)+'tvtp.csv', index=False)              


