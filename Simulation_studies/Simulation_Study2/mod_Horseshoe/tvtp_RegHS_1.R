########################################################################
# install.packages('rstan')
# install.packages('readr')
library(rstan)
library(readr)

options(mc.cores = parallel::detectCores())

ptm <- proc.time()
M_blocks <- 0:10*20
M <- 1:100 #(M_blocks[0]+1):M_blocks[0+1] 

n_dim <- 100
p_dim <- 7
corr_ <- 30
var_ <- 100

for (t_dim in c(30,40,50,60,70)){
  nom <- paste0("N",n_dim,"_T",t_dim,"_P",p_dim,"_corr",corr_,"_var",100)
  path <- paste0("~/BReg_MSAR/Generated_Data/",nom)
  Theta <- read_csv(paste0(path,"/Theta_tvtp.csv"),show_col_types = F)
  for (m in M){
    ### Load data
      y_df <- read_csv(paste0(path,"/Y_repnum",m,"tvtp.csv"),show_col_types = F)  # Within level data
      X_df <- read_csv(paste0(path,"/X_repnum",m,"tvtp.csv"),show_col_types = F)  # Between level data
      Z_df <- read_csv(paste0(path,"/Z_repnum",m,"tvtp.csv"),show_col_types = F)  # Between level data
    # prepare data for Stan
    mod_data <- list(
      N = dim(X_df)[1],
      T = dim(y_df)[1],
      P = dim(X_df)[2],
      y = y_df,
      X = X_df,
      Z = Z_df,
      scale_global = (pi/sqrt(3))/sqrt(n_dim*t_dim) # (nz_/100)/(1-(nz_/100))
    )
    # parameters
    pars = c('alpha','rho','gamma','sigma_e','beta_0','beta_x','beta_z','beta_xz','sigma_x')
    # Stan sampling
    p_var <- stan_model("RegHS_prior_tvtp.stan")
    estimates <- sampling(p_var, data = mod_data,chains = 4 ,iter = 2000,pars = pars,seed = 111122)
    # Estimates
    est_chains = summary(estimates,pars, probs = c(0.025, 0.975))  
    posterior_0 = est_chains$summary
    remove_<-paste0('M[',as.character(1:n_dim),',2]')
    posterior<-posterior_0[!rownames(posterior_0) %in% remove_,]

    write.csv(posterior,file=paste0(nom,'RegHS_prior_tvtp_rep',m,'.csv'))            
  } 
}   
proc.time() - ptm 
########################################################################
