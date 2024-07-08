########################################################################

# install.packages('rstan')
# install.packages('readr')

library(rstan)
library(readr)

options(mc.cores = parallel::detectCores())

ptm <- proc.time()
M_blocks <- 0:5*40
M <- (M_blocks[1]+1):M_blocks[1+1] 

for (n_dim in c(50,75,100)){
  for (t_dim in c(10,25,50)){
    for (p_dim in c(5,15,25)){
      for (nz_ in c(25,50)){
        nom <- paste0("N",n_dim,"_T",t_dim,"_P",p_dim,"_nz",nz_,"_var",100)
        path <- paste0("~/BReg_MSAR/Generated_Data/",nom)
        Theta <- read_csv(paste0(path,"/Theta.csv"),show_col_types = F)
 
        for (m in M){
          ### Load data
          y_df <- read_csv(paste0(path,"/Y_repnum",m,".csv"),show_col_types = F)  # Within level data
          X_df <- read_csv(paste0(path,"/X_repnum",m,".csv"),show_col_types = F)  # Between level data
          # prepare data for Stan
          mod_data <- list(
            N = dim(X_df)[1],
            T = dim(y_df)[1],
            P = dim(X_df)[2],
            y = y_df,
            X = X_df,
            scale_global = (pi/sqrt(3))/sqrt(n_dim) # (nz_/100)/(1-(nz_/100))
          )
          # parameters
          pars = c('alpha','rho','sigma_e','beta_eta','beta_0','sigma_u')
          # Stan sampling
          p_var <- stan_model("RegHS_prior_prior.stan")
          estimates <- sampling(p_var, data = mod_data,chains = 4 ,iter = 2000,pars = pars,seed = 111122)
          # Estimates
          est_chains = summary(estimates,pars, probs = c(0.025, 0.975))  
          posterior_0 = est_chains$summary
          remove_<-paste0('M[',as.character(1:n_dim),',2]')
          posterior<-posterior_0[!rownames(posterior_0) %in% remove_,]
          
          write.csv(posterior,file=paste0(nom,'RegHS_prior_prior_rep',m,'.csv'))            
        }
      }
    }
  }    
}    
proc.time() - ptm 
########################################################################
