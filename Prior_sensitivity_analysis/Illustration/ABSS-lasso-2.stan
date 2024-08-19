data {
  int<lower=1> N; 
  int<lower=1> T;
  array[N,T] vector[9] y;
  int<lower=1> N_ymiss; 
  array[N_ymiss] int y_miss1;
  array[N_ymiss] int y_miss2;
  array[N_ymiss] int y_miss3;
  int x_miss1;
  int x_miss2;
  int x_miss3;
  array[N] vector[3] X;
  array[N] vector[5] X_evt;
}
parameters {
  vector<lower=0>[6] loady;
  vector<lower=0>[2] loadX;
  matrix[3,3] B_;
  vector<lower=0>[3] alpha_;
  real<lower=0> beta_0;
  vector<lower=0>[27] lambda;
  vector[27] beta_slab;
  vector<lower=0,upper=1>[27] pi_;   
  vector[N] u_X;
  array[N] vector[3] eta_ini;
  array[N,T] vector[3] eta_y;
  vector<lower=0>[3] tau_e; 
  cholesky_factor_corr[3] L_e;
  real<lower=0> sigma_X;
  vector<lower=0>[3] sigma_y;
  real<lower=0> sigma_u;
  vector[N_ymiss] y_impute;        
}  
transformed parameters {
  array[N,T] vector[2] Lkd_AR;
  array[N,T] vector[2] alpha_var;
  array[N,T] real Lkd_MSAR;
  array[N,T] vector<lower=0,upper=1>[2] M;
  real<lower=0,upper=1> prob_ini;
  array[N,T] vector[3] mueta_y1;
  array[N,T] vector[3] mueta_y2;
  vector[N] eta_X;
  matrix[3,3] L_ = diag_pre_multiply(tau_e,L_e);
  vector[27] beta_raw = pi_ .* beta_slab;
  vector[27] beta_ = beta_raw ./ lambda;
  real beta_x = beta_[1];
  vector[5] beta_evt = beta_[2:6]; 
  vector[3] beta_y= beta_[7:9];
  vector[3] beta_xy= beta_[10:12];
  vector[5] beta_evty1= beta_[13:17];
  vector[5] beta_evty2= beta_[18:22];
  vector[5] beta_evty3= beta_[23:27];
 
  // Missing data treatments
  array[N,T] vector[9] y_merge;
  y_merge = y;
  for(u in 1:N_ymiss){
    y_merge[y_miss1[u],y_miss2[u],y_miss3[u]] = y_impute[u];
  }

  //////// Hamilton Filter
  for (i in 1:N){
    // Factor model
    eta_X[i] = sigma_u * u_X[i];

    ////// Transition probability
    M[i,1,1] = 1; 
    M[i,1,2] = 0.95; 
    prob_ini = 1; 

    real nu;
    for (t in 2:T){      
      nu = beta_0+eta_X[i]*beta_x+
           X_evt[i]' * beta_evt +
           eta_y[i,t-1]' * beta_y+
           eta_X[i]*eta_y[i,t-1]' * beta_xy+
           eta_y[i,t-1,1]*X_evt[i]' * beta_evty1+
           eta_y[i,t-1,2]*X_evt[i]' * beta_evty2+
           eta_y[i,t-1,3]*X_evt[i]' * beta_evty3;
      
      M[i,t,1] = inv_logit(nu); 
      M[i,t,2] = 0.95;
    }    

    ////// For t=1
    // Likelihood for the AR(1)
    mueta_y1[i,1] = rep_vector(0,3);
    Lkd_AR[i,1,1] = exp(multi_normal_cholesky_lpdf(eta_y[i,1]|mueta_y1[i,1],L_)); 

    mueta_y2[i,1] = alpha_+ B_* (eta_ini[i]);
    Lkd_AR[i,1,2] = exp(multi_normal_cholesky_lpdf(eta_y[i,1]|mueta_y2[i,1],L_));

    // Likelihood for the MSAR(1)        
    Lkd_MSAR[i,1] = M[i,1,1] *prob_ini* Lkd_AR[i,1,1] +
                (1-M[i,1,1]) *prob_ini* Lkd_AR[i,1,2] +
                M[i,1,2]  *(1-prob_ini)* Lkd_AR[i,1,2] +
                (1-M[i,1,2])*(1-prob_ini) * Lkd_AR[i,1,1];        
    
    alpha_var[i,1,1] = (M[i,1,1] *prob_ini* Lkd_AR[i,1,1] + 
                (1-M[i,1,2])*(1-prob_ini) * Lkd_AR[i,1,1])/Lkd_MSAR[i,1];
    
    alpha_var[i,1,2] = 1 - alpha_var[i,1,1];

    ////// For t>1
    for (t in 2:T){

      // Likelihood for the AR(1)
      mueta_y1[i,t] = rep_vector(0,3);
      Lkd_AR[i,t,1] = exp(multi_normal_cholesky_lpdf(eta_y[i,t]|mueta_y1[i,t],L_)); 
      
      mueta_y2[i,t] = alpha_  + B_ * (eta_y[i,t-1]);
      Lkd_AR[i,t,2] = exp(multi_normal_cholesky_lpdf(eta_y[i,t]|mueta_y2[i,t],L_));

      // Likelihood for the MSAR(1)        
      Lkd_MSAR[i,t] = M[i,t,1] * alpha_var[i,t-1,1] * Lkd_AR[i,t,1] +
                  (1-M[i,t,1]) * alpha_var[i,t-1,1] * Lkd_AR[i,t,2] +
                  M[i,t,2] * alpha_var[i,t-1,2] * Lkd_AR[i,t,2] +
                  (1-M[i,t,2]) * alpha_var[i,t-1,2] * Lkd_AR[i,t,1];    
      
      alpha_var[i,t,1] = (M[i,t,1] * alpha_var[i,t-1,1] * Lkd_AR[i,t,1] + 
                  (1-M[i,t,2]) * alpha_var[i,t-1,2] * Lkd_AR[i,t,1])/Lkd_MSAR[i,t];
      
      alpha_var[i,t,2] = 1 - alpha_var[i,t,1];
    }
  }
}
model {
  array[N,T] vector[9] mu_y;
  array[N] vector[3] mu_X;

  ////// Likelihood
  for (i in 1:N) {
    // Between-level model
    u_X[i] ~ std_normal();
    mu_X[i,1] = eta_X[i];
    mu_X[i,2] = loadX[1] * eta_X[i];
    mu_X[i,3] = loadX[2] * eta_X[i];
    if ((i != x_miss1) && (i != x_miss2) && (i != x_miss3)){
      X[i,1] ~ normal(mu_X[i,1],sigma_X);
    }
    X[i,2] ~ normal(mu_X[i,2],sigma_X);
    X[i,3] ~ normal(mu_X[i,3],sigma_X);

    // Within-level measurement model
    // Within-level initial
    eta_ini[i] ~ normal(0,0.1);
    for (t in 1:T) {
      // Dynamic factor model
      // icontent1, 3 items
      mu_y[i,t,1] = eta_y[i,t,1];
      mu_y[i,t,2] = loady[1] * eta_y[i,t,1];
      mu_y[i,t,3] = loady[2] * eta_y[i,t,1];
      y_merge[i,t,1] ~ normal(mu_y[i,t,1],sigma_y[1]);
      y_merge[i,t,2] ~ normal(mu_y[i,t,2],sigma_y[1]);
      y_merge[i,t,3] ~ normal(mu_y[i,t,3],sigma_y[1]);
      // PAP, 3 items
      mu_y[i,t,4] = eta_y[i,t,2];
      mu_y[i,t,5] = loady[3] * eta_y[i,t,2];
      mu_y[i,t,6] = loady[4] * eta_y[i,t,2];
      y_merge[i,t,4] ~ normal(mu_y[i,t,4],sigma_y[2]);
      y_merge[i,t,5] ~ normal(mu_y[i,t,5],sigma_y[2]);
      y_merge[i,t,6] ~ normal(mu_y[i,t,6],sigma_y[2]);
      // PAN, 3 items
      mu_y[i,t,7] = eta_y[i,t,3];
      mu_y[i,t,8] = loady[5] * eta_y[i,t,3];
      mu_y[i,t,9] = loady[6] * eta_y[i,t,3];
      y_merge[i,t,7] ~ normal(mu_y[i,t,7],sigma_y[3]);
      y_merge[i,t,8] ~ normal(mu_y[i,t,8],sigma_y[3]);
      y_merge[i,t,9] ~ normal(mu_y[i,t,9],sigma_y[3]);
    }
    // MSAR  
    target += sum(log(Lkd_MSAR[i]));
  }

  ////// Priors      
  // Within-level
  loady ~ std_normal();
  loadX ~ std_normal();
  sigma_y ~ cauchy(0,2.5); 

  to_vector(B_) ~ std_normal();
  alpha_ ~ normal(0,10);
  L_e ~ lkj_corr_cholesky(4.0);
  tau_e ~ cauchy(0,2.5); 

  // Between-level
  sigma_X ~ cauchy(0,2.5); 
  sigma_u ~ cauchy(0,2.5);

  beta_0 ~ normal(0,10); 
  lambda ~ cauchy(0,2.5);
  pi_ ~ beta(0.5,0.5); 
  beta_slab ~ double_exponential(0,4);
}
generated quantities{}
