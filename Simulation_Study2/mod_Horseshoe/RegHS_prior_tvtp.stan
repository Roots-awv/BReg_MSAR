data {
  int<lower=1> N; 
  int<lower=1> T;
  int<lower=1> P;
  matrix[T,N] y;
  matrix[T,N] Z;
  matrix[N,P] X;
  real<lower=0> scale_global ;  // scale for the half - t prior for tau
}
parameters {
  real<lower=0,upper=1> rho;
  real<lower=0> alpha;
  real gamma;
  real<lower=0> sigma_e;
  real<lower=0> sigma_x;
  real<lower=0> beta_0;
  vector[N] eps_x;
  real<lower=0> tau;                     // Reg HS hyperparameters
  vector<lower=0>[P] lambda_x;
  vector<lower=0>[P] lambda_xz;
  real<lower=0> lambda_z;
  vector[P] beta_x_raw;
  vector[P] beta_xz_raw;
  real beta_z_raw;
  real<lower=0> caux;
}  
transformed parameters {
  matrix[N,T] Lkd_AR[2];
  matrix<lower=0,upper=1>[N,T] c_prob_s[2];
  matrix[N,T] Lkd_MSAR;
  vector<lower=0,upper=1>[2] M[N,T];
  real<lower=0,upper=1> prob_ini;
  vector<lower=0>[P] lambda_x_tilde ;                 // Reg HS hyperparameters
  vector<lower=0>[P] lambda_xz_tilde ;
  real<lower=0> lambda_z_tilde ;
  vector[P] beta_x ;                             // regression coefficients
  vector[P] beta_xz ;                             // regression coefficients
  real beta_z;
  vector[N] mu_x ;                             // regression coefficients
  vector[N] mu_z ;                             // regression coefficients
  lambda_x_tilde = sqrt(caux * square(lambda_x) ./ (caux + tau^2 * square(lambda_x)));
  lambda_xz_tilde = sqrt(caux * square(lambda_xz) ./ (caux + tau^2 * square(lambda_xz)));
  lambda_z_tilde = sqrt(caux * square(lambda_z) ./ (caux + tau^2 * square(lambda_z)));
  beta_x = beta_x_raw .* lambda_x_tilde * tau;
  beta_xz = beta_xz_raw .* lambda_xz_tilde * tau;
  beta_z = beta_z_raw .* lambda_z_tilde * tau;
  mu_x = X * beta_x_raw; 
  mu_z = X * beta_xz_raw;
  for (i in 1:N){
    ////// Transition probability
    for (t in 1:T){
      M[i,t,1] = inv_logit(mu_x[i] + eps_x[i]*sigma_x + mu_z[i] * Z[t,i] + beta_z * Z[t,i] + beta_0); 
      M[i,t,2] = 0.95;
    }
    prob_ini = 1; //(1-M[i,1,1])/(2-M[i,1,1]-M[i,1,2]);
    ////// For t=1
    // Likelihood for the AR(1)
    Lkd_AR[1,i,1] = exp(normal_lpdf(y[1,i] |gamma * Z[1,i],sigma_e)); // No influence of rho on the variance
    Lkd_AR[2,i,1] = exp(normal_lpdf(y[1,i] |alpha+gamma * Z[1,i],sigma_e/sqrt(1 - (rho^2))));   
    // Likelihood for the MSAR(1)        
    Lkd_MSAR[i,1] = M[i,1,1] * prob_ini * Lkd_AR[1,i,1] +
                (1-M[i,1,1]) * prob_ini * Lkd_AR[2,i,1] +
                M[i,1,2] * (1-prob_ini) * Lkd_AR[2,i,1] +
                (1-M[i,1,2]) * (1-prob_ini) * Lkd_AR[1,i,1];        
    c_prob_s[1,i,1] = (M[i,1,1] * prob_ini * Lkd_AR[1,i,1] + 
                (1-M[i,1,2]) * (1-prob_ini) * Lkd_AR[1,i,1])/Lkd_MSAR[i,1];
    c_prob_s[2,i,1] = 1 - c_prob_s[1,i,1];

    ////// For t>1
    for (t in 2:T){
        // Likelihood for the AR(1)
        Lkd_AR[1,i,t] = exp(normal_lpdf(y[t,i] |gamma * Z[t,i] , sigma_e)); 
        Lkd_AR[2,i,t] = exp(normal_lpdf(y[t,i] |alpha + rho * (y[t-1,i]-alpha)+gamma * Z[t,i], sigma_e));
        // Likelihood for the MSAR(1)        
        Lkd_MSAR[i,t] = M[i,t,1] * c_prob_s[1,i,t-1] * Lkd_AR[1,i,t] +
                    (1-M[i,t,1]) * c_prob_s[1,i,t-1] * Lkd_AR[2,i,t] +
                    M[i,t,2] * c_prob_s[2,i,t-1] * Lkd_AR[2,i,t] +
                    (1-M[i,t,2]) * c_prob_s[2,i,t-1] * Lkd_AR[1,i,t];    
        c_prob_s[1,i,t] = (M[i,t,1] * c_prob_s[1,i,t-1] * Lkd_AR[1,i,t] + 
                    (1-M[i,t,2]) * c_prob_s[2,i,t-1] * Lkd_AR[1,i,t])/Lkd_MSAR[i,t];
        c_prob_s[2,i,t] = 1 - c_prob_s[1,i,t];
    }
  }
}
model {
  ////// Likelihood
  for (i in 1:N) {
    eps_x[i]~ normal(0,1);
    // Likelihood contributions
    target += sum(log(Lkd_MSAR[i]));
  }

  ////// Priors      
  // Within-level
  rho ~ beta(1,1);
  alpha ~ normal(0,10);
  sigma_e ~ cauchy(0,2.5); // This is equivalent to inv_gamma(16,4) for the variance, alternative ==> cauchy(0,2.5); 
  
  // Between-level
  beta_0 ~ cauchy(0,5); 
  beta_x_raw ~ normal(0,1);                       // Horseshoe hyperprior
  beta_xz_raw ~ normal(0,1);                       
  beta_z_raw ~ normal(0,1);                       
  lambda_x ~ cauchy(0,1);
  lambda_xz ~ cauchy(0,1);
  lambda_z ~ cauchy(0,1);
  tau ~ cauchy(0,scale_global);
  caux ~ inv_gamma (0.5,0.5);
  sigma_x ~ cauchy(0,2.5); // This is equivalent to inv_gamma(16,4) for the variance, alternative ==> inv_gamma(10,2);
}
