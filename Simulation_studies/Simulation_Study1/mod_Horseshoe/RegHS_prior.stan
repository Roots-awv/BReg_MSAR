data {
  int<lower=1> N; 
  int<lower=1> T;
  int<lower=1> P;
  matrix[T,N] y;
  matrix[N,P] X; 
  real<lower=0> scale_global ;  // scale for the half - t prior for tau
}
parameters {
  real<lower=0,upper=1> rho;
  real<lower=0> alpha;
  real<lower=0> sigma_e;
  real<lower=0> sigma_u;
  vector[N] u;
  real<lower=0> beta_0;
  real<lower=0> tau;                     // Horseshoe hyperparameters
  vector<lower=0>[P] lambda;
  vector[P] beta_raw;
  real<lower=0> caux;
}  
transformed parameters {
  matrix[N,T] Lkd_AR[2];
  matrix<lower=0,upper=1>[N,T] c_prob_s[2];
  matrix[N,T] Lkd_MSAR;
  vector<lower=0,upper=1>[2] M[N];
  real<lower=0,upper=1> prob_ini[N];
  // Horseshoe 
  vector<lower=0>[P] lambda_tilde ;                 // ’ truncated ’ local shrinkage parameter
  vector[P] beta_eta ;                             // regression coefficients
  vector[N] mu_eta;                                // latent function values
  lambda_tilde = sqrt(caux * square(lambda) ./ (caux + tau^2 * square(lambda)));
  beta_eta = beta_raw .* lambda_tilde * tau;
  mu_eta = X * beta_eta;
  for (i in 1:N){
    ////// Transition probability
    M[i,1] = inv_logit(mu_eta[i] + u[i] *sigma_u+ beta_0);
    M[i,2] = 0.95;
    prob_ini[i] = 1; //(1-M[i,1])/(2-M[i,1]-M[i,2]);

    ////// For t=1
    // Likelihood for the AR(1)
    Lkd_AR[1,i,1] = exp(normal_lpdf(y[1,i] |0,sigma_e)); // No influence of rho on the variance
    Lkd_AR[2,i,1] = exp(normal_lpdf(y[1,i] |alpha,sigma_e/(1 - (rho^2))));   
    // Likelihood for the MSAR(1)        
    Lkd_MSAR[i,1] = M[i,1] * prob_ini[i] * Lkd_AR[1,i,1] +
                (1-M[i,1]) * prob_ini[i] * Lkd_AR[2,i,1] +
                M[i,2] * (1-prob_ini[i]) * Lkd_AR[2,i,1] +
                (1-M[i,2]) * (1-prob_ini[i]) * Lkd_AR[1,i,1];        
    c_prob_s[1,i,1] = (M[i,1] * prob_ini[i] * Lkd_AR[1,i,1] + 
                (1-M[i,2]) * (1-prob_ini[i]) * Lkd_AR[1,i,1])/Lkd_MSAR[i,1];
    c_prob_s[2,i,1] = 1 - c_prob_s[1,i,1];
    
    ////// For t>1
    for (t in 2:T){
      // Likelihood for the AR(1)
      Lkd_AR[1,i,t] = exp(normal_lpdf(y[t,i] |0 , sigma_e)); 
      Lkd_AR[2,i,t] = exp(normal_lpdf(y[t,i] |alpha+ rho * (y[t-1,i]-alpha), sigma_e));
      // Likelihood for the MSAR(1)        
      Lkd_MSAR[i,t] = M[i,1] * c_prob_s[1,i,t-1] * Lkd_AR[1,i,t] +
                  (1-M[i,1]) * c_prob_s[1,i,t-1] * Lkd_AR[2,i,t] +
                  M[i,2] * c_prob_s[2,i,t-1] * Lkd_AR[2,i,t] +
                  (1-M[i,2]) * c_prob_s[2,i,t-1] * Lkd_AR[1,i,t];    
      c_prob_s[1,i,t] = (M[i,1] * c_prob_s[1,i,t-1] * Lkd_AR[1,i,t] + 
                  (1-M[i,2]) * c_prob_s[2,i,t-1] * Lkd_AR[1,i,t])/Lkd_MSAR[i,t];
      c_prob_s[2,i,t] = 1 - c_prob_s[1,i,t];
    }
  }
}
model {
  ////// Likelihood      
  for (i in 1:N) {
    u[i] ~ normal(0,1);
    target += sum(log(Lkd_MSAR[i]));
  }
  
  ////// Priors      
  // Within-level
  rho ~ beta(1,1);
  alpha ~ normal(0,10);
  sigma_e ~ cauchy(0,2.5); 
  
  // Between-level
  beta_0 ~ cauchy(0,5); 
  beta_raw ~ normal(0,1);                       // Horseshoe hyperprior
  lambda ~ cauchy(0,1);
  tau ~ cauchy(0,scale_global);
  caux ~ inv_gamma(0.5,0.5);
  sigma_u ~ cauchy(0,2.5); 
}
