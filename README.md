# Bayesian Regularization DLVM

This repository contains additional materials and codes used in the manuscript titled: _"Are Bayesian Regularization Methods a Must for Multilevel Dynamic Latent Variables Models?"_

Figures that could not be presented in the manuscript are included in the pdf document entitled _"supplementary_materials.pdf"_.  In particular, convergence rates and sampling accuracy rates are presented. In addition, we have also presented measures such as relative bias, absolute bias, root mean square error (RMSE), coverage rates and type I error rates.

We  also shared the code and syntax used to generate the data and calculate the results. 
 - First, we added the folder _"Generated_Data"_ containing 2 Python files that allow us to generate the data in simulation study 1 and simulation study 2 , respectively.
 - Second, we added two folders _"Simulation_Study1"_ and _"Simulation_Study2"_. Each folder contains four subfolders: _"mod_aBSS_lasso", "mod_B_lasso", "mod_Horseshoe", "mod_Weak_info"_ . Each of these subfolders respectively corresponds to each of four prior distributions used in the manuscript. They contain one Stan file and one R file each that were used to carry out the MCMC sampling across data conditions and replications.
 - Third, _"Simulation_Study1"_ and _"Simulation_Study2"_ included one R file and one .rds file each. The R files compute the perfomance measures (Bias, RMSE, convergence rates,...) and return an .rds files that contain a list of table of results.  

Note that instead of submitting one job per prior distribution, we had submitted severals sub-jobs per prior distribution in parallel. For example in the _"mod_aBSS_lasso"_ folder, instead of fixing the number of replication to ```
M<-1:200``` and submit only one file, we submitted 10 jobs with one group of 20 replicates each. In that case, the first job had to deal with replicate 1 to 20, the second job with replicate 21 to 40, and so on. To do that, we wrote the following lines in file :
```
  M_blocks <- 0:10*20
  M <- (M_blocks[1]+1):M_blocks[1+1] 
```
This means that 
