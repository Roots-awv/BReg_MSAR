# Bayesian Regularization DLVM

This repository contains supplementary materials and codes used in the manuscript titled: _"Are Bayesian Regularization Methods a Must for Multilevel Dynamic Latent Variables Models?"_

Figures that was not presented in the manuscript were included in the pdf document entitled _"supplementary_materials.pdf"_.  In particular, convergence rates and sampling accuracy rates were presented. In addition, we presented measures such as relative bias, absolute bias, root mean square error (RMSE), coverage rates and type I error rates.

We also shared the code and syntax used to generate the data and calculate the results. 
 - First, the _"Generated_Data"_ folder contains 2 Python files that allowed us to generate the data in simulation study 1 and simulation study 2 , respectively.
 - Second, we added two folders called "Simulation_Study1" and "Simulation_Study2". Each folder had four subfolders: "mod_aBSS_lasso", "mod_B_lasso", "mod_Horseshoe", "mod_Weak_info". Each of these subfolders respectively corresponds to one of the four prior distributions used in the manuscript. They each contain a Stan file and an R file each that were used to perform the MCMC sampling across data conditions and replications.
 - Third, _"Simulation_Study1"_ and _"Simulation_Study2"_ contain one R file and one .rds file each. The R files served to compute the perfomance measures (Bias, RMSE, convergence rates,...) and returned a .rds files containing a list of table of results.  

Note that instead of submitting one job per prior distribution, we split the workload into smaller sub-jobs. For example in the _"mod_aBSS_lasso"_ folder, instead of setting the number of replication to ```
M<-1:200``` and submitting only one job, we divided it into 10 smaller jobs, each handling 20 replicates. In this setup, the first job had to deal with replicate 1 to 20, the second job with replicate 21 to 40, and so on. To do that, we wrote the following lines in file _"MSAR_absslasso_wv1.R"_:
```
  M_blocks <- 0:10*20
  M <- (M_blocks[1]+1):M_blocks[1+1] 
```
After that, we created a new file called _"MSAR_absslasso_wv2.R"_ with the same script as _"MSAR_absslasso_wv1.R"_ but with the lines:
```
  M_blocks <- 0:10*20
  M <- (M_blocks[2]+1):M_blocks[2+1] 
```
We relied on these divisions of tasks throughout these two simulation studies. 
 
