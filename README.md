# Bayesian Regularization DLVM

This repository contains additional materials and codes used in the manuscript titled: _"Are Bayesian Regularization Methods a Must for Multilevel Dynamic Latent Variables Models?"_

Figures that could not be presented in the manuscript are included in the pdf document entitled _"supplementary_materials.pdf"_.  In particular, convergence rates and sampling accuracy rates are presented. In addition, we have also presented measures such as relative bias, absolute bias, root mean square error (RMSE), coverage rates and type I error rates.

We  also shared the code and syntax used to generate the data and calculate the results. 
 - First, we added the folder _"Generated_Data"_ containing 2 Python files that allow us to generate the data in simulation study 1 and simulation study 2 , respectively.
 - Second, we added the folder that contains four subfolders. Each subfolder respectively corresponds to each of four prior distributions used in the manuscript. These subfolder contains a Stan file and a R file each. We also included a R file that compute the perfomance measures (Bias, RMSE, convergence rates,...)

