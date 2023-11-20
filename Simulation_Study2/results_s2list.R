##########################################################################################
library(readr)

options(mc.cores = parallel::detectCores())

################ Conditions & Designs
n <- 100
p <- 7
corr_ <- 30
var_ <- 100
################ Gathering waves
##### aBSS Lasso Priors
weakinfo.ls<-blasso.ls<-absslasso.ls<-horseshoe.ls<-list()
folder<-c('mod_Weak_info','mod_B_lasso','mod_aBSS_lasso','mod_Horseshoe')
prior_<-c('WeakInfor','Blasso','absslasso','RegHS')
M<-200
ptm<-proc.time()

for (t in c(30,40,50,60,70)){
    for(pr in 1:length(prior_)){
      postmean_m<-Abs_Bias_m<-Rel_Bias_m<-MSE_m<-CIR_m<-NDR_m<-ESS400_m<-
        Prec400_m<-ESS1000_m<-Prec1000_m<-ESS100_m<-Prec100_m<-Rhat_m<-CVr_m<-
        posterior_ls<-vector('list',200)
      for(m in 1:M){
        name_fl<-paste0("N",n,"_T",t,"_P",p,"_corr",corr_,"_var100",prior_[pr],'_prior_tvtp',"_rep",m)
        # True Theta
        theta.csv<-read_csv("Theta_tvtp.csv")
        # Posterior
        posterior.csv<-read_csv(paste0(folder[pr],"/",name_fl,".csv"))
        posterior.char <- as.matrix(posterior.csv)
        posterior <- as.matrix(posterior.csv[,-1])
        rownames(posterior)<-as.vector(posterior.char[,'...1'])
        # Metrics
        posterior_ls[[m]] <- cbind(posterior,theta.csv)
        # Convegence and sampling precision
        ESS100_m[[m]] <- ifelse(posterior_ls[[m]][,'n_eff']>100,1,0)
        Prec100_m[[m]] <- rep(ifelse(sum(ESS100_m[[m]])==length(ESS100_m[[m]]),1,0),
                              length(ESS100_m[[m]]))
        ESS400_m[[m]] <- ifelse(posterior_ls[[m]][,'n_eff']>400,1,0)
        Prec400_m[[m]] <- rep(ifelse(sum(ESS400_m[[m]])==length(ESS400_m[[m]]),1,0),
                              length(ESS400_m[[m]]))
        ESS1000_m[[m]] <- ifelse(posterior_ls[[m]][,'n_eff']>1000,1,0)
        Prec1000_m[[m]] <- rep(ifelse(sum(ESS1000_m[[m]])==length(ESS1000_m[[m]]),1,0),
                               length(ESS1000_m[[m]]))
        Rhat_m[[m]] <- ifelse(posterior_ls[[m]][,"Rhat"]<1.1,1,0)
        CVr_m[[m]] <- rep(ifelse(sum(Rhat_m[[m]])==length(Rhat_m[[m]]),1,0),
                          length(Rhat_m[[m]]))        
        if (unique(CVr_m[[m]]) == 1) {
          postmean_m[[m]] <- posterior_ls[[m]][,'mean']
          Abs_Bias_m[[m]] <- abs(posterior_ls[[m]][,'mean'] - posterior_ls[[m]][,'Theta'])
          Rel_Bias_m[[m]] <- posterior_ls[[m]][,'mean'] / posterior_ls[[m]][,'Theta']
          MSE_m[[m]] <- (posterior_ls[[m]][,'mean'] - posterior_ls[[m]][,'Theta'])^2
          CIR_m[[m]] <-ifelse((posterior_ls[[m]][,'Theta']>=posterior_ls[[m]][,'2.5%'] &
                                 posterior_ls[[m]][,'Theta']<=posterior_ls[[m]][,'97.5%']),1,0)
          NDR_m[[m]] <- ifelse((0>=posterior_ls[[m]][,'2.5%'] & 0<=posterior_ls[[m]][,'97.5%']),0,1)#switch
        }   
      }
      ########## Transoforming each list into
      postmean_X<-do.call(cbind,postmean_m)
      Abs_Bias_X<-do.call(cbind,Abs_Bias_m)
      Rel_Bias_X<-do.call(cbind,Rel_Bias_m)
      MSE_X<-do.call(cbind,MSE_m)
      CIR_X<-do.call(cbind,CIR_m)
      NDR_X<-do.call(cbind,NDR_m)
      ESS100_X<-do.call(cbind,ESS100_m)
      ESS400_X<-do.call(cbind,ESS400_m)
      ESS1000_X<-do.call(cbind,ESS1000_m)
      Prec100_X <-do.call(cbind,Prec100_m)
      Prec400_X <-do.call(cbind,Prec400_m)
      Prec1000_X<-do.call(cbind,Prec1000_m)
      Rhat_X<-do.call(cbind,Rhat_m)
      CVr_X<-do.call(cbind,CVr_m)
      ########## Computing all metrics
      All_theta <- dim(postmean_X)[1]
      postmean<-Abs_Bias<-Rel_Bias<-MSE<-CIR<-NDR<-ESS100<-ESS400<-ESS1000<-
        Prec100<-Prec400<-Prec1000<-CVr<-Rhat<- c()
      for (theta in 1:All_theta){
        # average posterior estimate
        postmean[theta]<- mean(postmean_X[theta,])
        # Absolute bias
        Abs_Bias[theta]<-mean(Abs_Bias_X[theta,])
        # Relative bias
        Rel_Bias[theta]<-mean(Rel_Bias_X[theta,])
        # MSE
        MSE[theta]<-mean(MSE_X[theta,])
        # Credible Interval Rate
        CIR[theta]<-mean(CIR_X[theta,])
        # Non-null Detection Rate
        NDR[theta]<-mean(NDR_X[theta,])
        # Effective Sample Size
        ESS100[theta]<-mean(ESS100_X[theta,])
        ESS400[theta]<-mean(ESS400_X[theta,])
        ESS1000[theta]<-mean(ESS1000_X[theta,])
        Prec100[theta]<-mean(Prec100_X[theta,])
        Prec400[theta]<-mean(Prec400_X[theta,])
        Prec1000[theta]<-mean(Prec1000_X[theta,])
        # Convergence rate - based Rhat
        Rhat[theta]<-mean(Rhat_X[theta,])
        CVr[theta]<-mean(CVr_X[theta,])
      }
      ########### Output results
      #rownames(posterior_ls[[1]])
      Table_res<- cbind(
        posterior_ls[[1]]['Theta'],
        postmean,
        Abs_Bias,
        Rel_Bias,
        MSE,
        CIR,
        NDR,
        ESS100,ESS400,ESS1000,
        Prec100,Prec400,Prec1000,
        Rhat,
        CVr
      )
      ########### Saving results as .csv file
      # table.char<- gsub("_rep\\d+", "", name_fl)
      # write.csv(Table_res,file=paste0('Tables/Table_',table.char,'.csv'))
      ########## Building list
      Table.ls <-list(Table_res)
      names(Table.ls)<-gsub("_rep\\d+", "", name_fl)
      switch(pr,
             weakinfo.ls <- append(weakinfo.ls, Table.ls),
             blasso.ls <- append(blasso.ls, Table.ls),
             absslasso.ls <- append(absslasso.ls, Table.ls),
             horseshoe.ls <- append(horseshoe.ls, Table.ls)
            )
    }
}  
proc.time() - ptm 

##### GIANT LIST OF DATA FRAME
models_conditions.ls<- append(weakinfo.ls, absslasso.ls) 
models_conditions.ls<- append(models_conditions.ls, horseshoe.ls) 
models_conditions.ls<- append(models_conditions.ls, blasso.ls)

##### Save the list to a file
saveRDS(models_conditions.ls, "tvtp_models_conditions.rds")

