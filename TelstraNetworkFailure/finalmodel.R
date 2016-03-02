head(pred.xgboost)
head(pred.rf)
setwd("/Users/bikash/repos/kaggleCompetition1/TelstraNetworkFailure")

source('utils.R') # utility functions
source('base.R') # read files and feature engineering
source('xgboost1.R') # provides pred.xgboost
source('rf.R') # proivides pred.rf

# on construit un model blend sur fold3


combinaisons <- expand.grid(
  alpha0 = seq(0,1,by=0.1),
  alpha1 = seq(0,1,by=0.1),
  alpha2 = seq(0,1,by=0.1),
  beta0 = seq(0,1,by=0.1),
  beta1 = seq(0,1,by=0.1),
  beta2 = seq(0,1,by=0.1)
)
combinaisons$s0 <- combinaisons$alpha0 + combinaisons$beta0
combinaisons$s1 <- combinaisons$alpha1 + combinaisons$beta1
combinaisons$s2 <- combinaisons$alpha2 + combinaisons$beta2

combs <- combinaisons[which(combinaisons$s0 == 1 & combinaisons$s1 == 1 & combinaisons$s2 == 1),]
head(combs)

computeLogLoss <- function(row){
  alpha0 <- row[1]
  alpha1 <- row[2]
  alpha2 <- row[3]
  beta0 <- row[4]
  beta1 <- row[5]
  beta2 <- row[6]
  output.ens <- data.frame(
    #id = test$id,
    X0 = alpha0*pred.xgboost[,1]+ beta0*pred.rf[, 1],
    X1 = alpha1*pred.xgboost[,2]+ beta1*pred.rf[, 2],
    X2 = alpha2*pred.xgboost[,3]+ beta2*pred.rf[, 3],
    obs = paste0("X",ytrain[-folds$Resample1])
  )
  return(mnLogLoss(output.ens, lev = levels(output.ens$obs)))
  
}

loglosses <- base::apply(combs, 1, FUN = computeLogLoss)
bestComb <- combs[which.min(loglosses),]