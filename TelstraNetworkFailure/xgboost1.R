
## xgboost

library(xgboost)

set.seed(1718)
verboseXgboost <- T
genererSubmission <- T
notifyAndroid <- F
CVonly <- F
importance <- F
onFold <- F


if(onFold){
  writeLines(paste("Starting xgboost on a fold1....",nrow(xtrain[folds$Resample1,]), " lines" ))
  dtrain <- xgb.DMatrix(data = xtrain[folds$Resample1,], label = ytrain[folds$Resample1])
  xtest <- xtrain[-folds$Resample1,]
  test.id <- test.id[-folds$Resample1]
}else{
  writeLines("Starting xgboost on full train...")
  dtrain <- xgb.DMatrix(data = xtrain, label = ytrain)
}

registerDoMC(cores = 4)

xgparams.tree <- list(
  objective = "multi:softprob",
  num_class = 3,
  colsample_bytree = 0.3,
  max.depth = 8,
  eta = 0.05
)

xgboost.first <- xgb.cv(
  data = dtrain,
  params = xgparams.tree,
  nrounds = 5,
  nfold = 10,
  metrics = "mlogloss",
  verbose = verboseXgboost,
  print.every.n = 200
)

cat("xVal mlogloss : ", min(xgboost.first$test.mlogloss.mean),"\n")

if(!CVonly)
{
  pred.loop <- matrix(nrow = nrow(xtest)*3, ncol = 10)
  for(index in 1:10)
  {
    set.seed(28021980+index)
    xgboost.model <- xgboost(
      data = dtrain,
      params = xgparams.tree,
      nrounds = which.min(xgboost.first$test.mlogloss.mean),
      verbose = verboseXgboost
    )
    pred.loop[,index] <- xgboost::predict(xgboost.model, xtest)
  }
  
  pred.xgboost <- matrix(apply(pred.loop, MARGIN = 1, mean), ncol = 3, byrow = T)
  
  
  if(importance){
    writeLines("Computing importance...")
    imp <- xgb.importance(feature_names = names(xtrain), model = xgboost.model)
  }
  
  if (genererSubmission) {
    cat("Generating Submission...\n")
    output.xgboost <- data.frame(
      id = test.id,
      predict_0 = pred.xgboost[,1],
      predict_1 = pred.xgboost[,2],
      predict_2 = pred.xgboost[,3]
    )
    write.csv(output.xgboost, paste(sep = "-", format(Sys.time(), "%Y%m%d.%H%M"), "xgb.csv"), row.names = F, quote = F)
  }
}


msg = paste("Minimum xVal mlogloss : ", min(xgboost.first$test.mlogloss.mean))
print (msg)


