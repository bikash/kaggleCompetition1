# svm
library(caret)

setwd("/Users/bikash/repos/kaggleCompetition1/TelstraNetworkFailure")
#source('base.R')
#source('utils.R')


genererSubmission <- F

if(onFold){
  writeLines(paste("Starting randomForest on a fold with....",nrow(xtrain[folds$Resample1,]), " lines" ))
  x <- xtrain[folds$Resample1,]
  y <- as.factor(paste0("X",ytrain[folds$Resample1]))
  xtest <- xtrain[-folds$Resample1,]
  test.id <- test.id[-folds$Resample1]
}else{
  writeLines("Starting randomForest on full training set...")
  x <- xtrain
  y <- as.factor(paste0("X",ytrain))
}

tg <- expand.grid(mtry = 80)

tr <- trainControl(
  method = "cv", 
  number = 10, 
  classProbs = TRUE, 
  summaryFunction = mc_logloss,
  allowParallel = T)

rf_model <- caret::train(x, y, 
                          method = "rf", 
                          metric = "logLoss",
                          maximize = F,
                          trControl = tr,
                          tuneGrid = tg,
                          ntree = 500,
                          sampsize = c(1000,1000,600))

  msg = paste("Minimal CV mlogloss : ", min(rf_model$results$logLoss))
  print (msg)

pred.rf <- predict(rf_model, xtest, type="prob")

if (genererSubmission) {
  output.rf <- data.frame(
    id = test.id,
    predict_0 = pred.rf[,1],
    predict_1 = pred.rf[,2],
    predict_2 = pred.rf[,3]
  )
  write.csv(output.rf, paste(sep = "-", format(Sys.time(), "%Y%m%d.%H%M"), "output/rf-submission.csv"), row.names = F, quote = F)
}