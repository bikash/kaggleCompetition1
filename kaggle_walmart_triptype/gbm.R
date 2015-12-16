library(gbm)
library(doParallel)
library(glmnet)
library(useful)
library(dplyr)


setwd("/Users/bikash/repos/kaggleCompetition1/kaggle_walmart_triptype")

#read in data

train <- read.csv('data/train.csv', colClasses = c('factor'))
test <- read.csv('data/test.csv', colClasses = c('factor'))

#### GBM ####

sampleIndex <- unique(train$VisitNumber)
sampleIndex.subsetTrain  <- sample_frac(data.frame(sampleIndex), .05, replace = F)
sampleIndex <- data.frame(sampleIndex)
sampleIndex$keep <- sampleIndex$sampleIndex %in% sampleIndex.subsetTrain$sampleIndex 
sampleIndex <- dplyr::filter(sampleIndex, keep == F)

sampleIndex$keep <- NULL

sampleIndex.subsetTest <- sample_frac(sampleIndex, .05, replace = F)

train.subset <-inner_join(train, sampleIndex.subsetTrain, by = c("VisitNumber" = "sampleIndex"))

test.subset <-inner_join(train, sampleIndex.subsetTest, by = c("VisitNumber" = "sampleIndex"))

yTrain <- train.subset$TripType

yTest <- test.subset$TripType

trainX.Mat <- useful::build.x(TripType ~ Weekday + DepartmentDescription + FinelineNumber, data = train.subset)

testX.Mat <- useful::build.x(TripType ~ Weekday + DepartmentDescription + FinelineNumber, data = test.subset)

trainY.Mat <- as.numeric(useful::build.y(formula = TripType ~ Weekday + DepartmentDescription + FinelineNumber, 
                                         data = train.subset))

testY.Mat <- as.numeric(useful::build.y(formula = TripType ~ Weekday + DepartmentDescription + FinelineNumber, 
                                        data = test.subset))


# xTrain <- select(train.subset, -TripType, -VisitNumber, -Upc, -FinelineNumber)
# 
# xTest <- select(test, -VisitNumber, -Upc, -FinelineNumber)
# 

fittedGBM <- gbm(x = trainX.Mat, y = trainY.Mat, 
                     distribution = 'multinomial', 
                     n.trees = 200, shrinkage = .03, n.minobsinnode = 10, 
                     nTrain = (nrow(trainX.Mat)*.8),
                     interaction.depth = 2, verbose = T)

fittedGBM.df <- data.frame(summary(fittedGBM))

gbm.perf(fittedGBM)

for (i in 1:ncol(trainX.Mat)){
  plot.gbm(fittedGBM, i.var = i, n.trees = gbm.perf(fittedGBM), type = 'response')
  
}

yhat <- predict(fittedGBM, newdata = xTest, type = 'response', n.trees = gbm.perf(fittedGBM, plot.it = F))

