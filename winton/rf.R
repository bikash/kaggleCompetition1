
setwd("/Users/bikash/repos/kaggleCompetition1/winton")
library(ggplot2)
library(dplyr)
library(caret)

# In all scripts, training set is called 'train' and test set it called 'test'
# The point of these if blocks are to avoid having to read the csv files repeatedly since that
# is somewhat time consuming, the var 'current.train' basically just tracks what data set
# is represented currently by 'train'

train <- read.csv("data/train.csv")
test <- read.csv("data/test.csv")



# The predictors are:
#     Feature_1 - Feature_25
#     Ret_MinusTwo, Ret_MinusOne, Ret_2-Ret_120: The first 2 are previous daily/overnight returns
#           and 2-120 are minute-to-minute returns from the current day
#     Weight_Intraday and Weight_Daily are weight used to evaluate intraday return predictions Ret 121 to 180
#           and weight used to evaluate daily return predictions, respectively
#
# The dependent variables are:
#     Ret_121 to Ret_180
#     Ret_PlusOne, Ret_PlusTwo this is the return from the time Ret_180 is measured on 
#           day D (D+1) to the close of trading on day D+1 (D+2)

# For purposes of speed and for estimate of out-of-sample error, break training set into
# train and validation sets

inTrain <- createDataPartition(y = train$Ret_121, p = .1, list = FALSE)
training <- train[inTrain,]
validation <- train[-inTrain,]

predictors <- training[,2:147]
outcomes <- training[,148:209]

# randomForest
start <- proc.time()[3]
mtry_def <- floor(sqrt(ncol(training)))
t_grid <- expand.grid(mtry= c(mtry_def))

model.rf <- train(x = predictors, y = outcomes[,1],
                  method = "rf",
                  ntree = 50)
end <- proc.time()[3]
duration <- end - start
print(paste("This took", duration, "seconds", sep = " "))

rf.predict <- predict(model.rf, validation[,2:ncol(train.test)])
acc <- sum(rf.predict == train.test[,1])/nrow(train.test)