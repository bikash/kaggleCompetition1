## xgboost classification


library(dplyr)
library(xgboost)
library(Matrix)
library(ggplot2)


setwd("/Users/bikash/repos/kaggleCompetition1/TelstraNetworkFailure")
set.seed(12345)
event_type <- read.table("data/event_type.csv",sep=',',header = T)
##id    event_type
log <- read.table("data/log_feature.csv",sep=',',header = T)
##id log_feature volume
severity<- read.table("data/severity_type.csv",sep=',',header = T)
##id   severity_type
train <- read.table("data/train.csv",sep=',',header = T)
## id     location fault_severity

test <- read.table("data/test.csv",sep=',',header = T)

resource_type <- read.table("data/resource_type.csv",sep=',',header = T)
##id   resource_type

##join table
a <- merge(train, event_type, by=c('id'))
b <- merge(a, severity, by=c('id'))
c <- merge(b, log, by=c('id'))
train.data <- merge(c, resource_type, by=c('id'))

a <- merge(test, event_type, by=c('id'))
b <- merge(a, severity, by=c('id'))
c <- merge(b, log, by=c('id'))
test.data <- merge(c, resource_type, by=c('id'))
##  id     location fault_severity    event_type   severity_type log_feature volume   resource_type

## convert categorical varaible to integer 
# train.data$event_type <- as.integer(train.data$event_type)
# train.data$log_feature <- as.integer(train.data$log_feature)
# train.data$resource_type <- as.integer(train.data$resource_type)
# train.data$severity_type <- as.integer(train.data$severity_type)
# train.data$location <- as.integer(train.data$location)

#-----------------XG BOOST -------------------------------------------------#
# Create the predictor data set and encode categorical variables using caret library.
fault_severity = train.data$fault_severity
train.data$fault_severity <- NULL


dummies <- dummyVars(~ ., data = train.data)
train1 = predict(dummies, newdata = train.data)
#dummies <- dummyVars(~ ., data = test.data)
test1 = predict(dummies, newdata = test.data)

library(sqldf)
tbl <-data.frame(train1)
train2 <- sqldf('SELECT * FROM tbl GROUP BY id')

tbl <-data.frame(test1)
test2 <- sqldf('SELECT * FROM tbl GROUP BY id')


train.matrix <- as.matrix(train2)
train.matrix <- as(train.matrix, "dgCMatrix") # conversion to sparse matrix
dtrain <- xgb.DMatrix(data = train.matrix, label = fault_severity)

num.class <- length(unique(fault_severity))

print("Training the model")
param <- list("objective" = "multi:softprob",
              "num_class" = num.class,    # number of classes 
              "eval_metric" = "mlogloss",    # evaluation metric 
              "nthread" = 16,
              "eta" = .01,
              "max_depth" = 10,
              "lambda_bias" = 0,
              "gamma" = .8,
              "min_child_weight" = 3,
              "subsample" = .9,
              "colsample_bytree" = .45,
              "scale_pos_weight" = sum(y==0) / sum(y==1))


set.seed(4568)
cv.nround <- 700 # 200
cv.nfold <- 10 # 10
bst.cv <- xgb.cv(param=param, data=dtrain, 
                 nfold=cv.nfold, nrounds=cv.nround, prediction=TRUE) 
tail(bst.cv$dt)


# Index of minimum merror
min.error.index = which.min(bst.cv$dt[, test.mlogloss.mean]) 
min.error.index 

# Minimum error
bst.cv$dt[min.error.index, ]

## Model
nround = min.error.index # number of trees generated
bst <- xgboost(param = param, data = dtrain, nrounds = nround, verbose = TRUE)

#model <- xgb.dump(bst, with.stats = T)
#model[1:10]




print("Making prediction")

#test$FinelineNumber <- addNA(test$FinelineNumber)
test.matrix <- as.matrix(test2)
#test.matrix <- as(test.matrix, "dgCMatrix") # conversion to sparse matrix
## prediction
pred <- predict(bst, test.matrix)

# Decode prediction
pred1 <- matrix(pred, nrow=num.class, ncol=length(pred) / num.class)
pred1 <- data.frame(cbind(test$id, t(pred1)))


print("Storing Output")
# output
submit <- function(filename) {
  names(pred1) <- c("id", "predict_0", "predict_1", "predict_2") 
  write.table(format(pred1, scientific = FALSE), paste("output/", filename, sep = ""), row.names = FALSE, sep = ",")
}
submit("xgboost.csv")




