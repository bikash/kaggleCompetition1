# Data fields
# TripType - a categorical id representing the type of shopping trip the customer made. This is the ground truth that you are predicting. TripType_999 is an "other" category.
# VisitNumber - an id corresponding to a single trip by a single customer
# Weekday - the weekday of the trip
# Upc - the UPC number of the product purchased
# ScanCount - the number of the given item that was purchased. A negative value indicates a product return.
# DepartmentDescription - a high-level description of the item's department
# FinelineNumber - a more refined category for each of the products, created by Walmart

library(readr)
library(xgboost)

require(caret)
require(plyr)
require(Metrics)
require(ROCR)
library(doMC)
registerDoMC(cores = 4)

setwd("/Users/bikash/repos/kaggleCompetition1/kaggle_walmart_triptype")
# Set seed
set.seed(12345)


train <- read.table("data/train.csv",sep=',',header = T)
test <- read.table("data/test.csv",sep=',',header = T)

## 647054 obs. of  7 variables:
##TripType VisitNumber Weekday Upc ScanCount DepartmentDescription FinelineNumber


## convert triptype to 1 to 38 number
numclass =length(unique(train$TripType)) #38
y.list <- sort(unique(train$TripType)) #[1]   3   4   5   6   7   8   9  12  14  15  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35  36  37  38  39  40  41  42  43  44 999
train$TripType1 <- NULL
for(i in 1:numclass)
{
  train$TripType1[train$TripType == y.list[i]] <- i
}
############################################


# Create outcomes for xgboost
library(data.table)
outcomes <- data.table(TripType = sort(unique(train$TripType)))
outcomes$Index <- seq_along(outcomes$TripType) - 1

# Combine train and test
data <- data.table(rbind(train, cbind(TripType = -1, test))) ## test data has trip type = -1



# Save the name of the last column target
# nameLastCol <- names(train)[1] ## TripType
# y1 = train[,ncol(train)]
# y1 = gsub('TripType_','',y1)
# y = as.integer(y)-1 #xgboost take features in [0,numOfClass)

## remove UPC number
train$Upc <- NULL
test$Upc <- NULL
train$Weekday <- NULL
test$Weekday <- NULL

#y <- plyr::mapvalues(train$TripType, from = outcomes$TripType, to = outcomes$Index)

y <- train$TripType1
num.class <- length(unique(y))+1
train$TripType<-NULL
train$TripType1<-NULL

trainlength = nrow(train)


x = rbind(train, test)
x[is.na(x)] <- -1
dmy = dummyVars(" ~ . ", x)
x <- data.frame(predict(dmy, newdata = x))

x = as.matrix(x)

train1 = x[1:trainlength,]
test1  = x[(nrow(train)+1):nrow(x),]
headers = colnames(train)



print("Training the model")
# xgboost parameters
param <- list("objective" = "multi:softprob",    # multiclass classification 
              "num_class" = num.class,    # number of classes 
              "eval_metric" = "mlogloss",    # evaluation metric 
              "nthread" = 8,   # number of threads to be used 
              "max_depth" = 16,    # maximum depth of tree 
              "eta" = 0.3,    # step size shrinkage 
              "gamma" = 0,    # minimum loss reduction 
              "subsample" = 1,    # part of data instances to grow tree 
              "colsample_bytree" = 1,  # subsample ratio of columns when constructing each tree 
              "min_child_weight" = 12  # minimum sum of instance weight needed in a child 
)

# param <- list("objective" = "multi:softprob",
#               "eval_metric" = 'mlogloss', 
#               "num_class" = 39, 
#               "nthread" = 16,
#               "bst:eta" = .01,
#               "bst:max_depth" = 30,
#               "lambda" = 1,
#               "lambda_bias" = 0,
#               "gamma" = 1,
#               "alpha" = .8,
#               "min_child_weight" = 3,
#               "subsample" = .9,
#               "colsample_bytree" = .9)

cv.nround <- 50 # 200
cv.nfold <- 3 # 10

bst.cv <- xgb.cv(param=param, data = train1, label = y, 
                 nfold = cv.nfold, nrounds = cv.nround, prediction = TRUE, verbose = 1)

tail(bst.cv$dt)

nround = 400
bst = xgboost(param=param, data = train1, label = y, nrounds=nround, verbose = 1)

#print("Plotting Importance")
#importance_matrix <- xgb.importance(headers, model = bst)
#importance_matrix <- subset(importance_matrix, Gain > 0.01)
#xgb.plot.importance(importance_matrix)

pred = predict(bst, test1)
summary(pred)
pred = matrix(pred,1,length(pred))

pred = t(pred)

print("Storing Output")
pred = format(pred, digits=2,scientific=F) # shrink the size of submission
pred = data.frame(1:nrow(pred), pred)
names(pred) = c('VisitNumber', 'TripType')
write.csv(pred, file="Output/walmart-xgboost.csv", quote=FALSE,row.names=FALSE)

