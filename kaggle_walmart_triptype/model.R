# Data fields
# TripType - a categorical id representing the type of shopping trip the customer made. This is the ground truth that you are predicting. TripType_999 is an "other" category.
# VisitNumber - an id corresponding to a single trip by a single customer
# Weekday - the weekday of the trip
# Upc - the UPC number of the product purchased
# ScanCount - the number of the given item that was purchased. A negative value indicates a product return.
# DepartmentDescription - a high-level description of the item's department
# FinelineNumber - a more refined category for each of the products, created by Walmart


library(xgboost)
library(sqldf)
library(reshape2)
library(data.table)
library(plyr)
library(rpart)

library(doMC)
registerDoMC(cores = 4)


setwd("/Users/bikash/repos/kaggleCompetition1/kaggle_walmart_triptype")
# Set seed
set.seed(12345)


train <- read.table("data/train.csv",sep=',',header = T)
test <- read.table("data/test.csv",sep=',',header = T)

# Create unique trip types for result
outcomes <- data.frame(TripType = sort(unique(train$TripType)))
outcomes$Index <- seq_along(outcomes$TripType) - 1

####
data <-aggregate(TripType~ VisitNumber, train, max)
write.csv(data, file="train_result.csv", quote=FALSE,row.names=FALSE)


####
train <- read.table("data/train_result.csv",sep=',',header = T)
test <- read.table("data/test_result.csv",sep=',',header = T)

## load train result <visitnumer, triptype>
train_triptype <- read.table("data/train_label.csv",sep=',',header = T)


###
dtrain <- xgb.DMatrix(train, label = train_triptype)
dtest <- xgb.DMatrix(test)

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

bst <- xgb.train( param, dtrain, 1 )
# Note: we need the margin value instead of transformed prediction in set_base_margin
# do predict with output_margin=TRUE, will always give you margin values before logistic transformation
ptrain <- predict(bst, dtrain, outputmargin=TRUE)
ptest  <- predict(bst, dtest, outputmargin=TRUE)

