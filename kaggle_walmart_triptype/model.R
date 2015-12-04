# Data fields
# TripType - a categorical id representing the type of shopping trip the customer made. This is the ground truth that you are predicting. TripType_999 is an "other" category.
# VisitNumber - an id corresponding to a single trip by a single customer
# Weekday - the weekday of the trip
# Upc - the UPC number of the product purchased
# ScanCount - the number of the given item that was purchased. A negative value indicates a product return.
# DepartmentDescription - a high-level description of the item's department
# FinelineNumber - a more refined category for each of the products, created by Walmart


require(xgboost)
require(Matrix)
library(doMC)
registerDoMC(cores = 4)


setwd("/Users/bikash/repos/kaggleCompetition1/kaggle_walmart_triptype")
# Set seed
set.seed(12345)


train.data <- read.table("data/train.csv",sep=',',header = T)
test <- read.table("data/test.csv",sep=',',header = T)

# Create unique trip types for result
outcomes <- data.frame(TripType = sort(unique(train.data$TripType)))
outcomes$Index <- seq_along(outcomes$TripType) - 1

####
#data <-aggregate(TripType~ VisitNumber, train, max)
#write.csv(data, file="train_result.csv", quote=FALSE,row.names=FALSE)


####
train <- read.table("data/train_final.csv",sep=',',header = T)
test <- read.table("data/test_final.csv",sep=',',header = T)

## load train result <visitnumer, triptype>
train_triptype <- read.table("train_result.csv",sep=',',header = T)

### map trip type to unique number.
library(plyr)
label <- plyr::mapvalues(train_triptype$TripType, from = outcomes$TripType, to = outcomes$Index)


###
train.matrix <- as.matrix(train)
train.matrix <- as(train.matrix, "dgCMatrix") # conversion to sparse matrix
dtrain <- xgb.DMatrix(train.matrix, label = label)

test.matrix <- as.matrix(test)
test.matrix <- as(test.matrix, "dgCMatrix") # conversion to sparse matrix
dtest <- xgb.DMatrix(test.matrix)

# xgboost parameters
param <- list("objective" = "multi:softprob",    # multiclass classification 
              "num_class" = 38,    # number of classes 
              "eval_metric" = "mlogloss",    # evaluation metric 
              "nthread" = 4,   # number of threads to be used 
              "silent" =1,
              "max_depth" = 5,    # maximum depth of tree 
              "chi2_lim" = 0,
              "eta" = 0.1,    # step size shrinkage 
              "gamma" = 0,    # minimum loss reduction 
              "subsample" = 0.7,    # part of data instances to grow tree 
              "colsample_bytree" = 1,  # subsample ratio of columns when constructing each tree 
              "min_child_weight" = 12  # minimum sum of instance weight needed in a child 
)

nround = 400
bst <- xgb.train( param=param, data=dtrain, label=label, nrounds=nround,  verbose  = 1)
# Note: we need the margin value instead of transformed prediction in set_base_margin
# do predict with output_margin=TRUE, will always give you margin values before logistic transformation


# Get the feature real names
#names <- dimnames(dtrain)[[2]]
# Compute feature importance matrix
#importance_matrix <- xgb.importance(names, model = bst)
# Nice graph
xgb.plot.importance(importance_matrix[1:10,])


ptest  <- predict(bst, dtest)
head(ptest)


