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
set.seed(1765)


train <- read.table("data/train.csv",sep=',',header = T)
test <- read.table("data/test.csv",sep=',',header = T)


# Create unique trip types for result
outcomes <- data.frame(TripType = sort(unique(train$TripType)))
outcomes$Index <- seq_along(outcomes$TripType) - 1

# Combine train and test
dt <- rbind(train.data, cbind(TripType = -1, test))


# Convert NA values
# NA values are found in FinelineNumber and Upc. "NULL" string value is found in DepartmentDescription.
dt$FinelineNumber <- addNA(dt$FinelineNumber)
dt$Upc <- addNA(dt$Upc)
dt <- dt[, NullDescription:=ifelse (dt$DepartmentDescription == "NULL", 1, 0)]


# Feature engineering 
# Include ReturnCount column
dt$ReturnCount <- -dt$ScanCount
dt$ReturnCount[dt$ReturnCount < 0] <- 0
dt$ScanCount[dt$ScanCount < 0] <- 0
dt$ResultCount <- dt$ScanCount - dt$ReturnCount

# Calculate Scan and Return counts by VisitNumber
library(dplyr)
item.counts <- summarise(group_by(dt, VisitNumber),
                         TotalScan = sum(ScanCount), TotalReturn = sum(ReturnCount), TotalResult = sum(ResultCount))
###########################################
#   TotalScan TotalReturn TotalResult
# 1   1476396       34178     1442218
###########################################

### Ignore UPC and FinelineNumber
dt$Upc <- NULL
dt$FinelineNumber <- NULL


train.len <- length(train$VisitNumber)
total.len <- length(dt$VisitNumber)
train1 <- dt[1:train.len,]
test1 <- dt[-c(1:train.len),]


## convert categorial variable to dummary variable
dummies <- dummyVars(~ ., data = train1)
train1 = predict(dummies, newdata = train1)
test1 = predict(dummies, newdata = test1)


data <- dcast(data = dt,
                  VisitNumber + TripType + Weekday ~ DepartmentDescription + ItemCount,
                  value.var = "value",
                  fun.aggregate = sum)

library("plyr")
data <- aggregate( ScanCount ~ VisitNumber + TripType   , data = dt, sum)

## load train result <visitnumer, triptype>
train_triptype <- read.table("data/train_result.csv",sep=',',header = T)

### map trip type to unique number.
library(plyr)
label <- plyr::mapvalues(train_triptype$TripType, from = outcomes$TripType, to = outcomes$Index)



###
train.matrix <- as.matrix(train1)
train.matrix <- as(train.matrix, "dgCMatrix") # conversion to sparse matrix
dtrain <- xgb.DMatrix(train.matrix, label = label)

#test <- read.table("data/test_final.csv",sep=',',header = T)
test.matrix <- as.matrix(test1)
test.matrix <- as(test.matrix, "dgCMatrix") # conversion to sparse matrix
dtest <- xgb.DMatrix(test.matrix)

# xgboost parameters
param <- list("objective" = "multi:softprob",    # multiclass classification 
              "num_class" = 38,    # number of classes 
              "eval_metric" = "mlogloss",    # evaluation metric 
              "nthread" = 10,   # number of threads to be used 
              "silent" =1,
              "max_depth" = 9,    # maximum depth of tree 
              "chi2_lim" = 0,
              "eta" = 0.03,    # step size shrinkage 
              "gamma" = 0,    # minimum loss reduction 
              "subsample" = 0.5,    # part of data instances to grow tree 
              "colsample_bytree" = 1,  # subsample ratio of columns when constructing each tree 
              "min_child_weight" = 12  # minimum sum of instance weight needed in a child 
)

nround = 400
bst <- xgb.train( param=param, data=dtrain, label=label, nrounds=nround,  verbose  = 1)
# Note: we need the margin value instead of transformed prediction in set_base_margin
# do predict with output_margin=TRUE, will always give you margin values before logistic transformation
#save(bst,file="xgboost.Rda")
#load("xgboost.Rda")
# Get the feature real names
#names <- dimnames(dtrain)[[2]]
# Compute feature importance matrix
#importance_matrix <- xgb.importance(names, model = bst)
# Nice graph
#xgb.plot.importance(importance_matrix[1:10,])


ptest  <- predict(bst, dtest)
head(ptest)

# Decode prediction
ptest <- matrix(ptest, nrow=38, ncol=length(ptest) / 38)
pred <- t(ptest)

# output
submit <- function(filename) {
  pred <- data.frame(cbind(test$VisitNumber, pred))
  names(pred) <- c("VisitNumber", paste("TripType", outcomes$TripType, sep = "_")) 
  
  write.table(format(pred, scientific = FALSE), paste("output/", filename, sep = ""), row.names = FALSE, sep = ",")
}
submit("xgboost2.csv")




