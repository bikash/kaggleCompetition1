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

library(sqldf)
library(reshape2)
library(data.table)
library(plyr)
library(rpart)


setwd("/Users/bikash/repos/kaggleCompetition1/kaggle_walmart_triptype")
# Set seed
set.seed(1765)


train <- read.table("data/train.csv",sep=',',header = T)
test <- read.table("data/test.csv",sep=',',header = T)


# Create unique trip types for result
outcomes <- data.frame(TripType = sort(unique(train$TripType)))
outcomes$Index <- seq_along(outcomes$TripType) - 1


# Convert NA values
# NA values are found in FinelineNumber and Upc. "NULL" string value is found in DepartmentDescription.
train$FinelineNumber <- addNA(train$FinelineNumber)
train$Upc <- addNA(train$Upc)
#train <- train[, ifelse (train$DepartmentDescription == "NULL", 1, 0)]


# Unique Dept Name
Dept <- sqldf('select distinct DepartmentDescription from train')
Dept$Dept_No <- (1: dim(Dept)[1])

train$ReturnCount <- -train$ScanCount
train$ReturnCount[train$ReturnCount < 0] <- 0

VisitData1 <- sqldf('select distinct visitNumber, TripType, Weekday, sum(scancount) as tot_item, 
                    sum(ReturnCount) as ret_item, count(scancount) as uniq_item
                    from train group by visitNumber, Weekday')

VisitDept <- sqldf('select visitNumber, DepartmentDescription, sum(scancount) as tot_item
                   from train group by visitNumber, TripType, DepartmentDescription')
VisitDept <- sqldf('select a.VisitNumber, a.tot_item, b.Dept_No from VisitDept a left join Dept b on a.DepartmentDescription = b.DepartmentDescription')

VisitData2 <- data.frame(acast(VisitDept,VisitNumber ~ Dept_No, fill = 0))

setDT(VisitData2, keep.rownames = TRUE)
rename(VisitData2, c("rn" = "VisitNumber"))
VisitDataF <- sqldf('select a.TripType, a.Weekday, a.tot_item, a.uniq_item, a.ret_item
                    , b.* from VisitData1 a join VisitData2 b on a.VisitNumber = b.VisitNumber ')
VisitDataF$Weekday <- as.integer(VisitDataF$Weekday)
# VisitDataF$tot_item <- log1p(VisitDataF$tot_item)
# VisitDataF$ret_item <- log1p(VisitDataF$ret_item)
# VisitDataF$uniq_item <- log1p(VisitDataF$uniq_item)


######test
# Unique Dept Name
# Dept <- sqldf('select distinct DepartmentDescription from test')
# Dept$Dept_No <- (1: dim(Dept)[1])
length(test$VisitNumber) #653646
test$TripType <- c(1:length(test$VisitNumber))

test$ReturnCount <- -test$ScanCount
test$ReturnCount[test$ReturnCount < 0] <- 0

tVisitData1 <- sqldf('select distinct visitNumber, Weekday, sum(scancount) as tot_item, sum(ReturnCount) as ret_item, 
                     count(scancount) as uniq_item
                     from test group by visitNumber, Weekday')

tVisitDept <- sqldf('select visitNumber, DepartmentDescription, sum(scancount) as tot_item
                    from test group by visitNumber, TripType, Weekday')

tVisitDept <- sqldf('select a.VisitNumber, a.tot_item, b.Dept_No from tVisitDept a 
                    left join Dept b on a.DepartmentDescription = b.DepartmentDescription')

tVisitData2 <- data.frame(acast(tVisitDept, VisitNumber ~ Dept_No, fill = 0))

setDT(tVisitData2, keep.rownames = TRUE)
rename(tVisitData2, c("rn" = "VisitNumber"))

tVisitDataF <- sqldf('select  a.Weekday, a.tot_item, a.uniq_item, a.ret_item
                     , b.* from tVisitData1 a join tVisitData2 b on a.VisitNumber = b.VisitNumber')
tVisitDataF$Weekday <- as.integer(tVisitDataF$Weekday)
# VisitDataF$tot_item <- log1p(VisitDataF$tot_item)
# VisitDataF$ret_item <- log1p(VisitDataF$ret_item)
# VisitDataF$uniq_item <- log1p(VisitDataF$uniq_item)




# Decision Tree
VisitDataF$TripTypeF <- factor(VisitDataF$TripType)
xnam <- paste0("X", 1:69)


##
### map trip type to unique number.
library(plyr)
label <- plyr::mapvalues(VisitDataF$TripType, from = outcomes$TripType, to = outcomes$Index)


VisitDataF$TripTypeF <- NULL
VisitDataF$TripType <- NULL










####
##
head(VisitDataF)
head(tVisitDataF)
train <- read.table("data/train_final.csv",sep=',',header = T)
test <- read.table("data/test_final.csv",sep=',',header = T)
train$uniq_item <- VisitDataF$uniq_item
train$tot_item <- VisitDataF$tot_item
test$uniq_item <- tVisitDataF$uniq_item
test$tot_item <- tVisitDataF$tot_item

###
train.matrix <- as.matrix(train)
train.matrix <- as(train.matrix, "dgCMatrix") # conversion to sparse matrix
dtrain <- xgb.DMatrix(train.matrix, label = label)

#test <- read.table("data/test_final.csv",sep=',',header = T)
test.matrix <- as.matrix(test)
test.matrix <- as(test.matrix, "dgCMatrix") # conversion to sparse matrix
dtest <- xgb.DMatrix(test.matrix)

cat("Training model - Xgboost\n")
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
bst <- xgb.train( param=param, data=dtrain, label=label, nrounds=nround,  verbose = 1)
# Note: we need the margin value instead of transformed prediction in set_base_margin
# do predict with output_margin=TRUE, will always give you margin values before logistic transformation
#save(bst,file="xgboost.Rda")
#load("xgboost.Rda")
# Get the feature real names
#names <- dimnames(dtrain)[[2]]
# Compute feature importance matrix
#importance_matrix <- xgb.importance(names, model = bst)
# Nice graph
#xgb.plot.importance(importance_matrix[1:40,])


ptest  <- predict(bst, dtest)
head(ptest)



###RAndom Forest

cat("Training model - RF\n")
set.seed(8)
rf <- randomForest(dtrain, y, ntree=100, imp=TRUE, sampsize=10000, do.trace=TRUE)
#predict_rf1 <- predict(rf, mtrain)
predict_rf2 <- predict(rf, mtest)










# Decode prediction
ptest <- matrix(ptest, nrow=38, ncol=length(ptest) / 38)
pred <- t(ptest)

# output
print("Storing Output")
submit <- function(filename) {
  #pred = format(pred, digits=2,scientific=F) # shrink the size of submission
  pred1 <- data.frame(cbind(tVisitDataF$VisitNumber, pred))
  names(pred1) <- c("VisitNumber", paste("TripType", outcomes$TripType, sep = "_")) 
  
  write.table(format(pred1, scientific = FALSE), paste("output/", filename, sep = ""), row.names = FALSE, sep = ",")
}

submit("xgboost2.csv")







