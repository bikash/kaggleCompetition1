## xgboost classification

# names(train)
# [1] "TripType"              "VisitNumber"           "Weekday"              
# [4] "Upc"                   "ScanCount"             "DepartmentDescription"
# [7] "FinelineNumber"

library(dplyr)
library(magrittr)
library(reshape2)
library(xgboost)
library(Matrix)
library(Ckmeans.1d.dp)
library(DiagrammeR)
library(caret)
library(corrplot)
library(Rtsne)
library(stats)
library(ggplot2)
library(e1071)

setwd("/Users/bikash/repos/kaggleCompetition1/kaggle_walmart_triptype")
# Set seed
set.seed(12345)

time.start <- Sys.time()

# Read files
train.data <- read.csv("data/train.csv", header = TRUE)
test.data <- read.csv("data/test.csv", header = TRUE)



# Create outcomes for xgboost
outcomes <- data.frame(TripType = sort(unique(train.data$TripType)))
outcomes$Index <- seq_along(outcomes$TripType) - 1

# Combine train and test
dt <- rbind(train.data, cbind(TripType = -1, test.data))

## Preprocessing

# Convert NA values
# NA values are found in FinelineNumber and Upc. "NULL" string value is found in DepartmentDescription.

# Feature engineering 
# Include ReturnCount column
dt$ReturnCount <- -dt$ScanCount
dt$ReturnCount[dt$ReturnCount < 0] <- 0
dt$ScanCount[dt$ScanCount < 0] <- 0

dt$Weekday <- as.numeric(dt$Weekday)
dt$FinelineNumber <- as.factor(dt$FinelineNumber)

item.counts <- summarise(group_by(dt, VisitNumber), TotalScan = sum(ScanCount), TotalReturn = sum(ReturnCount))

# Convert dt data.frame from long to wide format using dcast from reshape2 package
# We want to aggregate on columns "TripType", "VisitNumber" and "Weekday" 
dt.long <- melt(data = data, measure.vars = c("ScanCount", "ReturnCount"))
dt.long <- rename(dt.long, ItemCount = variable)

dt.wide1 <- dcast(data = dt.long,
                  VisitNumber + TripType + Weekday ~ DepartmentDescription + ItemCount,
                  value.var = "value",
                  fun.aggregate = sum)

dt.wide2 <- dcast(data = dt.long,
                  VisitNumber ~ FinelineNumber,
                  value.var = "ScanCount",
                  fun.aggregate = sum)

dt.wide <- merge(dt.wide1, dt.wide2, by = "VisitNumber")
dt.wide <- merge(dt.wide, item.counts, by = "VisitNumber")

# Split train and test 
train <- dt.wide[dt.wide$TripType != -1, ]
test <- dt.wide[dt.wide$TripType == -1, ]

train$VisitNumber <- NULL
test.VisitNumber <- test$VisitNumber
test$VisitNumber <- NULL

# check for zero variances
zero.var = nearZeroVar(train, saveMetrics=TRUE)
zero.var
zero.var[zero.var$zeroVar == TRUE, ]
zero.var[zero.var$nzv == FALSE, ]

# # experimental - get rid of nzv features - resulted in worse performance
# cols <- row.names(zero.var[zero.var$nzv == TRUE, ]) # columns to discard
# colNums <- match(cols, names(train))
# train <- select(train, -colNums)
# test <- select(test, -colNums)

# # correlation matrix
# corrplot.mixed(cor(train), lower="circle", upper="color", 
#                tl.pos="lt", diag="n", order="hclust", hclust.method="complete")

# ## tsne plot
# # t-Distributed Stochastic Neighbor Embedding
# tsne = Rtsne(as.matrix(train), check_duplicates=FALSE, pca=TRUE, 
#              perplexity=30, theta=0.5, dims=2)
# 
# embedding = as.data.frame(tsne$Y)
# embedding$Class = outcome.org
# 
# g = ggplot(embedding, aes(x=V1, y=V2, color=Class)) +
#   geom_point(size=1.25) +
#   guides(colour=guide_legend(override.aes=list(size=6))) +
#   xlab("") + ylab("") +
#   ggtitle("t-SNE 2D Embedding of 'Classe' Outcome") +
#   theme_light(base_size=20) +
#   theme(axis.text.x=element_blank(),
#         axis.text.y=element_blank())
# 
# print(g)

# train.wide2 <- dcast(data = train.long, VisitNumber ~ FinelineNumber)
# train.wide3 <- dcast(data = train.long, VisitNumber ~ Upc)
# 
# train.wide <- merge(train.wide1, train.wide2, by = "VisitNumber")
# train.wide <- merge(train.wide, train.wide3, by = "VisitNumber")

## xgboost
y <- plyr::mapvalues(train$TripType, from = outcomes$TripType, to = outcomes$Index)

train$TripType <- NULL
test$TripType <- NULL

num.class <- length(unique(y))

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

train.matrix <- as.matrix(train)
train.matrix <- as(train.matrix, "dgCMatrix") # conversion to sparse matrix
dtrain <- xgb.DMatrix(data = train.matrix, label = y)

## cv
set.seed(456)

cv.nround <- 50 # 200
cv.nfold <- 3 # 10

bst.cv <- xgb.cv(param=param, data = train.matrix, label = y, 
                 nfold = cv.nfold, nrounds = cv.nround, prediction = TRUE)

tail(bst.cv$dt)

# Index of minimum merror
min.error.index = which.min(bst.cv$dt[, test.mlogloss.mean]) 
min.error.index 

# Minimum error
bst.cv$dt[min.error.index, ]

## Confusion matrix - needs checking
# cv prediction decoding
pred.cv = matrix(bst.cv$pred, nrow=length(bst.cv$pred)/num.class, ncol=num.class)
pred.cv = max.col(pred.cv, "last")

# Confusion matrix
confusionMatrix(factor(pred.cv), factor(y + 1))

## Model
nround = min.error.index # number of trees generated
bst <- xgboost(param = param, data = train1, label = y, nrounds = nround, verbose = TRUE)

model <- xgb.dump(bst, with.stats = T)
model[1:10]

# Get the feature real names
names <- dimnames(train1)[[2]]

# Compute feature importance matrix
importance_matrix <- xgb.importance(names, model = bst)

# Nice graph
xgb.plot.importance(importance_matrix[1:20,])

# Tree plot - not working
xgb.plot.tree(feature_names = names, model = bst, n_first_tree = 2)

## Prediction
pred <- predict(bst, test1)

# Decode prediction
pred <- matrix(pred, nrow=38, ncol=length(pred) / 38)
pred <- t(pred)

# output
submit <- function(filename) {
  pred <- data.frame(cbind(test$VisitNumber, pred))
  names(pred) <- c("VisitNumber", paste("TripType", outcomes$TripType, sep = "_")) 
  
  write.table(format(pred, scientific = FALSE), paste("output/", filename, sep = ""), row.names = FALSE, sep = ",")
}
submit("xgboost.csv")

time.end <- Sys.time()
time.end - time.start

