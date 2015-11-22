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


# Convert NA values
# NA values are found in FinelineNumber and Upc. "NULL" string value is found in DepartmentDescription.
test$FinelineNumber <- addNA(test$FinelineNumber)
test$Upc <- addNA(test$Upc)
train$FinelineNumber <- addNA(train$FinelineNumber)
train$Upc <- addNA(train$Upc)
train <- train[, NullDescription:=ifelse (train$DepartmentDescription == "NULL", 1, 0)]
test <- test[, NullDescription:=ifelse (train$DepartmentDescription == "NULL", 1, 0)]






