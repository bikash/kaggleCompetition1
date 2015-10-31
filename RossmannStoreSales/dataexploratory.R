

library(data.table)
library(forecast)
library(ggplot2)
library(zoo)
require(dplyr)


setwd("/Users/bikash/repos/kaggleCompetition1/RossmannStoreSales")
# Set seed
set.seed(123)

## data load
train <- read.table("data/train.csv",sep=',',header = T)
test <- read.table("data/test.csv",sep=',',header = T)
store <- read.table("data/store.csv",sep=',',header = T)

str(train) #1017209
str(store) ##1115
str(test) #41088

train$Date <- as.Date(train$Date)
test$Date <- as.Date(test$Date)


summary(train)
summary(test)

test[is.na(test$Open), ] # Only store 622
test$Open[test$Store == 622]
test[is.na(test)] <- 1

# Unique values per column
train[, lapply(.SD, function(x) length(unique(x)))]