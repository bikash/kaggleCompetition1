library(caret)
library(data.table)
library(forecast)
library(ggplot2)
library(zoo)
require(dplyr)
library(randomForest)

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

# check is.na
sum(is.na(train))     # 0
sum(is.na(store))                           # 1799
sum(is.na(store$CompetitionDistance))        #   3
sum(is.na(store$CompetitionOpenSinceYear))   # 354
sum(is.na(store$CompetitionOpenSinceMonth))  # 354
sum(is.na(store$Promo2SinceYear))            # 544
sum(is.na(store$Promo2SinceWeek))            # 544 #544 + 544 + 354 + 354 + 3   # = 1799

# seperating out the elements of the date column for the train set
train$Date <- ymd(train$Date)
test$Date <- ymd(test$Date)
train$day <- as.factor(day(train$Date))
test$day <- as.factor(day(test$Date))
train$month <- as.factor(month(train$Date))
test$month <- as.factor(month(test$Date))
train$year <- as.factor(year(train$Date))
test$year <- as.factor(year(test$Date))

## merge
train <- merge(train,store)
test <- merge(test,store)

# There are some NAs in the integer columns so conversion to zero
train[is.na(train)]   <- 0
test[is.na(test)]   <- 0



# looking at only stores that were open in the train set
train <- train[ which(train$Open=='1'),]