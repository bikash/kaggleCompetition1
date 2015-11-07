
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

summary(train)
summary(test)


train<- merge(store, train, by="Store")
test<- merge(store, test, by="Store")

#Lets explore the sales data for a quick seconds
hist(train$Sales, 100)


#Interesting. Shockingly, it seemst that stores which are not open cannot make sales. 
#Since predicting the sales figures of non-open stores seems simple, we should only consider cases
#store are open
train.opensample<- subset(train, Open >0)
hist(train.opensample$Sales, 100)



#Add Year, Month, Week and Day to trainingset
train = as.data.table(train)
train$day = day(as.Date(train$Date,"%y/%m/%d"))
train$week = week(as.Date(train$Date,"%y/%m/%d"))
train$month = month(as.Date(train$Date,"%y/%m/%d"))
train$year = year(as.Date(train$Date,"%y/%m/%d"))


rf1 <- randomForest( Sales~Customers+Promo+StateHoliday+SchoolHoliday, data = train.opensample, ntree= 100, importance= TRUE)
imp <- varImp(rf1)
imp
test <-importance(rf1, type=2)

featureImportance <- data.frame(Feature=row.names(test), Importance=test[,1])
p <- ggplot(featureImportance, aes(x=reorder(Feature, Importance), y=Importance)) +geom_bar(stat="identity", fill="#53cfff") +coord_flip() + theme_light(base_size=20) + ylab("Importance") + ggtitle("Random Forest Feature Importance\n") +theme(plot.title=element_text(size=18))

