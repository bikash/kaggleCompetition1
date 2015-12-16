## xgboost classification


library(dplyr)
library(xgboost)
library(Matrix)
library(ggplot2)


setwd("/Users/bikash/repos/kaggleCompetition1/TelstraNetworkFailure")
set.seed(12345)
event_type <- read.table("data/event_type.csv",sep=',',header = T)
##id    event_type
log <- read.table("data/log_feature.csv",sep=',',header = T)
##id log_feature volume
severity<- read.table("data/severity_type.csv",sep=',',header = T)
##id   severity_type
train <- read.table("data/train.csv",sep=',',header = T)
## id     location fault_severity

test <- read.table("data/test.csv",sep=',',header = T)

resource_type <- read.table("data/resource_type.csv",sep=',',header = T)
##id   resource_type

##join table
a <- merge(train, event_type, by=c('id'))
b <- merge(a, severity, by=c('id'))
c <- merge(b, log, by=c('id'))
train.data <- merge(c, resource_type, by=c('id'))
##  id     location fault_severity    event_type   severity_type log_feature volume   resource_type

## convert categorical varaible to integer 
train.data$event_type <- as.integer(train.data$event_type)
train.data$log_feature <- as.integer(train.data$log_feature)
train.data$resource_type <- as.integer(train.data$resource_type)
train.data$severity_type <- as.integer(train.data$severity_type)
train.data$location <- as.integer(train.data$location)
