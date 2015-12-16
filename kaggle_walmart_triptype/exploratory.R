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


train.data <- read.table("data/train.csv",sep=',',header = T)
test <- read.table("data/test.csv",sep=',',header = T)

# Create unique trip types for result
outcomes <- data.frame(TripType = sort(unique(train.data$TripType)))
outcomes$Index <- seq_along(outcomes$TripType) - 1

### Training data VS TripType
y <- plyr::mapvalues(train.data$TripType, from = outcomes$TripType, to = outcomes$Index)
train.data$label <- y
library(dplyr)
library(ggvis)
train.data %>%
  group_by(label) %>%
  summarize(count=length(label)) %>%
  ggvis(~label, ~count) %>%
  layer_bars(fill:="#20beff")

##training data VS DepartmentDescription
train.data$dep <- as.integer(train.data$DepartmentDescription)
train.data %>%
  group_by(dep) %>%
  summarize(count=length(dep)) %>%
  ggvis(~dep, ~count) %>%
  layer_bars(fill:="#20beff")

##training data VS FinelineNumber
train.data %>%
  group_by(FinelineNumber) %>%
  summarize(count=length(FinelineNumber)) %>%
  ggvis(~FinelineNumber, ~count) %>%
  layer_bars(fill:="#20beff")


##training data VS FinelineNumber
train.data %>%
  group_by(Upc) %>%
  summarize(count=length(Upc)) %>%
  ggvis(~Upc, ~count) %>%
  layer_bars(fill:="#20beff")

##training data VS Weekday
train.data$wday <- as.integer(train.data$Weekday)
train.data %>%
  group_by(wday) %>%
  summarize(count=length(wday)) %>%
  ggvis(~wday, ~count) %>%
  layer_bars(fill:="#20beff")

##testing data VS Weekday
test$wday <- as.integer(test$Weekday)
test %>%
  group_by(wday) %>%
  summarize(count=length(wday)) %>%
  ggvis(~wday, ~count) %>%
  layer_bars(fill:="#20beff")


## get different UPC
library(data.table)
t <- model.matrix(~ 0 + VisitNumber + FinelineNumber, data = train.data) %>% as.data.table()




