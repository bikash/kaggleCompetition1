# Load Dependeicies
library(caret)
library(randomForest)
library(readr)
library(lubridate)
library(plyr)
library(doMC)
registerDoMC(cores = 4)

setwd("/Users/bikash/repos/kaggleCompetition1/RossmannStoreSales")
# Set seed
set.seed(12345)

# train <- read_csv("data/train.csv")
# test  <- read_csv("data/test.csv")
# store <- read_csv("data/store.csv")

train <- read.table("data/train.csv",sep=',',header = T)
test <- read.table("data/test.csv",sep=',',header = T)
store <- read.table("data/store.csv",sep=',',header = T)
# Merge Store stuff
train <- merge(train, store, by=c('Store'))
test <- merge(test, store, by=c('Store'))
test <- arrange(test,Id)

train1 <- train
### 
## for testing purpose
train <- train[1:30000,]
test <- train1[30001:35000,]
length(train$Store)

# There are some NAs in the integer columns so conversion to zero
train[is.na(train)]   <- 0
test[is.na(test)]   <- 0

## summary of data
cat("train data column names and details\n")
names(train)
str(train)
summary(train)
cat("test data column names and details\n")
names(test)
str(test)
summary(test)

# looking at only stores that were open in the train set
# may change this later
train <- train[ which(train$Open=='1'),]
train <- train[ which(train$Sales!='0'),]
# seperating out the elements of the date column for the train set
train$Date <- ymd(train$Date)
test$Date <- ymd(test$Date)
train$day <- as.factor(day(train$Date))
test$day <- as.factor(day(test$Date))
train$month <- as.factor(month(train$Date))
test$month <- as.factor(month(test$Date))
train$year <- as.factor(year(train$Date))
test$year <- as.factor(year(test$Date))

# removing the date column (since elements are extracted) and also StateHoliday which has a lot of NAs (may add it back in later)
train <- train[,-c(3,8)] ## remove date, and stateHoliday
# removing the date column (since elements are extracted) and also StateHoliday which has a lot of NAs (may add it back in later)
test <- test[,-c(3,8)]

feature.names <- names(train)[c(1,2,5:19)] ##remove open and promo
cat("Feature Names\n")
feature.names

cat("assuming text variables are categorical & replacing them with numeric ids\n")
for (f in feature.names) {
  if (class(train[[f]])=="character") {
    levels <- unique(c(train[[f]], test[[f]]))
    train[[f]] <- as.integer(factor(train[[f]], levels=levels))
    test[[f]]  <- as.integer(factor(test[[f]],  levels=levels))
  }
}

cat("train data column names after slight feature engineering\n")
names(train)
cat("test data column names after slight feature engineering\n")
names(test)
tra<-train[,feature.names]
RMPSE<- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  elab<-exp(as.numeric(labels))-1
  epreds<-exp(as.numeric(preds))-1
  err <- sqrt(mean((epreds/elab-1)^2))
  return(list(metric = "RMPSE", value = err))
}
nrow(train)
h<-sample(nrow(train),10000)

dval<-xgb.DMatrix(data=data.matrix(tra[h,]),label=log(train$Sales+1)[h])
dtrain<-xgb.DMatrix(data=data.matrix(tra[-h,]),label=log(train$Sales+1)[-h])
watchlist<-list(val=dval,train=dtrain)
param <- list(  objective           = "reg:linear", 
                booster = "gbtree",
                eta                 = 0.02, # 0.06, #0.01,
                max_depth           = 10, #changed from default of 8
                subsample           = 0.9, # 0.7
                colsample_bytree    = 0.7 # 0.7
                #num_parallel_tree   = 2
                # alpha = 0.0001, 
                # lambda = 1
)

clf <- xgb.train(   params              = param, 
                    data                = dtrain, 
                    nrounds             = 3000, #300, #280, #125, #250, # changed from 300
                    verbose             = 0,
                    early.stop.round    = 100,
                    watchlist           = watchlist,
                    maximize            = FALSE,
                    feval=RMPSE
)
pred1 <- exp(predict(clf, data.matrix(test[,feature.names]))) -1

# pred2 should be identical to pred
print(paste("sum(abs(pred1-test$Sales))=", sum(abs(pred1-test$Sales))))

#cmat = confusionMatrix(pred1[1:10], test$Sales[1:10])
#accuracy(cmat)

# Only evaludated on Open days
train <- train[ which(train$Open=='1'),]
train$Open <- NULL
test$Open <- NULL

train$PromoInterval <-NULL
test$PromoInterval <-NULL


# Date stuff
train$Date <- ymd(train$Date)
test$Date <- ymd(test$Date)
train$day <- as.factor(day(train$Date))
test$day <- as.factor(day(test$Date))
train$month <- as.factor(month(train$Date))
test$month <- as.factor(month(test$Date))
train$year <- as.factor(year(train$Date))
test$year <- as.factor(year(test$Date))
train$Date <- NULL
test$Date <- NULL

# always 1 for training and test
train$StateHoliday <- NULL
test$StateHoliday <- NULL

#Factorize stuff
train$DayOfWeek <- as.factor(train$DayOfWeek)
test$DayOfWeek <- as.factor(test$DayOfWeek)
train$Promo <- as.factor(train$Promo)
test$Promo <- as.factor(test$Promo)
train$SchoolHoliday <- as.factor(train$SchoolHoliday)
test$SchoolHoliday <- as.factor(test$SchoolHoliday)

# Factorize store stuff
train$StoreType <- as.factor(train$StoreType)
test$StoreType <- as.factor(test$StoreType)
train$Assortment <- as.factor(train$Assortment)
test$Assortment <- as.factor(test$Assortment)
train$CompetitionDistance <- as.numeric(train$CompetitionDistance)
test$CompetitionDistance <- as.numeric(test$CompetitionDistance)
train$Promo2 <- as.factor(train$Promo2)
test$Promo2 <- as.factor(test$Promo2)
train$PromoInterval <- as.factor(train$PromoInterval)
test$PromoInterval <- as.factor(test$PromoInterval)

competition_start <- strptime('20.10.2015', format='%d.%m.%Y')
train$CompetitionDaysOpen <- as.numeric(difftime(competition_start,
                                                 strptime(paste('1',
                                                                train$CompetitionOpenSinceMonth,
                                                                train$CompetitionOpenSinceYear, sep = '.'),
                                                          format='%d.%m.%Y'), units='days'))
test$CompetitionDaysOpen <- as.numeric(difftime(competition_start,
                                                strptime(paste('1',
                                                               test$CompetitionOpenSinceMonth,
                                                               test$CompetitionOpenSinceYear, sep = '.'),
                                                         format='%d.%m.%Y'), units='days'))
train$CompetitionDaysOpen[is.na(train$CompetitionDaysOpen)] <- 0
test$CompetitionDaysOpen[is.na(test$CompetitionDaysOpen)] <- 0

train$CompetitionWeeksOpen <- train$CompetitionDaysOpen/7
test$CompetitionWeeksOpen <- test$CompetitionDaysOpen/7

train$CompetitionMonthsOpen <- train$CompetitionDaysOpen/30
test$CompetitionMonthsOpen <- test$CompetitionDaysOpen/30

train$CompetitionYearsOpen <- train$CompetitionWeeksOpen/52
test$CompetitionYearsOpen <- test$CompetitionWeeksOpen/52

train$CompetitionOpenSinceMonth <- NULL
train$CompetitionOpenSinceYear <- NULL
test$CompetitionOpenSinceMonth <- NULL
test$CompetitionOpenSinceYear <- NULL

# target variables
train$Sales <- as.numeric(train$Sales)
train$Customers <- NULL #as.numeric(train$Customers)

train$Sales <- log(train$Sales+1)

set.seed(1234)
fitControl <- trainControl(method="cv", number=3, verboseIter=T)
rfFit <- train(Sales ~.,
               method="rf", data=train, ntree=50, importance=TRUE,
               sampsize=100000,
               do.trace=10, trControl=fitControl)


pred <- predict(rfFit, test)
submit = data.frame(Id = test$Id, Sales = (exp(pred) -1))

write.csv(submit, "output/rf_sub.csv", row.names = FALSE, quote = FALSE)