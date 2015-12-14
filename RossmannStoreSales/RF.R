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


train <- read.table("data/train.csv",sep=',',header = T)
test <- read.table("data/test.csv",sep=',',header = T)
store <- read.table("data/store.csv",sep=',',header = T)
# Merge Store stuff
train <- merge(train,store)
test <- merge(test,store)




## summary of data
cat("train data column names and details\n")
names(train)
str(train)
summary(train)
cat("test data column names and details\n")
names(test)
str(test)
summary(test)




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

##replace NA to default value
train$CompetitionOpenSinceMonth[is.na(train$CompetitionOpenSinceMonth)] <- 8
test$CompetitionOpenSinceMonth[is.na(test$CompetitionOpenSinceMonth)] <- 8
train$CompetitionOpenSinceYear[is.na(train$CompetitionOpenSinceYear)] <- 2009
test$CompetitionOpenSinceYear[is.na(test$CompetitionOpenSinceYear)] <- 2009



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

# There are some NAs in the integer columns so conversion to zero
train[is.na(train)]   <- 0
test[is.na(test)]   <- 0



# target variables
train$Sales <- as.numeric(train$Sales)
train$Customers <- NULL #as.numeric(train$Customers)

train$Sales <- log(train$Sales+1)

set.seed(1785)
fitControl <- trainControl(method="cv", number=3, verboseIter=T)
rfFit <- train(Sales ~.,
               method="rf", data=train, ntree=50, importance=TRUE,
               sampsize=100000,
               do.trace=10, trControl=fitControl)
pred <- predict(rfFit, test)
submit = data.frame(Id = test$Id, Sales = (exp(pred) -1))

write.csv(submit, "output/rf_sub.csv", row.names = FALSE, quote = FALSE)