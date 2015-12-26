# airbnb X Kaggle

#############################################################################################################################
#There are 12 possible outcomes of the destination country: 'US', 'FR', 'CA', 'GB', 'ES', 'IT', 'PT', 'NL','DE', 'AU', 
#'NDF' (no destination found), and 'other'. 
#'Please note that 'NDF' is different from 'other' because 'other' means there was a booking, but is to a country not 
#'included in the list, while 'NDF' means there wasn't a booking.
##############################################################################################################################
# read in csv files
setwd("/Users/bikash/repos/kaggleCompetition1/airbnb")
# Set seed
set.seed(1785)
ageSummary <- read.table('data/age_gender_bkts.csv', sep=',', stringsAsFactors=F, header=T)
countriesSummary <- read.table('data/countries.csv', sep=',', stringsAsFactors=F, header=T)
sessions <- read.table('data/sessions.csv', sep=',', stringsAsFactors=T, header=T)
trainUsers <- read.table('data/train_users.csv', sep=',', stringsAsFactors=T, header=T)
#############################################################################################################################
#id: user id
#date_account_created: the date of account creation
#timestamp_first_active ->timestamp of the first activity, note that it can be earlier than date_account_created or date_first_booking because a user can search before signing up
#date_first_booking  ->date of first booking  (No longer in use)
#gender 
#age 
#signup_method 
#signup_flow ->the page a user came to signup up from
#language -> international language preference
#affiliate_channel ->what kind of paid marketing
#affiliate_provider ->where the marketing is e.g. google, craigslist, other
#first_affiliate_tracked ->whats the first marketing the user interacted with before the signing up
#signup_app first_device_type 
#first_browser 
#country_destination -> this is the target variable you are to predict
#############################################################################################################################

testUsers <- read.table('data/test_users.csv', sep=',', stringsAsFactors=T, header=T)


## Remove unused column date_first_booking
testUsers$date_first_booking <- NULL
trainUsers$date_first_booking <- NULL

###Total number of training and testing datasets
str(testUsers) 
str(trainUsers)

###Analysis training datasets
df <- rbind(trainUsers[,-15],testUsers)

## total gender in dataset is 275547 out of which 68209 is male and 77524 is female and rest is -unknown- and OTHER
length(df[(df$gender=="MALE"),]$gender)
#[1] 68209
length(df[(df$gender=="FEMALE"),]$gender)
#[1] 77524 
length(df[(df$gender=="OTHER"),]$gender)
## 334 are Other



# libraries
library(ggplot2)
library(sqldf)
library(reshape2)
library(randomForest)

# merge ageSummary and countriesSummary for plotting
ageCountries <- merge(ageSummary, countriesSummary, by="country_destination", all.x=TRUE)
#write.csv(ageCountries, "ageCountries.csv", row.names=FALSE)

# Build ABT for classification algorithm
# Using sqldf to manipulate dataset
sessions$action_type[sessions$action_type==''] <- '-unknown-'
sessionsActionType <- sqldf("select user_id, 
                            case when action_type=='booking_request' then 'booking_request'
                            when action_type=='booking_response' then 'booking_response'
                            when action_type=='click' then 'click'
                            when action_type=='data' then 'data'
                            when action_type=='message_post' then 'message_post'
                            when action_type=='modify' then 'modify'
                            when action_type=='partner_callback' then 'partner_callback'
                            when action_type=='submit' then 'submit'
                            when action_type=='view' then 'view'
                            else 'other_action_types' end as action_type,
                            sum(secs_elapsed) as secs_elapsed
                            from sessions
                            group by user_id, action_type
                            ")
sessionsDeviceType <- sqldf("select user_id,
                            case when device_type=='Android App Unknown Phone/Tablet' then 'android_app_unknown_phone_tablet'
                            when device_type=='Android Phone' then 'android_phone'
                            when device_type=='Blackberry' then 'blackberry'
                            when device_type=='Chromebook' then 'chromebook'
                            when device_type=='iPad Tablet' then 'iPad_tablet'
                            when device_type=='iPodtouch' then 'iPodtouch'
                            when device_type=='Linux Desktop' then 'linux_desktop'
                            when device_type=='Mac Desktop' then 'mac_desktop'
                            when device_type=='Opera Phone' then 'opera_phone'
                            when device_type=='Tablet' then 'tablet'
                            when device_type=='Windows Desktop' then 'windows_desktop'
                            when device_type=='Windows Phone' then 'windows_phone'
                            else 'other_device_types' end as device_type,
                            sum(secs_elapsed) as secs_elapsed
                            from sessions
                            group by user_id, device_type
                            ")
# use reshape2 package to 'dcast' the above 2 tables separately
sessionsActionTypeNew <- subset(sessionsActionType, user_id!='')
sessionsActionTypeNew <- subset(sessionsDeviceType, user_id!='')
row.names(sessionsActionTypeNew) <- NULL
row.names(sessionsDeviceTypeNew) <- NULL
userActionType <- dcast(sessionsActionTypeNew, user_id~action_type, sum)
userDeviceType <- dcast(sessionsDeviceTypeNew, user_id~device_type, sum)

# merge userActionType & userDeviceType
userActionDeviceSecsElapsed <- merge(userActionType, userDeviceType, by="user_id", all.x=T, all.y=T)
# replace NA with 0
userActionDeviceSecsElapsed[is.na(userActionDeviceSecsElapsed)] <- 0

# clean trainUsers/testUsers
## change id to user_id
names(trainUsers)[1] <- "user_id"
names(testUsers)[1] <- "user_id"

## transform date variables to date format; time variable to time format
trainUsers$date_account_created <- as.Date(trainUsers$date_account_created, "%Y-%m-%d")
testUsers$date_account_created <- as.Date(testUsers$date_account_created, "%Y-%m-%d")
trainUsers$date_first_booking <- as.Date(trainUsers$date_account_created, "%Y-%m-%d")
testUsers$date_first_booking <- as.Date(testUsers$date_account_created, "%Y-%m-%d")
trainUsers$timestamp_first_active <- as.character(trainUsers$timestamp_first_active)
testUsers$timestamp_first_active <- as.character(testUsers$timestamp_first_active)
trainUsers$timestamp_first_active <- strptime(trainUsers$timestamp_first_active, "%Y%m%d%H%M%S")
testUsers$timestamp_first_active <- strptime(testUsers$timestamp_first_active, "%Y%m%d%H%M%S")
## impute missing values: mean substitution, replace age NA with mean(age)
trainUsers[is.na(trainUsers$age), "age"] <- round(mean(trainUsers$age, na.rm=T))
testUsers[is.na(testUsers$age), "age"] <- round(mean(testUsers$age, na.rm=T))
## create new variable
trainUsers$date_first_active <- as.Date(trainUsers$timestamp_first_active)
testUsers$date_first_active <- as.Date(testUsers$timestamp_first_active)
trainUsers$days_first_booking_active <- trainUsers$date_first_booking-trainUsers$date_first_active
testUsers$days_first_booking_active <- testUsers$date_first_booking-testUsers$date_first_active
trainUsers$days_first_booking_created <- trainUsers$date_first_booking-trainUsers$date_account_created
testUsers$days_first_booking_created <- testUsers$date_first_booking-testUsers$date_account_created

# merge trainUsers/testUsers with userActionDeviceSecsElapsed
train <- merge(trainUsers, userActionDeviceSecsElapsed, by="user_id", all.x=TRUE)
test <- merge(testUsers, userActionDeviceSecsElapsed, by="user_id", all.x=TRUE)

# export training/test datasets
write.csv(train, "data/train_v1.csv", row.names=FALSE)
write.csv(test, "data/test_v1.csv", row.names=FALSE)

# Time for CLASSIFICATION!

## Use Random Forest
## Read in training/test datasets

train <- read.table('data/train_v1.csv', sep=',', header=T, colClasses=c("date_account_created"="Date", "timestamp_first_active"="POSIXct", "date_first_booking"="Date", "date_first_active"="Date"))
test <- read.table('data/test_v1.csv', sep=',', header=T, colClasses=c("date_account_created"="Date", "timestamp_first_active"="POSIXct", "date_first_booking"="Date", "date_first_active"="Date"))

### option1: replace NA with 0
train[is.na(train)] <- 0
test[is.na(test)] <- 0
### option2: take complete cases only
train <- train[complete.cases(train),]

## Merge countriesSummary with train for visualization
trainVisual <- merge(train, countriesSummary, by="country_destination", all.x=TRUE)
write.csv(trainVisual, "data/trainVisual.csv", row.names=FALSE)

## run model
rfModel <- randomForest(x=train[, c(-1,-16)], y=train[, 16], mtry=21, nodesize=10,importance=TRUE)#sampsize
## plot variable importance
varImpPlot(rfModel)
## output importance dataframe
importance <- importance(rfModel)
importanceDf <- data.frame(importance)
varNames <- rownames(importanceDf)
rownames(importanceDf) <- NULL
impDf <- cbind(varNames, importanceDf)
write.csv(impDf, "output/trainrf.csv", row.names=FALSE)
