library(caret)
#library(Quicktune)


library(caret)
library(randomForest)
library(compare)
library(magrittr)

#library(Standard)

setwd("/Users/bikash/repos/kaggleCompetition1/homesite-quote-conversion")


# EDA

Xtt0 <- read.csv("input/train.csv", stringsAsFactors = TRUE, header = TRUE)
test0 <- read.csv("input/test.csv", stringsAsFactors = TRUE, header = TRUE)

# Class distribution
table(Xtt0$QuoteConversion_Flag)

# How many columns are numeric, factors, etc.?
classes <- NULL
for(i in seq(ncol(Xtt0))){
  
  v <- Xtt0[,i]
  classes <- c(classes, class(v))
}

table(classes)

# take a look at the categorical variables
summary(Xtt0[,which(classes == "factor")])
for(i in which(classes == "factor")){
  cat(paste0("i: ", i, "\n"))
  cat(paste0("Unique:" , length(unique(Xtt0[,i])), "\n"))
}

# How many variables have NAs?
which(ColWithNA(Xtt0) == TRUE)
which(ColWithNA(test0) == TRUE)

# Able to identify customers with mulitple quotes? This indicate interest
personalfields <- grep(x = colnames(Xtt0), pattern = "PersonalField")
str(Xtt0[,personalfields])
uniquecust <- unique(Xtt0[,personalfields])
dim(uniquecust) # 178941 possibly unique customers (at most)

# What if I extend the analysis to property and geographic fields?
propertyfields <- grep(x = colnames(Xtt0), pattern = "PropertyField")
geographicfields <- grep(x = colnames(Xtt0), pattern = "GeographicField")
uniquecustandhse <- unique(Xtt0[, union(personalfields, union(propertyfields, geographicfields))])
dim(uniquecustandhse)




# ====
# Xtt1
# Drop categorical variables, only keep numeric and integer ones

# Drop categorical variables
keep <- union(which(classes == "numeric"), which(classes == "integer"))
Xtt1 <- Xtt0[,keep]

# Drop columns with NAs
Xtt1 <- subset(Xtt1, select = -c(PersonalField84, PropertyField29))

test1 <- test0[,which(colnames(test0) %in% colnames(Xtt1))]

Xtt1$QuoteConversion_Flag <- factor(make.names(Xtt1$QuoteConversion_Flag))

# Prep for CV
set.seed(86647)
tr_idx <- createDataPartition(Xtt1$QuoteConversion_Flag, p = 0.7, list = FALSE)
Xtrain1 <- Xtt1[tr_idx,]
Xtest1 <- Xtt1[-tr_idx,]

Xtrain1_small <- Xtrain1[1:(nrow(Xtrain1)/10),]
Xtrain1_small_down <- downSample(x = subset(Xtrain1_small, select = -QuoteConversion_Flag),
                                 y = Xtrain1_small$QuoteConversion_Flag,
                                 list = FALSE,
                                 yname = "QuoteConversion_Flag")

# ====
# Xtt2
# Drop variables with NA
# Cate2Prob


Xtt2 <- Xtt0
Xtt2[,which(classes == "factor")] <- apply(Xtt2[,which(classes == "factor")],
                                           FUN = Cate2Prob,
                                           MARGIN = 2)
test2 <- test0

classes_test2 <- NULL
for(i in seq(ncol(test2))){
  
  v <- test2[,i]
  classes_test2 <- c(classes_test2, class(v))
}

test2[,which(classes_test2 == "factor")] <- apply(test2[,which(classes_test2 == "factor")],
                                                  FUN = Cate2Prob,
                                                  MARGIN = 2)

# Remove two variables with NAs
Xtt2 <- subset(Xtt2, select = -c(PersonalField84, PropertyField29))
test2 <- test2[,which(colnames(test2) %in% colnames(Xtt2))]

Xtt2$QuoteConversion_Flag <- factor(make.names(Xtt2$QuoteConversion_Flag))

save(list = "Xtt2", file = "data/Xtt2.RData")
save(list = "test2", file = "data/test2.RData")

# Prep for CV
set.seed(86647)
tr_idx <- createDataPartition(Xtt2$QuoteConversion_Flag, p = 0.7, list = FALSE)
Xtrain2 <- Xtt2[tr_idx,]
Xtest2 <- Xtt2[-tr_idx,]

Xtrain2_small <- Xtrain2[1:(nrow(Xtrain2)/10),]
Xtrain2_small_down <- downSample(x = subset(Xtrain2_small, select = -QuoteConversion_Flag),
                                 y = Xtrain2_small$QuoteConversion_Flag,
                                 list = FALSE,
                                 yname = "QuoteConversion_Flag")


save(list = "Xtrain2", file = "data/Xtrain2.RData")
save(list = "Xtest2", file = "data/Xtest2.RData")
save(list = "Xtrain2_small", file = "data/Xtrain2_small.RData")
save(list = "Xtrain2_small_down", file = "data/Xtrain2_small_down.RData")


# ====
# Xtt3
# Cate2Prob
# Drop variables with NA
# Include three additional noise features as VI benchmarks
# Break each date into day, weekday, month, year, quarters
# OneHotEncode the temporal variables


Xtt3 <- Xtt0
test3 <- test0

Xtt3$Original_Quote_Date <- as.Date(Xtt3$Original_Quote_Date, format = "%Y-%m-%d")
test3$Original_Quote_Date <- as.Date(test3$Original_Quote_Date, format = "%Y-%m-%d")

# Treat categorical variables
classes <- NULL
for(i in seq(ncol(Xtt3))){
  
  v <- Xtt3[,i]
  classes <- c(classes, class(v))
}


Xtt3[,which(classes == "factor")] <- apply(Xtt3[,which(classes == "factor")],
                                           FUN = Cate2Prob,
                                           MARGIN = 2)

classes_test3 <- NULL
for(i in seq(ncol(test3))){
  
  v <- test3[,i]
  classes_test3 <- c(classes_test3, class(v))
}

test3[,which(classes_test3 == "factor")] <- apply(test3[,which(classes_test3 == "factor")],
                                                  FUN = Cate2Prob,
                                                  MARGIN = 2)

# Remove two variables with NAs
Xtt3 <- subset(Xtt3, select = -c(PersonalField84, PropertyField29))
test3 <- test3[,which(colnames(test3) %in% colnames(Xtt3))]

Xtt3$QuoteConversion_Flag <- factor(make.names(Xtt3$QuoteConversion_Flag))

Xtt3$Noise1 <- runif(n = nrow(Xtt3), min = 0, max = nrow(Xtt3))
Xtt3$Noise2 <- runif(n = nrow(Xtt3), min = 0, max = nrow(Xtt3))
Xtt3$Noise3 <- runif(n = nrow(Xtt3), min = 0, max = nrow(Xtt3))


test3$Noise1 <- runif(n = nrow(test3), min = 0, max = nrow(test3))
test3$Noise2 <- runif(n = nrow(test3), min = 0, max = nrow(test3))
test3$Noise3 <- runif(n = nrow(test3), min = 0, max = nrow(test3))

Xtt3$Quarters <- quarters(Xtt3$Original_Quote_Date)
test3$Quarters <- quarters(test3$Original_Quote_Date)

Xtt3$Day <- weekdays(Xtt3$Original_Quote_Date)
test3$Day <- weekdays(test3$Original_Quote_Date)

Xtt3$Month <- months(Xtt3$Original_Quote_Date)
test3$Month <- months(test3$Original_Quote_Date)

Xtt3$Year <- as.numeric(format(Xtt3$Original_Quote_Date, "%Y"))
test3$Year <- as.numeric(format(test3$Original_Quote_Date, "%Y"))

dim(Xtt3)
dim(test3)
setdiff(colnames(Xtt3), colnames(test3))

# combine the dataset for one hot encoding
y <- Xtt3$QuoteConversion_Flag
Xtt3 <- subset(Xtt3, select = -QuoteConversion_Flag)
Xtt3_test3 <- rbind(Xtt3, test3)

Xtt3_test3 <- OneHotEncode(Xtt3_test3, type = "test")

# Split back
Xtt3 <- cbind(y, Xtt3_test3[(1:length(y)),])
colnames(Xtt3)[1] <- "QuoteConversion_Flag"
test3 <- Xtt3_test3[((length(y) + 1): nrow(Xtt3_test3)),]


dim(Xtt3)
dim(test3)
setdiff(colnames(Xtt3), colnames(test3))

# Drop the dates
Xtt3 <- subset(Xtt3, select = -Original_Quote_Date)
test3 <- subset(test3, select = -Original_Quote_Date)


save(list = "Xtt3", file = "data/Xtt3.RData")
save(list = "test3", file = "data/test3.RData")


# Prep for CV
set.seed(86647)
tr_idx <- createDataPartition(Xtt3$QuoteConversion_Flag, p = 0.7, list = FALSE)
Xtrain3 <- Xtt3[tr_idx,]
Xtest3 <- Xtt3[-tr_idx,]

Xtrain3_small <- Xtrain3[1:(nrow(Xtrain3)/10),]
Xtrain3_small_down <- downSample(x = subset(Xtrain3_small, select = -QuoteConversion_Flag),
                                 y = Xtrain3_small$QuoteConversion_Flag,
                                 list = FALSE,
                                 yname = "QuoteConversion_Flag")


save(list = "Xtrain3", file = "data/Xtrain3.RData")
save(list = "Xtest3", file = "data/Xtest3.RData")
save(list = "Xtrain3_small", file = "data/Xtrain3_small.RData")
save(list = "Xtrain3_small_down", file = "data/Xtrain3_small_down.RData")


# ====
# Xtt4
# Cate2Prob
# Impute NAs with -1, count/sum -1's (https://www.kaggle.com/c/homesite-quote-conversion/forums/t/18225/two-insights-for-0-96852)
# Break each date into day, weekday, month, year, quarters
# OneHotEncode the temporal variables

# (done on training and test sets simultaneously)
# *number of quotes a (possibly unique) customer makes
# *the last quote made by a customer given that he/she made multiple quotes (binary)
# *whether the customer accepted before? (binary)

Xtt4 <- Xtt0
test4 <- test0

# Combine training and testing sets
y <- Xtt4$QuoteConversion_Flag
Xtt4 <- subset(Xtt4, select = -QuoteConversion_Flag)
Xtt4_test4 <- rbind(Xtt4, test4)



personalfields <- grep(x = colnames(Xtt4_test4), pattern = "PersonalField")
uniquecust <- unique(Xtt4_test4[,personalfields])
str(uniquecust); dim(uniquecust)

# Xtt4_test4_personal <- Xtt4_test4[,personalfields]
# comp <- compare(uniquecust, Xtt4_test4_personal[10:100,], allowAll = FALSE)
# head(comp$tM)


HasMultipleQuotes <- apply(Xtt4_test4[1:1000,], MARGIN = 1, FUN = function(r){
  #print(str(r)); print(length(r))
  r <- r[personalfields]
  r <- as.data.frame(t(r))
  #print(colnames(r))
  tmp <- match_df(uniquecust, r)
  if(nrow(tmp) == 0) return(0)
  else return(1)
})


Xtt4$Original_Quote_Date <- as.Date(Xtt4$Original_Quote_Date, format = "%Y-%m-%d")
test4$Original_Quote_Date <- as.Date(test4$Original_Quote_Date, format = "%Y-%m-%d")

# Treat categorical variables
# Should write a function for this in SW..
classes <- NULL
for(i in seq(ncol(Xtt4))){v <- Xtt4[,i]; classes <- c(classes, class(v))}
Xtt4[,which(classes == "factor")] <- apply(Xtt4[,which(classes == "factor")],
                                           FUN = Cate2Prob,
                                           MARGIN = 2)
classes_test4 <- NULL
for(i in seq(ncol(test4))){v <- test4[,i]; classes_test4 <- c(classes_test4, class(v))}
test4[,which(classes_test4 == "factor")] <- apply(test4[,which(classes_test4 == "factor")],
                                                  FUN = Cate2Prob,
                                                  MARGIN = 2)

# Impute PersonalField84 and PropertyField29 with -1
Xtt4$PersonalField84[which(is.na(Xtt4$PersonalField84))] <- -1
Xtt4$PropertyField29[which(is.na(Xtt4$PropertyField29))] <- -1
test4$PersonalField84[which(is.na(test4$PersonalField84))] <- -1
test4$PropertyField29[which(is.na(test4$PropertyField29))] <- -1

# Count number of -1's
Xtt4$NumMinus1 <- apply(Xtt4, MARGIN = 1, FUN = function(r){return(length(which(r == -1)))})
test4$NumMinus1 <- apply(test4, MARGIN = 1, FUN = function(r){return(length(which(r == -1)))})

# make.names
Xtt4$QuoteConversion_Flag <- factor(make.names(Xtt4$QuoteConversion_Flag))

# Temporal variables
Xtt4$Quarters  <- quarters(Xtt4$Original_Quote_Date)
test4$Quarters <- quarters(test4$Original_Quote_Date)
Xtt4$Day       <- weekdays(Xtt4$Original_Quote_Date)
test4$Day      <- weekdays(test4$Original_Quote_Date)
Xtt4$Month     <- months(Xtt4$Original_Quote_Date)
test4$Month    <- months(test4$Original_Quote_Date)
Xtt4$Year      <- as.numeric(format(Xtt4$Original_Quote_Date, "%Y"))
test4$Year     <- as.numeric(format(test4$Original_Quote_Date, "%Y"))

dim(Xtt4)
dim(test4)
setdiff(colnames(Xtt4), colnames(test4))

# combine the dataset for one hot encoding
y <- Xtt4$QuoteConversion_Flag
Xtt4 <- subset(Xtt4, select = -QuoteConversion_Flag)
Xtt4_test4 <- rbind(Xtt4, test4)

Xtt4_test4 <- OneHotEncode(Xtt4_test4, type = "test")

# Split back
Xtt4 <- cbind(y, Xtt4_test4[(1:length(y)),])
colnames(Xtt4)[1] <- "QuoteConversion_Flag"
test4 <- Xtt4_test4[((length(y) + 1): nrow(Xtt4_test4)),]

dim(Xtt4)
dim(test4)
setdiff(colnames(Xtt4), colnames(test4))

# Drop the dates
Xtt4 <- subset(Xtt4, select = -Original_Quote_Date)
test4 <- subset(test4, select = -Original_Quote_Date)

save(list = "Xtt4", file = "data/Xtt4.RData")
save(list = "test4", file = "data/test4.RData")

# Prep for CV
set.seed(86647)
tr_idx <- createDataPartition(Xtt4$QuoteConversion_Flag, p = 0.7, list = FALSE)
Xtrain4 <- Xtt4[ tr_idx,]
Xtest4  <- Xtt4[-tr_idx,]

Xtrain4_small <- Xtrain4[1:(nrow(Xtrain4)/10),]
Xtrain4_small_down <- downSample(x = subset(Xtrain4_small, select = -QuoteConversion_Flag),
                                 y = Xtrain4_small$QuoteConversion_Flag,
                                 list = FALSE,
                                 yname = "QuoteConversion_Flag")

save(list = "Xtrain4", file = "data/Xtrain4.RData")
save(list = "Xtest4", file = "data/Xtest4.RData")
save(list = "Xtrain4_small", file = "data/Xtrain4_small.RData")
save(list = "Xtrain4_small_down", file = "data/Xtrain4_small_down.RData")

# ====
# Xtt5


Xtt5 <- Xtt0
test5 <- test0

load("data/temp.RData")

# # Combine training and testing sets
# y <- Xtt5$QuoteConversion_Flag
# Xtt5 <- subset(Xtt5, select = -QuoteConversion_Flag)
# Xtt5_test5 <- rbind(Xtt5, test5)

# treat dates
Xtt5$Original_Quote_Date <- as.Date(Xtt5$Original_Quote_Date, format = "%Y-%m-%d")
test5$Original_Quote_Date <- as.Date(test5$Original_Quote_Date, format = "%Y-%m-%d")
dat$Original_Quote_Date <- as.Date(dat$Original_Quote_Date, format = "%Y-%m-%d", origin = "1970-01-01")

# insert the HPI columns
Xtt5 <- cbind(dat[,1:5], Xtt5)

# match dates for testing set
test5$`30yr_M_rates`        <- rep(0, nrow(test5))
test5$HPI_Qrtly_AT_NSA      <- rep(0, nrow(test5))
test5$`HPI_Qrtly_AT_NSA(%)` <- rep(0, nrow(test5))
test5$HPI_Mthly_P_SA        <- rep(0, nrow(test5))
test5$`HPI_Mthly_P_SA(%)`   <- rep(0, nrow(test5))

for(this_date in dat$Original_Quote_Date %>% unique){
  
  idx_in_dat <- which(dat$Original_Quote_Date == this_date)
  idx_in_test5 <- which(test5$Original_Quote_Date == this_date)
  
  values <- c(dat$`30yr_M_rates`[idx_in_dat]        %>% unique,
              dat$HPI_Qrtly_AT_NSA[idx_in_dat]      %>% unique,
              dat$`HPI_Qrtly_AT_NSA(%)`[idx_in_dat] %>% unique,
              dat$HPI_Mthly_P_SA[idx_in_dat]        %>% unique,
              dat$`HPI_Mthly_P_SA(%)`[idx_in_dat]   %>% unique)
  
  test5$`30yr_M_rates`[idx_in_test5]        <- values[1]
  test5$HPI_Qrtly_AT_NSA[idx_in_test5]      <- values[2]
  test5$`HPI_Qrtly_AT_NSA(%)`[idx_in_test5] <- values[3]
  test5$HPI_Mthly_P_SA[idx_in_test5]        <- values[4]
  test5$`HPI_Mthly_P_SA(%)`[idx_in_test5]   <- values[5]
  
}



# Treat categorical variables
# Should write a function for this in SW..
classes <- NULL
for(i in seq(ncol(Xtt5))){v <- Xtt5[,i]; classes <- c(classes, class(v))}
Xtt5[,which(classes == "factor")] <- apply(Xtt5[,which(classes == "factor")],
                                           FUN = Cate2Prob,
                                           MARGIN = 2)
classes_test5 <- NULL
for(i in seq(ncol(test5))){v <- test5[,i]; classes_test5 <- c(classes_test5, class(v))}
test5[,which(classes_test5 == "factor")] <- apply(test5[,which(classes_test5 == "factor")],
                                                  FUN = Cate2Prob,
                                                  MARGIN = 2)

# Impute PersonalField84 and PropertyField29 with -1
Xtt5$PersonalField84[which(is.na(Xtt5$PersonalField84))] <- -1
Xtt5$PropertyField29[which(is.na(Xtt5$PropertyField29))] <- -1
test5$PersonalField84[which(is.na(test5$PersonalField84))] <- -1
test5$PropertyField29[which(is.na(test5$PropertyField29))] <- -1

# Count number of -1's
Xtt5$NumMinus1 <- apply(Xtt5, MARGIN = 1, FUN = function(r){return(length(which(r == -1)))})
test5$NumMinus1 <- apply(test5, MARGIN = 1, FUN = function(r){return(length(which(r == -1)))})

# Count number of zeroes
Xtt5$NumZeroes <- apply(Xtt5, MARGIN = 1, FUN = function(r){return(length(which(r == 0)))})
test5$NumZeroes <- apply(test5, MARGIN = 1, FUN = function(r){return(length(which(r == 0)))})


# make.names
Xtt5$QuoteConversion_Flag <- factor(make.names(Xtt5$QuoteConversion_Flag))

# Temporal variables
Xtt5$Quarters  <- quarters(Xtt5$Original_Quote_Date)
test5$Quarters <- quarters(test5$Original_Quote_Date)
Xtt5$Day       <- weekdays(Xtt5$Original_Quote_Date)
test5$Day      <- weekdays(test5$Original_Quote_Date)
Xtt5$Month     <- months(Xtt5$Original_Quote_Date)
test5$Month    <- months(test5$Original_Quote_Date)
Xtt5$Year      <- as.numeric(format(Xtt5$Original_Quote_Date, "%Y"))
test5$Year     <- as.numeric(format(test5$Original_Quote_Date, "%Y"))

dim(Xtt5)
dim(test5)
setdiff(colnames(Xtt5), colnames(test5))

# combine the dataset for one hot encoding
y <- Xtt5$QuoteConversion_Flag
Xtt5 <- subset(Xtt5, select = -QuoteConversion_Flag)
Xtt5_test5 <- rbind(Xtt5, test5)

Xtt5_test5 <- OneHotEncode(Xtt5_test5, type = "test")

# Split back
Xtt5 <- cbind(y, Xtt5_test5[(1:length(y)),])
colnames(Xtt5)[1] <- "QuoteConversion_Flag"
test5 <- Xtt5_test5[((length(y) + 1): nrow(Xtt5_test5)),]

dim(Xtt5)
dim(test5)
setdiff(colnames(Xtt5), colnames(test5))

# Drop the dates
Xtt5 <- subset(Xtt5, select = -Original_Quote_Date)
test5 <- subset(test5, select = -Original_Quote_Date)

save(list = "Xtt5", file = "data/Xtt5.RData")
save(list = "test5", file = "data/test5.RData")

# Prep for CV
set.seed(86647)
tr_idx <- createDataPartition(Xtt5$QuoteConversion_Flag, p = 0.7, list = FALSE)
Xtrain5 <- Xtt5[ tr_idx,]
Xtest5  <- Xtt5[-tr_idx,]

Xtrain5_small <- Xtrain5[1:(nrow(Xtrain5)/10),]
Xtrain5_small_down <- downSample(x = subset(Xtrain5_small, select = -QuoteConversion_Flag),
                                 y = Xtrain5_small$QuoteConversion_Flag,
                                 list = FALSE,
                                 yname = "QuoteConversion_Flag")

save(list = "Xtrain5", file = "data/Xtrain5.RData")
save(list = "Xtest5", file = "data/Xtest5.RData")
save(list = "Xtrain5_small", file = "data/Xtrain5_small.RData")
save(list = "Xtrain5_small_down", file = "data/Xtrain5_small_down.RData")



