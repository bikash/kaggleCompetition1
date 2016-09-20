# R script -- for cleaning the raw data
# Input: train.csv, test.csv
# Output: cleaned versions of the data saved 
# Version: 1
# Author: Bikash Agrawal (UiS)


#rm(list=ls())


library(RWeka)
library(stringr)
library(readr)
library(stringdist)
library(tm)
library(qdap)

# ************** CHANGE THIS TO POINT TO RAW DATA ***************
setwd("/home/bikash/repos/kaggleCompetition1/healthcare2016")
ds1.raw <- read_csv("data/train.csv")
ds2.raw <- read_csv("data/test.csv")

# ***************************************************************

ds1.clean <- ds1.raw
ds2.clean <- ds2.raw
id <- ds2.raw$ID

# convert to lower case
ds1.clean$Title <- tolower(ds1.clean$Title)
ds1.clean$Question <- tolower(ds1.clean$Question)
ds2.clean$Title <- tolower(ds2.clean$Title)
ds2.clean$Question <- tolower(ds2.clean$Question)

# replace all punctuation and special characters by space
ds1.clean$Title <- gsub("[ &<>)(_,.;:!?/-]+", " ", ds1.clean$Title)
ds1.clean$Question <- gsub("[ \n\t&<>)(_,.;:!?/-]+", " ", ds1.clean$Question)
ds2.clean$Title <- gsub("[ &<>)(_,.;:!?/-]+", " ", ds2.clean$Title)
ds2.clean$Question <- gsub("[ \n\t&<>)(_,.;:!?/-]+", " ", ds2.clean$Question)

# remove the apostrophe's
ds1.clean$Title <- gsub("'s\\b", "", ds1.clean$Title)
ds1.clean$Question <- gsub("'s\\b", "", ds1.clean$Question)
ds2.clean$Title <- gsub("'s\\b", "", ds2.clean$Title)
ds2.clean$Question <- gsub("'s\\b", "", ds2.clean$Question)

# remove the apostrophe
ds1.clean$Title <- gsub("[']+", "", ds1.clean$Title)
ds1.clean$Question <- gsub("[']+", "", ds1.clean$Question)
ds2.clean$Title <- gsub("[']+", "", ds2.clean$Title)
ds2.clean$Question <- gsub("[']+", "", ds2.clean$Question)

# remove the double quotes
ds1.clean$Title <- gsub("[\"]+", "", ds1.clean$Title)
ds1.clean$Question <- gsub("[\"]+", "", ds1.clean$Question)
ds2.clean$Title <- gsub("[\"]+", "", ds2.clean$Title)
ds2.clean$Question <- gsub("[\"]+", "", ds2.clean$Question)


# stem
pt1.stemmed <-  sapply(ds1.clean$Title, function(x){ paste(stemmer(x, capitalize=FALSE), collapse=" ")  })
pd1.stemmed <-  sapply(ds1.clean$Question, function(x){ paste(stemmer(x, capitalize=FALSE), collapse=" ") })


pt2.stemmed <-  sapply(ds2.clean$Title, function(x){ paste(stemmer(x, capitalize=FALSE), collapse=" ")  })
pd2.stemmed <-  sapply(ds2.clean$Question, function(x){ paste(stemmer(x, capitalize=FALSE), collapse=" ") })



ds1.clean$Title <- pt1.stemmed
ds1.clean$Question <-   pd1.stemmed

ds2.clean$Title <- pt2.stemmed
ds2.clean$Question <-   pd2.stemmed



# remove stop words stopwords("english")

ds1.clean$Title <- gsub(paste("(\\b", paste(stopwords("english"), collapse="\\b|\\b"), "\\b)", sep=""), "", ds1.clean$Title)
ds1.clean$Question <- gsub(paste("(\\b", paste(stopwords("english"), collapse="\\b|\\b"), "\\b)", sep=""), "", ds1.clean$Question)


ds2.clean$Title <- gsub(paste("(\\b", paste(stopwords("english"), collapse="\\b|\\b"), "\\b)", sep=""), "", ds2.clean$Title)
ds2.clean$Question <- gsub(paste("(\\b", paste(stopwords("english"), collapse="\\b|\\b"), "\\b)", sep=""), "", ds2.clean$Question)

ds1.clean$Title <- gsub("[ ]+", " ",ds1.clean$Title)
ds1.clean$Question<- gsub("[ ]+", " ", ds1.clean$Question)

ds2.clean$Title <- gsub("[ ]+", " ", ds2.clean$Title)
ds2.clean$Question <- gsub("[ ]+", " ", ds2.clean$Question)


target <- ds1.clean$Category
ds1.clean$ID<- NULL
ds1.clean$Category <- NULL

#target_cat<-ds1.clean$Category
ds1.clean$Category <- NULL
id.test <- ds2.clean$ID
ds2.clean$ID <- NULL

# save in .RData file
save(list=c("ds1.clean", "ds2.clean", "id.test", "target"), file="data/cleanData.RData")