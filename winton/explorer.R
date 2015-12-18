
setwd("/Users/bikash/repos/kaggleCompetition1/winton")
library(ggplot2)
library(dplyr)
library(reshape2)

# In all scripts, training set is called 'train' and test set it called 'test'
# The point of these if blocks are to avoid having to read the csv files repeatedly since that
# is somewhat time consuming, the var 'current.train' basically just tracks what data set
# is represented currently by 'train'


train <- read.csv("data/train.csv")
test <- read.csv("data/test.csv")


numNa <- integer()
numUnique <- integer()
for(i in 1:ncol(train)){
  numNa <- append(numNa, sum(is.na(train[,i])))
  numUnique <- append(numUnique, length(unique(train[,i])))
}

preProc <- preProcess(predictors, method = "pca", thresh = .99)

# Histograms
# Use BoxCox transformation for skewed predictors
df.plot <- train[,2:26]
df.plot <- melt(df.plot)
df.plot.temp <- df.plot[which(df.plot$variable %in% unique(df.plot$variable)[1:5]),] #First 5 features
qplot(value, data = df.plot.temp, facets = variable ~ .)

df.plot.temp <- df.plot[which(df.plot$variable %in% unique(df.plot$variable)[6]),] #6 continuous skewed right
qplot(value, data = df.plot.temp)

df.plot.temp <- df.plot[which(df.plot$variable %in% unique(df.plot$variable)[7]),] #7 continuous, uniform?
qplot(value, data = df.plot.temp)

df.plot.temp <- df.plot[which(df.plot$variable %in% unique(df.plot$variable)[8]),] #8 maybe convert this to factor
qplot(value, data = df.plot.temp)

df.plot.temp <- df.plot[which(df.plot$variable %in% unique(df.plot$variable)[9]),] #9 continuous
qplot(value, data = df.plot.temp)

df.plot.temp <- df.plot[which(df.plot$variable %in% unique(df.plot$variable)[10]),] #10 discrete
qplot(value, data = df.plot.temp)

df.plot.temp <- df.plot[which(df.plot$variable %in% unique(df.plot$variable)[11]),] #11 continuous skewed left
qplot(value, data = df.plot.temp)

df.plot.temp <- df.plot[which(df.plot$variable %in% unique(df.plot$variable)[12]),] #12 continuous, 
#sticky around whole decimals (i.e. .2 instead of .21)
qplot(value, data = df.plot.temp)

df.plot.temp <- df.plot[which(df.plot$variable %in% unique(df.plot$variable)[13]),] #13 discrete
qplot(value, data = df.plot.temp)

df.plot.temp <- df.plot[which(df.plot$variable %in% unique(df.plot$variable)[14]),] #14 approx normal
qplot(value, data = df.plot.temp)

df.plot.temp <- df.plot[which(df.plot$variable %in% unique(df.plot$variable)[15]),] #15 exp decay?
qplot(value, data = df.plot.temp)

df.plot.temp <- df.plot[which(df.plot$variable %in% unique(df.plot$variable)[16]),] #16 98% == 1, .725% == 2 rest NA
qplot(value, data = df.plot.temp)

df.plot.temp <- df.plot[which(df.plot$variable %in% unique(df.plot$variable)[17]),] #17 bimodal normal(s)
qplot(value, data = df.plot.temp)

df.plot.temp <- df.plot[which(df.plot$variable %in% unique(df.plot$variable)[18]),] #18 non-normal continuous
qplot(value, data = df.plot.temp)

df.plot.temp <- df.plot[which(df.plot$variable %in% unique(df.plot$variable)[19]),] #18 non-normal continuous
qplot(value, data = df.plot.temp)

df.plot.temp <- df.plot[which(df.plot$variable %in% unique(df.plot$variable)[20]),] #20 discrete
qplot(value, data = df.plot.temp)

df.plot.temp <- df.plot[which(df.plot$variable %in% unique(df.plot$variable)[21]),] #21 non-normal continuous
qplot(value, data = df.plot.temp)

df.plot.temp <- df.plot[which(df.plot$variable %in% unique(df.plot$variable)[22]),] #22 non-normal continuous
qplot(value, data = df.plot.temp)

df.plot.temp <- df.plot[which(df.plot$variable %in% unique(df.plot$variable)[23]),] #23 non-normal continuous
qplot(value, data = df.plot.temp)

df.plot.temp <- df.plot[which(df.plot$variable %in% unique(df.plot$variable)[24]),] #24 non-normal continuous
qplot(value, data = df.plot.temp)

df.plot.temp <- df.plot[which(df.plot$variable %in% unique(df.plot$variable)[25]),] #25 bimodal approx normals
qplot(value, data = df.plot.temp)

# Relationships between variables
# dependent var: return at minute 121, first response variable
featurePlot(x = train[,2:5], y = train[,148], plot = "pairs")
featurePlot(x = train[,6:10], y = train[,148], plot = "pairs")
featurePlot(x = train[,11:15], y = train[,148], plot = "pairs")
featurePlot(x = train[,16:20], y = train[,148], plot = "pairs")
featurePlot(x = train[,21:26], y = train[,148], plot = "pairs")

# dep var: ret_plusone, the overall return from the next day
featurePlot(x = train[,2:26], y = train[,208], plot = "scatter")