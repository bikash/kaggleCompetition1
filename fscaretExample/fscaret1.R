# warning: could take over an hour to install all models the first time you install the fscaret package
# install.packages("fscaret", dependencies = c("Depends", "Suggests"))

library(fscaret)

# list of models fscaret supports:
data(funcRegPred)
funcRegPred

library(caret)
# list of models caret supports:
names(getModelInfo())

# using dataset from the UCI Machine Learning Repository (http://archive.ics.uci.edu/ml/)
titanicDF <- read.csv('http://math.ucdenver.edu/RTutorial/titanic.txt',sep='\t')

# creating new title feature
titanicDF$Title <- ifelse(grepl('Mr ',titanicDF$Name),'Mr',ifelse(grepl('Mrs ',titanicDF$Name),'Mrs',ifelse(grepl('Miss',titanicDF$Name),'Miss','Nothing')))
titanicDF$Title <- as.factor(titanicDF$Title)

# impute age to remove NAs
titanicDF$Age[is.na(titanicDF$Age)] <- median(titanicDF$Age, na.rm=T)

# reorder data set so target is last column
titanicDF <- titanicDF[c('PClass', 'Age',    'Sex',   'Title', 'Survived')]

# binarize all factors
titanicDummy <- dummyVars("~.",data=titanicDF, fullRank=F)
titanicDF <- as.data.frame(predict(titanicDummy,titanicDF))

# split data set into train and test portion
set.seed(1234)
splitIndex <- createDataPartition(titanicDF$Survived, p = .75, list = FALSE, times = 1)
trainDF <- titanicDF[ splitIndex,]
testDF  <- titanicDF[-splitIndex,]

# limit models to use in ensemble and run fscaret
fsModels <- c("glm", "gbm", "treebag", "ridge", "lasso")
myFS<-fscaret(trainDF, testDF, myTimeLimit = 40, preprocessData=TRUE,
              Used.funcRegPred = fsModels, with.labels=TRUE,
              supress.output=FALSE, no.cores=2)

# analyze results
print(myFS$VarImp)
print(myFS$PPlabels)


#Check the MSE
myFS$VarImp$matrixVarImp.MSE

#We need to do a little wrangling in order to clean this up and get a nicely ordered list with the actual variable names attached:
results <- myFS$VarImp$matrixVarImp.MSE
results$Input_no <- as.numeric(results$Input_no)
results <- results[c("SUM","SUM%","ImpGrad","Input_no")]
myFS$PPlabels$Input_no <-  as.numeric(rownames(myFS$PPlabels))
results <- merge(x=results, y=myFS$PPlabels, by="Input_no", all.x=T)
results <- results[c('Labels', 'SUM')]
results <- subset(results,results$SUM !=0)
results <- results[order(-results$SUM),]
print(results)
