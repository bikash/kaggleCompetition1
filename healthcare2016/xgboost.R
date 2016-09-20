
require(xgboost)
require(methods)
library(data.table)
library(tm)

setwd("/home/bikash/repos/kaggleCompetition1/healthcare2016")
#train = read.csv('data/train.tsv',header=TRUE,stringsAsFactors = F)
#test = read.csv('data/test.tsv',stringsAsFactors = F)

test <- fread("data/test.tsv")
train <- fread("data/train.tsv")

test_id <- test$ID
test$ID<-NULL
test$Category <-NA
#combine data set
combi <- rbind(train, test)



#install package
library(tm)
#create corpus
corpus <- Corpus(VectorSource(combi$Question))

#Convert text to lowercase
corpus <- tm_map(corpus, tolower)
corpus[[1]]

#Remove Punctuation
corpus <- tm_map(corpus, removePunctuation)
corpus[[1]]

#Remove Stopwords
corpus <- tm_map(corpus, removeWords, c(stopwords('english')))
corpus[[1]]

#Remove Whitespaces
corpus <- tm_map(corpus, stripWhitespace)
corpus[[1]]

# Perform Stemming
corpus <- tm_map(corpus, stemDocument)
corpus[[1]]

 #After we are done with pre-processing, it is necessary to convert the text into plain text document. This helps in pre-processing documents as text documents.
corpus <- tm_map(corpus, PlainTextDocument)


#For further processing, we’ll create a document matrix where the text will categorized in columns
#document matrix
frequencies <- DocumentTermMatrix(corpus) 
frequencies

#Step 6. Data Exploration
freq <- colSums(as.matrix(frequencies))
length(freq)

ord <- order(freq)
ord

#if you wish to export the matrix (to see how it looks) to an excel file
m <- as.matrix(frequencies)
dim(m) write.csv(m, file = 'matrix.csv')

#check most and least frequent words
freq[head(ord)]
freq[tail(ord)]

#check our table of 20 frequencies
head(table(freq),20)
tail(table(freq),20)

#Hence, I’ll remove only the terms having frequency less than 3

#remove sparse terms
sparse <- removeSparseTerms(frequencies, 1 - 3/nrow(frequencies))
dim(sparse)

# Let’s visualize the data now. But first, we’ll create a data frame.

#create a data frame for visualization
wf <- data.frame(word = names(freq), freq = freq)
head(wf)
#plot terms which appear atleast 10,000 times
library(ggplot2)
chart <- ggplot(subset(wf, freq >10000), aes(x = word, y = freq))
chart <- chart + geom_bar(stat = 'identity', color = 'black', fill = 'white')
chart <- chart + theme(axis.text.x=element_text(angle=45, hjust=1))
chart


#We can also create a word cloud to check the most frequent terms. It is easy to build and gives an enhanced understanding of ingredients in this data. 

#create wordcloud
library(wordcloud)
set.seed(1742)
#plot word cloud
wordcloud(names(freq), freq, min.freq = 2500, scale = c(6, .1), colors = brewer.pal(4, "BuPu"))

#plot 5000 most used words
wordcloud(names(freq), freq, max.words = 5000, scale = c(6, .1), colors = brewer.pal(6, 'Dark2'))


#Now I’ll make final structural changes in the data.

#create sparse as data frame
newsparse <- as.data.frame(as.matrix(sparse))
dim(newsparse)
#check if all words are appropriate
colnames(newsparse) <- make.names(colnames(newsparse))
#check for the dominant dependent variable
table(train$Question)


#add cuisine
newsparse$dominant <- as.factor(c(train$dominant, rep('italian', nrow(test))))
#split data 
mytrain <- newsparse[1:nrow(train),]
mytest <- newsparse[-(1:nrow(train)),]













train = train[,-1]
test = test[,-1]

y = train[,ncol(train)]
y = gsub('Class_','',y)
y = as.integer(y)-1 #xgboost take features in [0,numOfClass)

x = rbind(train[,-ncol(train)],test)
x = as.matrix(x)
x = matrix(as.numeric(x),nrow(x),ncol(x))
trind = 1:length(y)
teind = (nrow(train)+1):nrow(x)

# Set necessary parameter
param <- list("objective" = "multi:softprob",
              "eval_metric" = "mlogloss",
              "num_class" = 9,
              "nthread" = 8)

# Run Cross Valication
cv.nround = 50
bst.cv = xgb.cv(param=param, data = x[trind,], label = y, 
                nfold = 3, nrounds=cv.nround)

# Train the model
nround = 50
bst = xgboost(param=param, data = x[trind,], label = y, nrounds=nround)

# Make prediction
pred = predict(bst,x[teind,])
pred = matrix(pred,9,length(pred)/9)
pred = t(pred)

# Output submission
pred = format(pred, digits=2,scientific=F) # shrink the size of submission
pred = data.frame(1:nrow(pred),pred)
names(pred) = c('id', paste0('Class_',1:9))
write.csv(pred,file='submission.csv', quote=FALSE,row.names=FALSE)