require(xgboost)
require(methods)
library(data.table)
library(tm)

setwd("/home/bikash/repos/kaggleCompetition1/healthcare2016")
#train = read.csv('data/train.tsv',header=TRUE,stringsAsFactors = F)
#test = read.csv('data/test.tsv',stringsAsFactors = F)

test <- fread("data/test.csv")
train <- fread("data/train.csv")

# test_id <- test$ID
# test$ID<-NULL
# test$Category <-NA

setdiff(names(train), names(test))

nrow(train)

nrow(test)


unique(train$Question)[1:10]


unique(train$Title)[1:10]

length(setdiff(unique(train$Question), unique(test$Question)))

# The number of product titles that are in both the train and test sets
length(intersect(unique(train$Title), unique(test$Title)))


# The number of product titles that are in both the train and test sets
length(intersect(unique(train$Question), unique(test$Question)))

#combine data set
#combi <- rbind(train, test)

#Now let’s start with some basic text analysis on the queries. First, we’ll create a helper function

# We'll use the library ggvis for data visualization
library(ggvis)
# And the library tm to help with text processing
library(tm)

# Creating a function plot_word_counts to plot counts of word occurences in different sets
plot_word_counts <- function(documents) {
  # Keep only unique documents and convert them to lowercase
  corpus <- Corpus(VectorSource(tolower(unique(documents))))
  # Remove punctuation from the documents
  corpus <- tm_map(corpus, removePunctuation)
  # Remove english stopwords, such as "the" and "a"
  corpus <- tm_map(corpus, removeWords, stopwords("english"))
  
  doc_terms <- DocumentTermMatrix(corpus)
  doc_terms <- as.data.frame(as.matrix(doc_terms))
  word_counts <- data.frame(Words=colnames(doc_terms), Counts=colSums(doc_terms))
  # Sort from the most frequent words to the least frequent words
  word_counts <- word_counts[order(word_counts$Counts, decreasing=TRUE),]
  
  top_words <- word_counts[1:10,]
  top_words$Words <- factor(top_words$Words, levels=top_words$Words)
  
  # Plot the 10 most frequent words with ggvis
  top_words %>%
    ggvis(~Words, ~Counts) %>%
    layer_bars(fill:="#20beff")
}


# The top words in the Title 
plot_word_counts(c(train$Title, test$Title))


# The top words in the Title 
plot_word_counts(sample(c(train$Question, test$Question), 1000))
