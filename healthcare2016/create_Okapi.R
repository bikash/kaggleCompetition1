#Generate Okapi bm25 socre features for our data
#for info about Okapi bm25 read: https://en.wikipedia.org/wiki/Okapi_BM25
library(RWeka)
library(stringdist)
library(combinat)

#Set working directory where the cleandData and Okapi.R files are located
#setwd('/home/sergio/Documents/Kaggle/CrowdFlower/Data')


#The core score computing script for the create_Okapi Script
#for info about Okapi bm25 read: https://en.wikipedia.org/wiki/Okapi_BM25
Okapi <- function(string1, string2,data_clean,mn_qlength) {

  #number of queries
  N=nrow(data_clean)
    
  #how many times this query happens in the data  
  nq_data<-length(which(data_clean$Title==string1)) 
  
  w1 <-NGramTokenizer(string1, Weka_control(min = 1, max = 1, delimiters = " \\r\\n.?!:"))
  
  #Sum the Score, for each token
  score<-0
  
  IDF=log((N-nq_data+0.5)/(nq_data+0.5))
  idf<-rep(0,length(w1))
  
  k=1.6
  b=0.75
  
  for (i in 1:length(w1)) {
    fq=ntimes(w1[i],string1)/length(w1)
    
    score<-score+(IDF*(fq*(1+k))/(fq+k*(1-b+b*length(w1)/mn_qlength)))
    
  } 

  nq_data<-nq_data/N
  
  out <- c(nq_data, IDF, score)
  return(out)

}

ntimes <- function (string1, string2){
  string1 <- gsub("[ -]", "[ ]?", string1)
  return((nchar(string2)- nchar(gsub(paste("", string1, "", sep=""), "", string2)))/nchar(string1))
}

# cleaned versions of train.csv & test.csv, stemmed, stop words removed
load(file = "data/cleanData.RData")

# Add features to train data
X2 <- data.frame()

qlengths<-rep(0,nrow(ds1.clean))
for (i in 1:nrow(ds1.clean)) {
  wx <-NGramTokenizer(ds1.clean$Title[i], Weka_control(min = 1, max = 1, delimiters = " \\r\\n.?!:"))
  qlengths[i]=length(wx)
}

mn_qlength=mean(qlengths)

for (i in 1:nrow(ds1.clean)) {
  print('go go')
  temp <- Okapi(ds1.clean$Title[i], ds1.clean$Question[i],ds1.clean,mn_qlength)
  X2 <- rbind(X2, temp)
}

names(X2) <- c("nq_data", "IDF", "Okapi")

X2$nq_data<-10*log(1+X2$nq_data)
X2$IDF<-X2$IDF/(X2$IDF[which.max(X2$IDF)])
X2$Okapi<-X2$Okapi/(X2$Okapi[which.max(X2$Okapi)])

# Add features to test data
X3 <- data.frame()
for (i in 1:nrow(ds2.clean)) {
  #print('go go')
  temp <- Okapi(ds2.clean$Title[i], ds2.clean$Question[i],ds2.clean,mn_qlength)
  X3 <- rbind(X3, temp)
}
print('X3 finish')

names(X3) <- c("nq_data", "IDF", "Okapi")

X3$nq_data<-10*log(1+X3$nq_data)
X3$IDF<-X3$IDF/(X3$IDF[which.max(X3$IDF)])
X3$Okapi<-X3$Okapi/(X3$Okapi[which.max(X3$Okapi)])

write.csv(X2, "data/Okapi_train.csv", row.names=FALSE)
write.csv(X3, "data/Okapi_test.csv", row.names=FALSE)