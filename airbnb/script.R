library(data.table)
library(bit64)
library(ggplot2)
library(sqldf)
library(dplyr)
library(caret)
library(doMC)
library(gbm)
library(xgboost)
library(Matrix)
age_gender<-fread('age_gender_bkts.csv') # use this to assign different weight to different age group and gender group
countries<-fread('countries.csv')  # aggregate to add geographical distance and language distance 
train<-fread('train_users_2.csv',stringsAsFactors = F)
test<-fread('test_users.csv',stringsAsFactors = F)
###############################Clean Data's Format
train<-as.data.frame(train)
test<-as.data.frame(test)
converttime<-function(x){
  x$date_account_created<-as.Date(strptime(x$date_account_created,format = '%Y-%m-%d'))
  x$timestamp_first_active<-strptime(as.character( x$timestamp_first_active),format = '%Y%m%d%H%M%S')
  x$timestamp_first_active<-as.Date(format.Date( x$timestamp_first_active,'%Y-%m-%d'))
  x
}
train_data<-converttime(train)
test_data<-converttime(test)
########################Creat a new variable which calculate the difference between first 2 date cols:
timeint<-function(x){
  x<-mutate(x,time_interval=as.numeric(date_account_created)-as.numeric(timestamp_first_active))
  x
}
train_data<-timeint(train_data)
test_data<-timeint(test_data)
###################################Get the year month day of the first 2 date cols (date_first_booking in test in all NAs therefore I ignored it)
ymd<-function(x){
  colname<-c("date_account_created","timestamp_first_active")
  cols<-x[,colnames(x) %in% colname]
  newdat<-NULL
  for (y in 1:ncol(cols)){
    newdat<-cbind(newdat,year(cols[,y]))
    colnames(newdat)[ncol(newdat)]<- paste(colname[y],'year',sep = '_')
    newdat<-cbind(newdat,month(cols[,y]))
    colnames(newdat)[ncol(newdat)]<- paste(colname[y],'month',sep = '_')
    newdat<-cbind(newdat,mday(cols[,y]))
    colnames(newdat)[ncol(newdat)]<- paste(colname[y],'day',sep = '_')
    newdat<-cbind(newdat,wday(cols[,y]))
    colnames(newdat)[ncol(newdat)]<- paste(colname[y],'weekday',sep = '_')
  }
  newdat
}
train_data<-cbind(train_data,ymd(train_data))
test_data<-cbind(test_data,ymd(test_data))
#get rid of old cols:
train_data<-select(train_data,-date_account_created,-timestamp_first_active,-date_first_booking)
test_data<-select(test_data,-date_account_created,-timestamp_first_active,-date_first_booking)
##rearrange train data
train_data<-as.data.frame(train_data[,c(1:12,14:22,13)])
#age: unknown to -1/0
train_data[is.na(train_data$age),]$age <- -1
test_data[is.na(test_data$age),]$age <- -1
train_data[train_data$age>=100,]$age <- -1
test_data[test_data$age>=100,]$age <- -1
###########further clean several cols
train_data[train_data$first_affiliate_tracked=='',]$first_affiliate_tracked <- '-unkown-'
test_data[test_data$first_affiliate_tracked=='',]$first_affiliate_tracked <- '-unkown-'
#### Processing and aggregate the session data
session<-fread('sessions.csv',stringsAsFactors = F)
###cleanup some values 
session[session$action=='',]$action <- '-unkown-'
session<-filter(session,user_id!='')
session[session$action_type=='',]$action_type <- '-unknown-'
session[session$action_detail=='',]$action_detail <- '-unknown-'
# because NA values cannot be assigned to -1, or affect total time ,assign to zero
session[is.na(session$secs_elapsed),]$secs_elapsed <- 0
#aggregate all vars to see the total sec of each combination
agg<-sqldf('select user_id,action_type,device_type,action_detail,  sum(secs_elapsed) as totalsec from session group by user_id,action_type,device_type,action_detail')
agg<-agg[agg$action_type!='-unknown-',]
agg<-agg[agg$device_type!='-unknown-',]
agg<-agg[agg$action_detail!='-unknown-',]
agg<-filter(agg,totalsec!=0)
agg$new_action<-apply(agg[,2:4],1,function(x) paste(x,collapse = '-'))
agg<-agg[,-(2:4)]
agg<-agg[,c(1,3,2)]
agg_ag<-reshape(agg,v.names = 'totalsec' ,timevar ='new_action' ,idvar = 'user_id',direction = 'wide')    
# impute NAs as 0
agg_ag[is.na(agg_ag)] <- 0
#combine the session_ag to train_data and test_data
train_data<-left_join(train_data,agg_ag,by = c('id'='user_id'))
test_data<-left_join(test_data,agg_ag,by = c('id'='user_id'))
#clean up the id cols 
tarcol<-which(names(train_data)=='country_destination')
train_data<-train_data[,c(1:(tarcol-1),(tarcol+1):ncol(train_data),tarcol)]
train_data[,22:ncol(train_data)][is.na(train_data[,22:ncol(train_data)])] <- 0
test_data[,22:ncol(test_data)][is.na(test_data[,22:ncol(test_data)])] <- 0
###add another aggregation from session into train and test:
#action
action_ag<-sqldf('select user_id, action, count(*) as total from session group by user_id,action')
action_ag<-reshape(action_ag,v.names = 'total' ,timevar ='action' ,idvar = 'user_id',direction = 'wide')
#action_type
actiontype_ag<-sqldf('select user_id, action_type, count(*) as total from session group by user_id,action_type')
actiontype_ag<-reshape(actiontype_ag,v.names = 'total' ,timevar ='action_type' ,idvar = 'user_id',direction = 'wide')
#action_details
actiondetail_ag<-sqldf('select user_id, action_detail, count(*) as total from session group by user_id,action_detail')
actiondetail_ag<-reshape(actiondetail_ag,v.names = 'total' ,timevar ='action_detail' ,idvar = 'user_id',direction = 'wide')
#device_type
device_ag<-sqldf('select user_id, device_type, count(*) as total from session group by user_id,device_type')
device_ag<-reshape(device_ag,v.names = 'total' ,timevar ='device_type' ,idvar = 'user_id',direction = 'wide')
#sec_elapsed
sec_ag<-sqldf('select user_id, sum(secs_elapsed) as totalsec from session group by user_id')
#join these three dataset:
session_ag<-full_join(actiontype_ag,actiondetail_ag,by = c('user_id'='user_id'))
session_ag<-full_join(session_ag,device_ag,by=c('user_id'='user_id'))
session_ag<-full_join(session_ag,action_ag,by=c('user_id'='user_id'))
session_ag<-full_join(session_ag,sec_ag,by=c('user_id'='user_id'))
# impute NAs as 0
session_ag[is.na(session_ag)] <- 0
#combine the session_ag to train_data and test_data
train_data<-left_join(train_data,session_ag,by = c('id'='user_id'))
test_data<-left_join(test_data,session_ag,by = c('id'='user_id'))
#####
train_data<-select(train_data,-id)
train_data[is.na(train_data)] <- 0
test_data<-select(test_data,-id)
test_data[is.na(test_data)] <- 0
##Data order:
train_data<-train_data[,c(1,3,5,6,7,8,9,10,11,2,4,12:ncol(train_data))]
test_data<-test_data[,c(1,3,5,6,7,8,9,10,11,2,4,12:ncol(test_data))]
target<-which(names(train_data)=='country_destination')
names(train_data)[-target] <-sapply(c(1:(ncol(train_data)-1)),function(x) paste('col',x,sep=''))
names(test_data) <-sapply(c(1:ncol(test_data)),function(x) paste('col',x,sep=''))
#dummify some cols:
dmy_train<-dummyVars(~.,data=train_data[,1:9])
newcol_train<-predict(dmy_train,train_data[,1:9])
train_data<-cbind(train_data[,-c(1:9)],newcol_train)
target<-which(names(train_data)=='country_destination')
train_data<-train_data[,c(1:(target-1),(target+1):ncol(train_data),target)]
dmy_test<-dummyVars(~.,data=test_data[,1:9])
newcol_test<-predict(dmy_test,test_data[,1:9])
test_data<-cbind(test_data[,-c(1:9)],newcol_test)
#remove punctuation in names
names(train_data) <-sapply(names(train_data),function(x) gsub('[[:punct:]]','',x))
names(test_data) <-sapply(names(test_data),function(x) gsub('[[:punct:]]','',x))
names(train_data) <-sapply(names(train_data),function(x) gsub(' ','',x))
names(test_data) <-sapply(names(test_data),function(x) gsub(' ','',x))
#dummify target vars for xgboost (make sure for xgboost set the first from 0 instead of 1)
train_data2<-train_data
test_data2<-test_data
train_data2[train_data2$countrydestination=='NDF',]$countrydestination <- 0
train_data2[train_data2$countrydestination=='US',]$countrydestination <-1
train_data2[train_data2$countrydestination=='other',]$countrydestination <-2
train_data2[train_data2$countrydestination=='FR',]$countrydestination <-3
train_data2[train_data2$countrydestination=='CA',]$countrydestination <-4
train_data2[train_data2$countrydestination=='GB',]$countrydestination <-5
train_data2[train_data2$countrydestination=='ES',]$countrydestination <-6
train_data2[train_data2$countrydestination=='IT',]$countrydestination <-7
train_data2[train_data2$countrydestination=='PT',]$countrydestination <-8
train_data2[train_data2$countrydestination=='NL',]$countrydestination <-9
train_data2[train_data2$countrydestination=='DE',]$countrydestination <-10
train_data2[train_data2$countrydestination=='AU',]$countrydestination <-11
###########
###########
###Train###
###########
###########
###########
train_data2<-as.data.frame(train_data2,stringsAsFactors = F)
test_data2<-as.data.frame(test_data2,stringsAsFactors = F)
#############################################################
#############################################################
#############################################################
#############################################################
#############################################################
#############################################################
#############################################################

# which var imp 
dtrainlb<-train_data2$countrydestination
dtrain <- train_data2[,-ncol(train_data2)]
col_names<-colnames(dtrain)
list_of_features = paste0(col_names, collapse=' ', sep=' + ')
list_of_features = substr(list_of_features, 1, nchar(list_of_features)-3)
xgb_formula = paste0('countrydestination ~ ', list_of_features)
xgb_formula = as.formula(xgb_formula)
trainmat <- sparse.model.matrix(xgb_formula, data = train_data2 ) # one-hot encoding of all variables
param <- list("objective" = "multi:softprob",
              "eval_metric" = "mlogloss",
              "num_class" = 12)
whichvar <-xgboost(data=trainmat,label=train_data2$countrydestination,params = param,subsample=0.6,colsample_bytree =0.4,nround = 100,nthread=6,early.stop.round = 40,maximize=T)
#varimp
mdimp<-xgb.importance(feature_names = trainmat@Dimnames[[2]],model=whichvar)
#selectfeatures:
#200: 0.87777
#250: 0.87800
#300: 0.87847
#303: 0.87841
#304: 0.87870
#305: 0.87860
#306: 0.87788
#309: 0.87803
#310: 0.87831
#315: 0.87858
#325: 0.87827
#375: 0.87796
#395: 0.87829
#400: 0.87840
#404: 0.87876
#405: 0.87890 ----------  best---------- 
#406: 0.87828
#410: 0.87839
#500: 0.87317
#662: 0.87384
goodfeat<-mdimp$Feature[1:405]
#######
bestdata <- train_data2[,which(colnames(train_data2) %in% goodfeat)]
bestlb<-as.numeric(train_data2$countrydestination)
#bestdata<-cbind(bestdata,bestlb)
besttest <- test_data2[,which(colnames(test_data2) %in% goodfeat)]
bestrain<- xgb.DMatrix(data=as.matrix(bestdata),label=bestlb)


#using bestdata to create watchlist
# bestdata2<-cbind(bestdata,bestlb)
# watch<-createDataPartition(bestdata2$bestlb,p = 0.7,list = F)
# ####train and vali
# besttrain<-bestdata2[watch,]
# besttrain_lb<-besttrain$bestlb
# besttrain<-besttrain[,-ncol(besttrain)]
# bestvali <- bestdata2[-watch,]
# bestvali_lb<-bestvali$bestlb
# bestvali <- bestvali[,-ncol(bestvali)]

# best_t <- xgb.DMatrix(data=as.matrix(besttrain),label=as.numeric(besttrain_lb))
# bestvali <- xgb.DMatrix(data=as.matrix(bestvali),label=as.numeric(bestvali_lb))
#------------------------------------------------------------------------------------
#change subsample: best:0.7
#max.depth15,eta0.05,nround100,subsample0.5,colsamplebytree0.6 --0.87798
#max.depth15,eta0.05,nround100,subsample0.6,colsamplebytree0.6 --0.87941 --
#max.depth15,eta0.05,nround100,subsample0.68,colsamplebytree0.6--0.87924
#max.depth15,eta0.05,nround100,subsample0.7,colsamplebytree0.6 --0.87967----best
#max.depth15,eta0.05,nround100,subsample0.73,colsamplebytree0.6--0.87941

# change colsamplebytree:
#max.depth15,eta0.05,nround100,subsample0.7,colsamplebytree0.55--0.87917
#max.depth15,eta0.05,nround100,subsample0.7,colsamplebytree0.59--0.87947
#max.depth15,eta0.05,nround100,subsample0.7,colsamplebytree0.6 --0.87967---
#max.depth15,eta0.05,nround100,subsample0.7,colsamplebytree0.61--0.87882
#max.depth15,eta0.05,nround100,subsample0.7,colsamplebytree0.65--0.87895
#max.depth15,eta0.05,nround100,subsample0.7,colsamplebytree0.70--0.87977----
#max.depth15,eta0.05,nround100,subsample0.7,colsamplebytree0.71--0.87925
#max.depth15,eta0.05,nround100,subsample0.7,colsamplebytree0.8-- 0.87914


#change max_depth
#max.depth13 ,eta0.05,nround100,subsample0.7,colsamplebytree0.7--0.87828
#max.depth14 ,eta0.05,nround100,subsample0.7,colsamplebytree0.7--0.87979----best
#max.depth15,eta0.05,nround100,subsample0.7,colsamplebytree0.7-- 0.87977

#change nround and change eta

#max.depth14 ,eta0.05,nround100,subsample0.7,colsamplebytree0.7--0.87979----best
#max.depth14 ,eta0.05,nround110,subsample0.7,colsamplebytree0.7--0.87926
#max.depth14 ,eta0.04,nround110,subsample0.7,colsamplebytree0.7--0.87960
#max.depth14 ,eta0.04,nround120,subsample0.7,colsamplebytree0.7--0.87977


param <- list(objective = "multi:softprob",
              eval_metric = "mlogloss",
              num_class = 12,
              max_depth = 14,
              eta = 0.04, 
              subsample = 0.7,
              colsample_bytree = 0.7
)
#md <-xgb.cv(data=bestrain,params=param,nround = 150,nthread=6,early.stop.round = 5,nfold = 5)
##################
md <-xgb.train(data=bestrain,params=param,nround = 140,nthread=6)
###
res_test<-predict(md,data.matrix(besttest))
res_data<-matrix(res_test,nrow=12,ncol=62096)
res_data<-as.data.frame(res_data)
row.names(res_data)<-c('NDF','US','other','FR','CA','GB','ES','IT','PT','NL','DE','AU')
res_data<-t(res_data)
new_res<-apply(res_data,1, function(x) names(sort(x,decreasing = T))[1:5] )
last_res<-data.frame(result1=new_res[1,],result2=new_res[2,],result3=new_res[3,],result4=new_res[4,],result5=new_res[5,])
df<-do.call(rbind,lapply(1:nrow(last_res),function(x) t(last_res[x,])))
ids<-data.frame(id1=test$id,id2=test$id,id3=test$id,id4=test$id,id5=test$id)
ids<-do.call(rbind,lapply(1:nrow(ids),function(x) t(ids[x,])))
final<-data.frame(id=ids[,1],country=df[,1])
write.table(final,'submission1669.csv',row.name=FALSE,sep=',')


################################################################################

# 
#         cv.nround = 1000
#         cv.nfold = 5
#         seed.number = sample.int(10000, 1)[[1]]
#         set.seed(seed.number)
#         mdcv <- xgb.cv(data=bestrain, params = param, nthread=6, 
#                        nfold=cv.nfold, nrounds=cv.nround,
#                        verbose = T, early.stop.round=5, maximize=FALSE,stratified = T)
#         
#         min_logloss = min(mdcv[, test.mlogloss.mean])
#         min_logloss_index = which.min(mdcv[, test.mlogloss.mean])
#         
#         if (min_logloss < best_logloss) {
#                 best_logloss = min_logloss
#                 best_logloss_index = min_logloss_index
#                 best_seednumber = seed.number
#                 best_param = param
#         }
# }
################################################################################





