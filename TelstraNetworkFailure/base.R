
library(data.table)
library(magrittr)
library(stringr)
library(ggplot2)
library(doMC)
library(caret)


#
remove0cols <- T
writeFiles <- F
computeFeatures <- T
computeManualInteractions <- T
groupCardinalities <- T

log_feature_rarityThreshold <- 50
event_type_rarity_threshold <- 5
#

setwd("/Users/bikash/repos/kaggleCompetition1/TelstraNetworkFailure")
source('utils.R')

train <- fread("data/train.csv") %>% setkey(id)
test <- fread("data/test.csv") %>% setkey(id)
log_feature <-fread("data/log_feature.csv") %>% setkey(id)
event_type <-fread("data/event_type.csv") %>% setkey(id)
resource_type <-fread("data/resource_type.csv") %>% setkey(id)
severity_type <- fread("data/severity_type.csv") %>% setkey(id)



log_feature[, ":="(numlf = makeNumeric(log_feature), log_feature = makeReadable(log_feature))]
event_type[, ":="(numet = makeNumeric(event_type), event_type = makeReadable(event_type))]
resource_type[, ":="(resource_type = makeReadable(resource_type))]
severity_type[, ":="(numst = makeNumeric(severity_type), severity_type = makeReadable(severity_type))]


train[, ":="(numloc = makeNumeric(location),
             location = makeReadable(location))]
test[, ":="(numloc = makeNumeric(location),
            location = makeReadable(location), fault_severity = -1)]

total <- rbind(train, test)%>%setkey("id")

total <-total[1:1000,]

if(groupCardinalities){
  #LF reduction
  rare_lf <- log_feature[total][,.N, by=log_feature][N<=log_feature_rarityThreshold][,log_feature]
  log_feature$log_feature[log_feature$log_feature%in%rare_lf] <- "rare_lf"
  
  rare_et <- event_type[total][,.N, by=event_type][N<=event_type_rarity_threshold][,event_type]
  event_type$event_type[event_type$event_type%in%rare_et] <- "rare_et"
  rm(list=c("rare_lf", "rare_et"))
}

if(computeFeatures){
  t1 <- log_feature[total][,.(
    loc_nid = unique(id),
    loc_nlf = unique(log_feature),
    loc_sumvol = sum(volume),
    loc_avgvol = mean(volume),
    loc_sdvol = ifelse(is.na(sd(volume)), 0, sd(volume)),
    loc_logvol = log(sum(volume)),
    loc_vollog = sum(log(volume+1)),
    loc_sqrtvol = sqrt(sum(volume)),
    loc_volsqrt = sum(sqrt(volume))), keyby=location]
  
  t2 <- resource_type[total][, .(loc_nrt = unique(resource_type)), keyby=location]
  t3 <- event_type[total][, .(loc_net = unique(event_type)), keyby=location]
  t4 <- severity_type[total][, .(loc_nst = unique(severity_type)), keyby=location]
  
  joined_total <- resource_type[event_type,allow.cartesian=TRUE][severity_type][total]
  # nombres de combinaisons diff??rentes de resource_type/event_type par location
  t5 <- joined_total[,etrtcomb := paste(resource_type, event_type, sep = "x")][, .(loc_etrtcomb = unique(etrtcomb)), keyby=location]
  t6 <- joined_total[,etstcomb := paste(severity_type, event_type, sep = "x")][, .(loc_etstcomb = unique(etstcomb)), keyby=location]
  t7 <- joined_total[,rtstcomb := paste(severity_type, resource_type, sep = "x")][, .(loc_rtstcomb = unique(rtstcomb)), keyby=location]
  
  location_info_total <- t1[t2,allow.cartesian=TRUE][t3,allow.cartesian=TRUE][t4,allow.cartesian=TRUE][t5,allow.cartesian=TRUE][t6,allow.cartesian=TRUE][t7,allow.cartesian=TRUE]
  
  # tests sur les log features en temps que num??rique moyenne/id, somme/id, etc
  lf_info <- log_feature[total][,.(
    sumlf = sum(numlf), 
    avglf = mean(numlf), 
    sdlf = ifelse(is.na(sd(numlf)), 0, sd(numlf)),
    minlf = min(numlf),
    maxlf = max(numlf)),by=id]
  
  # moyenne numerique de l'event_type * severity_type
  etst <- event_type[, .(
    id, 
    minumet = min(numet), 
    maxnumet = max(numet), 
    avgnumet = mean(numet)), by=id][severity_type][total][,.(et.st = avgnumet*numst),by=id]
  
  total <- total[lf_info][etst]
  total <- merge(total, location_info_total, by = "location",allow.cartesian=TRUE)
}

setkeyv(total[,":="(location=NULL)], c("id", "fault_severity"))

# drop NA value
na.ids <- log_feature[total,allow.cartesian=TRUE][fault_severity != -1][is.na(log_feature)][,id]
if(length(na.ids) > 0)
  total <- total[-na.ids]
####################################
# dcast des data pour avoir une ligne par id

# total_lf_volume  <- dcast(
#   log_feature[total],
#   id + fault_severity ~ log_feature,
#   value.var = list("volume", "log_feature"),
#   fun = list(sum, length)
# )
library(reshape2)
# on garde uniquement le volume
total_lf_volume  <- dcast( log_feature[total,allow.cartesian=TRUE],
  id + fault_severity ~ log_feature, value.var = "volume",  fun = sum)


total_et <-
  dcast(event_type[total,allow.cartesian=TRUE], id + fault_severity ~ event_type, value.var = "event_type", fun = length)
total_rt <-
  dcast(
    resource_type[total,allow.cartesian=TRUE], id + fault_severity ~ resource_type, value.var = "resource_type", fun = length
  )
total_st <-
  dcast(
    severity_type[total,allow.cartesian=TRUE], id + fault_severity ~ severity_type, value.var = "severity_type", fun = length
  )

total.wide <- total[total_lf_volume][total_et][total_rt][total_st]

######################################################################################
#calculs d'intercations entre event_type, resource_types, log_features
if(computeManualInteractions){
  total.wide[,":="(
    
    som_vol_test = (f203 * f312 * f232 * f170),
    
    som_vol_feat = (f82 + f203 + f71 + f193 + f80),
    som_vol_feat_c0 = (f313 + f233 + f315),
    som_vol_feat_c1 = (f82 + f203 + f170),
    som_vol_feat_c2 = (f71 + f193 + f80) ) ]
}

######################################################################################

## remove id from data
train.wide <- total.wide[fault_severity != -1,-1, with = FALSE]
test.wide <- total.wide[fault_severity == -1,-1, with = FALSE]



# write files with train and test
if(writeFiles){
  writeLines("Writing train.csv and test.csv...")
  write.csv(train.wide, paste(sep = "-", "train.csv"), row.names = F, quote = F)
  write.csv(test.wide[,.SD, .SDcols = -"fault_severity"], paste(sep = "-", "test.csv"), row.names = F, quote = F)
  writeLines("...done")}

## remove the categorial values loc_nlf, loc_nrt, loc_net, loc_nst, loc_etrtcomb, loc_etstcomb, loc_rtstcomb [10,18,19,20,21,22,23]

## remove fault_severity from train and it is first column so it is -1
xtrain <- as.matrix(train.wide[,-c(1,10,18,19,20,21,22,23), with = F])
names(xtrain) <- setdiff(names(train.wide), "fault_severity")
ytrain <- train.wide$fault_severity

## remove fault_severity from test and it is first column
xtest <-  as.matrix(test.wide[,-c(1,10,18,19,20,21,22,23), with = F])
names(xtest) <- setdiff(names(test.wide), "fault_severity")
test.id <- test$id

#
# folds <- createFolds(train.wide$fault_severity, k = 3)
# 80% train + 20% test
if(onFold) folds <- createDataPartition(train.wide$fault_severity, p = 0.8)


writeLines("Cleaning up...")
#rm(list=c("na.ids", "total.wide", "total_et", "total_rt", "total_st", 
#          "total_lf_volume", "train.wide", "test.wide", "total",
#          "location_info_total","joined_total", "lf_info", paste0("t", 1:7)))

