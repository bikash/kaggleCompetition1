
library(geosphere)
library(TSP)
library(dplyr)
library(aspace)
library (data.table)
library(readr)
library(fpc)
setwd("/Users/bikash/repos/kaggleCompetition1/santa")
AVG_EARTH_RADIUS = 6371

haversine <- function(lat1,lng1,lat2,lng2, miles=FALSE){
  lat1=as_radians(lat1)
  lat2=as_radians(lat2)
  lng1=as_radians(lng1)
  lng2=as_radians(lng2)
  
  lat = lat2 - lat1
  lng = lng2 - lng1
  d = sin(lat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(lng / 2) ** 2
  h = 2 * AVG_EARTH_RADIUS * asin(sqrt(d))
  if(miles==TRUE) return (h * 0.621371)
  if(miles==FALSE) return (h)
}

base_data <- read.csv("data/gifts.csv")
##calculates total distance for each trip
##function input - vector of gift numbers
weighted_trip_length <- function(trip_gifts) {
  x=data.frame(Latitude=90,Longitude=0,Weight=0,r=1)
  x=rbind(x,data.frame(base_data[trip_gifts,2:4],r=2:(length(trip_gifts)+1)))
  x=rbind(x,data.frame(Latitude=90,Longitude=0,Weight=0,r=length(trip_gifts)+2))
  x=x %>%
    arrange(desc(r)) %>%
    mutate(TotalWeight=cumsum(Weight)+10) %>%
    arrange(r) %>%
    mutate(Latitude_next=lead(Latitude),Longitude_next=lead(Longitude),W=lead(TotalWeight)) %>%
    mutate(Distance=haversine(Latitude,Longitude,Latitude_next,Longitude_next)) %>%
    na.omit()
  x=sum(x$Distance*x$W)
  x
}


gifts <- read.csv("data/gifts.csv")

clstrs = 3600
set.seed(2222)

model <- kmeansCBI(gifts[, 2:3], k=clstrs, iter.max = 10000,
                   scaling=TRUE)

delivery <- data.frame(GiftId = gifts$GiftId, TripId = model$partition - 1)

weightedDelivery <- merge(delivery, gifts)

tripSums <- aggregate(Weight~TripId, data=weightedDelivery, FUN=sum)
nrow(tripSums[tripSums$Weight > 990, ])

weightSubmission <- weightedDelivery[with(weightedDelivery, order(-Weight)), c('GiftId', 'TripId')]
orderedSubmission <- delivery

TSPsubmission <- data.frame(GiftId=integer(0), TripId=integer(0))
TSPSecondSubmission <- data.frame(GiftId=integer(0), TripId=integer(0))
TSPThirdSubmission <- data.frame(GiftId=integer(0), TripId=integer(0))
TSPForthSubmission <- data.frame(GiftId=integer(0), TripId=integer(0))
TSPFifthSubmission <- data.frame(GiftId=integer(0), TripId=integer(0))
TSPSixthSubmission <- data.frame(GiftId=integer(0), TripId=integer(0))
TSPSeventhSubmission <- data.frame(GiftId=integer(0), TripId=integer(0))

for (i in unique(weightSubmission$TripId)) {
  tripGiftsByWeight <- gifts[delivery$TripId==i, ]
  tripGiftsByWeight <- tripGiftsByWeight[with(tripGiftsByWeight, order(-Weight)), ]
  
  if(nrow(gifts[delivery$TripId==i,]) == 1 || nrow(gifts[delivery$TripId==i,]) == 2) {
    TSPTrip <- data.frame(GiftId = tripGiftsByWeight$GiftId,
                          TripId = i)
    TSPsubmission <- rbind(TSPsubmission, TSPTrip)
    TSPSecondSubmission <- rbind(TSPSecondSubmission, TSPTrip)
    
    next
  }
  
  distMatrix <- dist(tripGiftsByWeight[, c('Longitude', 'Latitude')])
  atsp <- TSP(distMatrix, labels=tripGiftsByWeight$GiftId)
  
  tour_atsp <- solve_TSP(atsp, method="nn", control = list(start = 1))
  TSPTrip <- data.frame(GiftId=as.integer(labels(tour_atsp)), TripId=i)
  TSPsubmission <- rbind(TSPsubmission, TSPTrip)
  
  tour_atsp_second <- solve_TSP(atsp, method="nn", control = list(start = 2))
  TSPTripSecond <- data.frame(GiftId=as.integer(labels(tour_atsp_second)), TripId=i)
  TSPSecondSubmission <- rbind(TSPSecondSubmission, TSPTripSecond)
  
  tour_atsp_third <- solve_TSP(atsp, method="nn", control = list(start = 3))
  TSPTripThird <- data.frame(GiftId=as.integer(labels(tour_atsp_third)), TripId=i)
  TSPThirdSubmission <- rbind(TSPThirdSubmission, TSPTripThird)
  
  if(nrow(gifts[delivery$TripId==i,]) > 3) {
    tour_atsp_forth <- solve_TSP(atsp, method="nn", control = list(start = 4))
    TSPTripForth <- data.frame(GiftId=as.integer(labels(tour_atsp_forth)), TripId=i)
    TSPForthSubmission <- rbind(TSPForthSubmission, TSPTripForth)
  }
  
  if(nrow(gifts[delivery$TripId==i,]) > 4) {
    tour_atsp_fifth <- solve_TSP(atsp, method="nn", control = list(start = 5))
    TSPTripFifth <- data.frame(GiftId=as.integer(labels(tour_atsp_fifth)), TripId=i)
    TSPFifthSubmission <- rbind(TSPFifthSubmission, TSPTripFifth)
  }
  
  if(nrow(gifts[delivery$TripId==i,]) > 5) {
    tour_atsp_sixth <- solve_TSP(atsp, method="nn", control = list(start = 6))
    TSPTripSixth <- data.frame(GiftId=as.integer(labels(tour_atsp_sixth)), TripId=i)
    TSPSixthSubmission <- rbind(TSPSixthSubmission, TSPTripSixth)
  }
  
  if(nrow(gifts[delivery$TripId==i,]) > 6) {
    tour_atsp_seventh <- solve_TSP(atsp, method="nn", control = list(start = 7))
    TSPTripSeventh <- data.frame(GiftId=as.integer(labels(tour_atsp_seventh)), TripId=i)
    TSPSeventhSubmission <- rbind(TSPSeventhSubmission, TSPTripSeventh)
  }
}

TSPDistances <- data.frame()
TSPDist=0.0
for (i in unique(TSPsubmission$TripId)) {
  weightedDist = weighted_trip_length(TSPsubmission$GiftId[TSPsubmission$TripId==i])
  
  TSPDistances <- rbind(TSPDistances, data.frame(TripID= i,WD=weightedDist))
  TSPDist = TSPDist + weightedDist
}
print(TSPDist)

TSPSecondDistances <- data.frame()
TSPSecondDist=0.0
for (i in unique(TSPSecondSubmission$TripId)) {
  weightedDist = weighted_trip_length(TSPSecondSubmission$GiftId[TSPSecondSubmission$TripId==i])
  
  TSPSecondDistances <- rbind(TSPSecondDistances, data.frame(TripID= i,WD=weightedDist))
  TSPSecondDist = TSPSecondDist + weightedDist
}
print(TSPSecondDist)

TSPThirdDistances <- data.frame()
TSPThirdDist=0.0
for (i in unique(TSPThirdSubmission$TripId)) {
  weightedDist = weighted_trip_length(TSPThirdSubmission$GiftId[TSPThirdSubmission$TripId==i])
  
  TSPThirdDistances <- rbind(TSPThirdDistances, data.frame(TripID= i,WD=weightedDist))
  TSPThirdDist = TSPThirdDist + weightedDist
}
print(TSPThirdDist)

TSPForthDistances <- data.frame()
TSPForthDist=0.0
for (i in unique(TSPForthSubmission$TripId)) {
  weightedDist = weighted_trip_length(TSPForthSubmission$GiftId[TSPForthSubmission$TripId==i])
  
  TSPForthDistances <- rbind(TSPForthDistances, data.frame(TripID= i,WD=weightedDist))
  TSPForthDist = TSPForthDist + weightedDist
}
print(TSPForthDist)

TSPFifthDistances <- data.frame()
TSPFifthDist=0.0
for (i in unique(TSPFifthSubmission$TripId)) {
  weightedDist = weighted_trip_length(TSPFifthSubmission$GiftId[TSPFifthSubmission$TripId==i])
  
  TSPFifthDistances <- rbind(TSPFifthDistances, data.frame(TripID= i,WD=weightedDist))
  TSPFifthDist = TSPFifthDist + weightedDist
}
print(TSPFifthDist)

TSPSixthDistances <- data.frame()
TSPSixthDist=0.0
for (i in unique(TSPSixthSubmission$TripId)) {
  weightedDist = weighted_trip_length(TSPSixthSubmission$GiftId[TSPSixthSubmission$TripId==i])
  
  TSPSixthDistances <- rbind(TSPSixthDistances, data.frame(TripID= i,WD=weightedDist))
  TSPSixthDist = TSPSixthDist + weightedDist
}
print(TSPSixthDist)

TSPSeventhDistances <- data.frame()
TSPSeventhDist=0.0
for (i in unique(TSPSeventhSubmission$TripId)) {
  weightedDist = weighted_trip_length(TSPSeventhSubmission$GiftId[TSPSeventhSubmission$TripId==i])
  
  TSPSeventhDistances <- rbind(TSPSeventhDistances, data.frame(TripID= i,WD=weightedDist))
  TSPSeventhDist = TSPSeventhDist + weightedDist
}
print(TSPSeventhDist)

weightedDistances <- data.frame()
dist=0.0
for (i in unique(weightSubmission$TripId)) {
  weightedDist = weighted_trip_length(weightSubmission$GiftId[weightSubmission$TripId==i])
  weightedDistances <- rbind(weightedDistances, data.frame(TripID= i,WD=weightedDist))
  dist = dist + weightedDist
}
print(dist)

orderDistances <- data.frame()
orderedDist=0.0
for (i in unique(orderedSubmission$TripId)) {
  weightedDist = weighted_trip_length(orderedSubmission$GiftId[orderedSubmission$TripId==i])
  orderDistances <- rbind(orderDistances, data.frame(TripID= i,WD=weightedDist))
  orderedDist = orderedDist + weightedDist
}
print(orderedDist)

allDistances <- merge(orderDistances, TSPDistances, by=c('TripID'), all = TRUE,
                      suffixes = c(".ordered",".TSP"))
allDistances <- merge(allDistances, TSPSecondDistances, by=c('TripID'),
                      all = TRUE)
names(allDistances)[4] <- 'WD.TSPSecond'
allDistances <- merge(allDistances, TSPThirdDistances, by=c('TripID'),
                      all = TRUE)
names(allDistances)[5] <- 'WD.TSPThird'
allDistances <- merge(allDistances, TSPForthDistances, by=c('TripID'),
                      all = TRUE)
names(allDistances)[6] <- 'WD.TSPForth'
allDistances <- merge(allDistances, TSPFifthDistances, by=c('TripID'),
                      all = TRUE)
names(allDistances)[7] <- 'WD.TSPFifth'
allDistances <- merge(allDistances, TSPSixthDistances, by=c('TripID'),
                      all = TRUE)
names(allDistances)[8] <- 'WD.TSPSixth'
allDistances <- merge(allDistances, TSPSeventhDistances, by=c('TripID'),
                      all = TRUE)
names(allDistances)[9] <- 'WD.TSPSeventh'
allDistances <- merge(allDistances, weightedDistances, by=c('TripID'),
                      all = TRUE)
names(allDistances)[10] <- 'WD.weighted'

allDistances$Min <- with(allDistances, pmin(WD.ordered, WD.TSP, WD.TSPSecond,
                                            WD.TSPThird, WD.TSPForth, 
                                            WD.TSPFifth, WD.TSPSixth, 
                                            WD.TSPSeventh, 
                                            WD.weighted, na.rm=TRUE))


submission <- data.frame()
for (i in 1:nrow(allDistances)) {
  row <- allDistances[i, ]
  
  if(row$Min == row$WD.weighted) {
    submission <- rbind(submission,
                        weightSubmission[weightSubmission$TripId == row$TripID, ])
  }
  else if(row$Min == row$WD.ordered){
    submission <- rbind(submission,
                        orderedSubmission[orderedSubmission$TripId == row$TripID, ])
  }
  else if(row$Min == row$WD.TSP) {
    submission <- rbind(submission,
                        TSPsubmission[TSPsubmission$TripId == row$TripID, ])
  }
  else if(row$Min == row$WD.TSPSecond) {
    submission <- rbind(submission, 
                        TSPSecondSubmission[TSPSecondSubmission$TripId == row$TripID, ])
  }
  else if(row$Min == row$WD.TSPThird) {
    submission <- rbind(submission, 
                        TSPThirdSubmission[TSPThirdSubmission$TripId == row$TripID, ])
  }
  else if(row$Min == row$WD.TSPForth) {
    submission <- rbind(submission, 
                        TSPForthSubmission[TSPForthSubmission$TripId == row$TripID, ])
  }
  else if(row$Min == row$WD.TSPFifth) {
    submission <- rbind(submission, 
                        TSPFifthSubmission[TSPFifthSubmission$TripId == row$TripID, ])
  }
  else if(row$Min == row$WD.TSPSixth) {
    submission <- rbind(submission, 
                        TSPSixthSubmission[TSPSixthSubmission$TripId == row$TripID, ])
  }
  else if(row$Min == row$WD.TSPSeventh) {
    submission <- rbind(submission, 
                        TSPSeventhSubmission[TSPSeventhSubmission$TripId == row$TripID, ])
  }
  
}

submissionDistances <- data.frame()
submissionDist=0.0
for (i in unique(TSPsubmission$TripId)) {
  weightedDist = weighted_trip_length(submission$GiftId[submission$TripId==i])
  submissionDistances <- rbind(submissionDistances, data.frame(TripID= i,WD=weightedDist))
  submissionDist = submissionDist + weightedDist
}
print(submissionDist)

write.csv(submission,file="data/try_again.csv",row.names=FALSE)