library(TSP)
library(dplyr)
library(aspace)
library (data.table)
library(readr)
library(fpc)

AVG_EARTH_RADIUS = 6371
setwd("/Users/bikash/repos/kaggleCompetition1/santa")
# Set seed
haversine <- function(lat1,lng1,lat2,lng2){
  lat1=as_radians(lat1)
  lat2=as_radians(lat2)
  lng1=as_radians(lng1)
  lng2=as_radians(lng2)
  
  lat = lat2 - lat1
  lng = lng2 - lng1
  d = sin(lat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(lng / 2) ** 2
  h = 2 * AVG_EARTH_RADIUS * asin(sqrt(d))
  return (h)
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

numberOfTours = 20
AllTSPSubmissions <- c()
for(i in 1:numberOfTours) {
  AllTSPSubmissions[[i]] <- data.frame(GiftId=integer(0), TripId=integer(0))
}

for (i in unique(weightSubmission$TripId)) {
  tripGiftsByWeight <- gifts[delivery$TripId==i, ]
  tripGiftsByWeight <- tripGiftsByWeight[with(tripGiftsByWeight, order(-Weight)), ]
  distMatrix <- dist(tripGiftsByWeight[, c('Longitude', 'Latitude')])
  atsp <- TSP(distMatrix, labels=tripGiftsByWeight$GiftId)
  
  for(tourNum in 1:numberOfTours) {
    
    # low gift trips
    if(nrow(gifts[delivery$TripId==i,]) <= 2 && 
         nrow(gifts[delivery$TripId==i,]) == tourNum) {
      TSPTour <- data.frame(GiftId = tripGiftsByWeight$GiftId,
                            TripId = i)
      AllTSPSubmissions[[tourNum]] <- rbind(AllTSPSubmissions[[tourNum]], TSPTour)
    }
    else if(nrow(gifts[delivery$TripId==i,]) >= tourNum) {
      tour_atsp <- solve_TSP(atsp, method="nn", control = list(start = tourNum))
      TSPTour <- data.frame(GiftId=as.integer(labels(tour_atsp)), TripId=i)
      AllTSPSubmissions[[tourNum]] <- rbind(AllTSPSubmissions[[tourNum]], TSPTour)
    }
  }
}

# add GiftId ordered and Weight order submissions
AllTSPSubmissions[[numberOfTours + 1]] <- orderedSubmission
AllTSPSubmissions[[numberOfTours + 2]] <- weightSubmission

tourDistances <- c()
for(tourNum in 1:length(AllTSPSubmissions)) {
  tourDistances[[tourNum]] <- data.frame()
  tourWD = 0.0
  
  for (i in unique(AllTSPSubmissions[[tourNum]]$TripId)) {
    edgeWeightedDist = weighted_trip_length(
      AllTSPSubmissions[[tourNum]]$GiftId[AllTSPSubmissions[[tourNum]]$TripId==i])
    
    tourDistances[[tourNum]] <- rbind(tourDistances[[tourNum]], 
                                      data.frame(TripID= i,WD=edgeWeightedDist))
    tourWD = tourWD + edgeWeightedDist
  }
  
  print(tourWD)
}


allDistances <- merge(tourDistances[[numberOfTours + 1]], 
                      tourDistances[[numberOfTours + 2]], 
                      by=c('TripID'), all = TRUE, 
                      suffixes = c(".ordered",".weighted"))

for(tourNum in 1:numberOfTours) {
  allDistances <- merge(allDistances, tourDistances[[tourNum]], 
                        by=c('TripID'), all = TRUE)
  names(allDistances)[tourNum+3] <- paste('WD', tourNum, sep='.')
}

allDistances$Min <- with(allDistances, 
                         pmin(WD.ordered, WD.weighted, 
                              WD.1, WD.2, WD.3, WD.4, WD.5,
                              WD.6, WD.7, WD.8, WD.9, WD.10,
                              WD.11, WD.12, WD.13, WD.14, WD.15,
                              WD.16, WD.17, WD.18, WD.19, WD.20,
                              na.rm=TRUE))


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
  else {
    for(tourNum in 1:numberOfTours) {
      if(row$Min == row[, tourNum+3]) {
        submission <- rbind(submission,
                            AllTSPSubmissions[[tourNum]][AllTSPSubmissions[[tourNum]]$TripId == row$TripID, ])
        break
      }
    }
  } 
}

submissionDistances <- data.frame()
submissionDist=0.0
for (i in unique(orderedSubmission$TripId)) {
  weightedDist = weighted_trip_length(submission$GiftId[submission$TripId==i])
  submissionDistances <- rbind(submissionDistances, data.frame(TripID= i,WD=weightedDist))
  submissionDist = submissionDist + weightedDist
}
print(submissionDist)

write.csv(submission,file="output/TSP_3600.csv",row.names=FALSE)