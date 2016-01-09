library(TSP)
library(dplyr)
library(aspace)
library (data.table)
library(readr)
library(fpc)

setwd("/Users/bikash/repos/kaggleCompetition1/santa")

AVG_EARTH_RADIUS = 6371

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
goodClusters <- read.csv("data/try_again.csv")

tspTypes = c('nn', 'nn')
numberOfTours = length(tspTypes)
AllTSPSubmissions <- c()
for(i in 1:numberOfTours) {
  AllTSPSubmissions[[i]] <- data.frame(GiftId=integer(0), TripId=integer(0))
}

for (i in unique(goodClusters$TripId)) {
  TripGifts <- goodClusters$GiftId[goodClusters$TripId==i]
  clusterTrip <- gifts[TripGifts, ]
  distMatrix <- dist(clusterTrip[, c('Longitude', 'Latitude')])
  atsp <- TSP(distMatrix, labels=clusterTrip$GiftId)
  
  for(tourNum in 1:length(tspTypes)) {
    
    # low gift trips
    if(nrow(clusterTrip) <= 2 &&
         nrow(clusterTrip) == tourNum) {
      TSPTour <- data.frame(GiftId = clusterTrip$GiftId,
                            TripId = i)
      AllTSPSubmissions[[tourNum]] <- rbind(AllTSPSubmissions[[tourNum]], TSPTour)
    }
    else if(nrow(clusterTrip) >= tourNum) {
      if(tourNum == 1) {
        tour_atsp <- solve_TSP(atsp, method=tspTypes[tourNum], control = c(start = 1))
      }
      else if(tourNum == 2) {
        tour_atsp <- solve_TSP(atsp, method=tspTypes[tourNum], two_opt=TRUE, control = c(start = 1))
      }
      
      TSPTour <- data.frame(GiftId=as.integer(labels(tour_atsp)), TripId=i)
      AllTSPSubmissions[[tourNum]] <- rbind(AllTSPSubmissions[[tourNum]], TSPTour)
    }
  }
}

AllTSPSubmissions[[numberOfTours + 1]] <- goodClusters

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


allDistances <- merge(tourDistances[[numberOfTours+1]],
                      tourDistances[[1]],
                      by=c('TripID'), all = TRUE,
                      suffixes = c(".Original",".1"))

for(tourNum in 2:numberOfTours) {
  allDistances <- merge(allDistances, tourDistances[[tourNum]],
                        by=c('TripID'), all = TRUE)
  names(allDistances)[tourNum+2] <- paste('WD', tourNum, sep='.')
}

allDistances$Min <- with(allDistances,
                         pmin(WD.Original,
                              WD.1, WD.2, #WD.3, WD.4, WD.5,
                              #WD.6, WD.7, WD.8, WD.9, WD.10,
                              #WD.11, WD.12, WD.13, WD.14, WD.15,
                              #WD.16, WD.17, WD.18, WD.19, WD.20,
                              na.rm=TRUE))

allDistances$TSPDiff = allDistances$WD.Original - allDistances$Min

print(paste('TSP Improvement:', sum(allDistances$TSPDiff)))

submission <- data.frame()
for (i in 1:nrow(allDistances)) {
  row <- allDistances[i, ]
  
  if(row$Min == row$WD.Original) {
    submission <- rbind(submission,
                        goodClusters[goodClusters$TripId == row$TripID, ])
  }
  else {
    for(tourNum in 1:numberOfTours) {
      if(row$Min == row[, tourNum+2]) {
        submission <- rbind(submission,
                            AllTSPSubmissions[[tourNum]][AllTSPSubmissions[[tourNum]]$TripId == row$TripID, ])
        break
      }
    }
  }
}

submissionDistances <- data.frame()
submissionDist=0.0
for (i in unique(goodClusters$TripId)) {
  weightedDist = weighted_trip_length(submission$GiftId[submission$TripId==i])
  submissionDistances <- rbind(submissionDistances, data.frame(TripID= i,WD=weightedDist))
  submissionDist = submissionDist + weightedDist
}
print(submissionDist)

currentScore = 12486506727

if(submissionDist < currentScore) {
  print('IMPROVEMENT')
  write.csv(submission,file="output/high_score.csv",row.names=FALSE)
}else {
  print('Keep trying')
  print(paste('Difference:', submissionDist - currentScore))
}

