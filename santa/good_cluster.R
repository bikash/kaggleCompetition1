setwd("/Users/bikash/repos/kaggleCompetition1/santa")
# Set seed
gifts = read.csv("data/gifts.csv")

library(geosphere)
library(aspace)

# Create longitudinal intervals in degrees
j_step <- 10
gifts$j <- ceil(gifts$Longitude/j_step)*j_step
gifts$TripId <- rep(0,nrow(gifts))

# Shift the interval indices (for tuning)
j_shift <- 35 # degrees
gifts$j <- ifelse(gifts$j+j_shift>180, gifts$j+j_shift-360, gifts$j+j_shift)

# Assign TripId
id <- 0
cost <- 0
chnk <- 94
trips <- gifts[gifts$TripId==0,]
trips <- trips[order(trips$j, trips$Longitude, trips$Latitude),][1:chnk ,]
trips <- trips[order(-trips$Latitude),]

while (sum(gifts$TripId==0)>0) {
  
  g <- numeric()
  id <- id + 1
  wt <- 0
  dst <- 0
  st <- c(0,90)
  
  for (ii in 1:nrow(trips)) {
    if ((wt + trips$Weight[ii])<=1000) {
      wt <- wt +  trips$Weight[ii]
      dst <- dst + distHaversine(st, trips[ii, c("Longitude", "Latitude")], r=6371)
      cost <- cost + dst* trips$Weight[ii]
      st <-  trips[ii, c("Longitude", "Latitude")]
      g <- c(g,trips[ii,1])
    }
  }
  
  dst <- dst + distHaversine(st, c(0,90))/1000
  cost <- cost + dst*10
  gifts$TripId[gifts$GiftId %in% g] <- id
  
  trips <- gifts[gifts$TripId==0,]
  trips <- trips[order(trips$j, trips$Longitude, trips$Latitude),]
  trips <- trips[1:min(chnk ,nrow(trips)),]
  trips <- trips[order(-trips$Latitude),]
  
  print(paste0(sum(gifts$TripId==0), " gifts remaining. score = ", cost/1e9))
  flush.console()
  
}


gifts <- gifts[order(gifts$TripId, -gifts$Latitude), ]
write.csv( data.frame(GiftId=gifts$GiftId,TripId=gifts$TripId), "data/try_again.csv", row.names=FALSE)