library(readr)



setwd("/Users/bikash/repos/kaggleCompetition1/kaggle_walmart_triptype")
# Set seed
set.seed(12345)

twal <- read.table("data/train.csv",sep=',',header = T)
testwal <- read.table("data/test.csv",sep=',',header = T)



######Visualizin data####
table(twal$TripType)
table(twal$FinelineNumber)
table(twal$ScanCount)
table(twal$Upc)
table(twal$DepartmentDescription)
unique(twal$DepartmentDescription)
#####converting separate columns for weekday values####
###traindata####


twal$sun<-0
twal$sun[twal$Weekday=="Sunday"]<-1      

twal$mon<-0
twal$mon[twal$Weekday=="Monday"]<-1  

twal$tue<-0
twal$tue[twal$Weekday=="Tuesday"]<-1  

twal$wed<-0
twal$wed[twal$Weekday=="Wednesday"]<-1

twal$thu<-0
twal$thu[twal$Weekday=="Thursday"]<-1

twal$fri<-0
twal$fri[twal$Weekday=="Friday"]<-1

twal$sat<-0
twal$sat[twal$Weekday=="Saturday"]<-1


###testdata####

testwal$sun<-0
testwal$sun[twal$Weekday=="Sunday"]<-1      

testwal$mon<-0
testwal$mon[twal$Weekday=="Monday"]<-1  

testwal$tue<-0
testwal$tue[twal$Weekday=="Tuesday"]<-1  

testwal$wed<-0
testwal$wed[twal$Weekday=="Wednesday"]<-1

testwal$thu<-0
testwal$thu[twal$Weekday=="Thursday"]<-1

testwal$fri<-0
testwal$fri[twal$Weekday=="Friday"]<-1

testwal$sat<-0
testwal$sat[twal$Weekday=="Saturday"]<-1



#####creating differnt columns for departmnet description####
####test data#####

testwal$HR_PHOT<-0
testwal$HR_PHOT[twal$DepartmentDescription=="1-HR PHOTO"]<-1

testwal$accessories<-0
testwal$accessories[twal$DepartmentDescription=="ACCESSORIES"]<-1

testwal$automotive<-0
testwal$automotive[twal$DepartmentDescription=="AUTOMOTIVE"]<-1

testwal$bakery<-0
testwal$bakery[twal$DepartmentDescription=="BAKERY"]<-1

testwal$bath_shower<-0
testwal$bath_shower[twal$DepartmentDescription=="BATH AND SHOWER"]<-1

testwal$beauty<-0
testwal$beauty[twal$DepartmentDescription=="BEAUTY"]<-1

testwal$bedding<-0
testwal$bedding[twal$DepartmentDescription=="BEDDING"]<-1


testwal$books_magzines<-0
testwal$books_magzines[twal$DepartmentDescription=="BOOKS AND MAGAZINES"]<-1

testwal$boyswear<-0
testwal$boyswear[twal$DepartmentDescription=="BOYS WEAR"]<-1

testwal$bras<-0
testwal$bras[twal$DepartmentDescription=="BRAS & SHAPEWEAR"]<-1

testwal$camerassupplies<-0
testwal$camerassupplies[twal$DepartmentDescription=="CAMERAS AND SUPPLIES"]<-1

testwal$candytobacco<-0
testwal$candytobacco[twal$DepartmentDescription=="CANDY, TOBACCO, COOKIES"]<-1

testwal$celebration<-0
testwal$celebration[twal$DepartmentDescription=="CELEBRATION"]<-1

testwal$commbread<-0
testwal$commbread[twal$DepartmentDescription=="COMM BREAD"]<-1

testwal$conceptstores<-0
testwal$conceptstores[twal$DepartmentDescription=="CONCEPT STORES"]<-1

testwal$cookanddine<-0
testwal$cookanddine[twal$DepartmentDescription=="COOK AND DINE"]<-1

testwal$dairy<-0
testwal$dairy[twal$DepartmentDescription=="DAIRY"]<-1

testwal$dsdgrocery<-0
testwal$dsdgrocery[twal$DepartmentDescription=="DSD GROCERY"]<-1

testwal$electronics<-0
testwal$electronics[twal$DepartmentDescription=="ELECTRONICS"]<-1

testwal$fabrics<-0
testwal$fabrics[twal$DepartmentDescription=="FABRICS AND CRAFTS"]<-1

testwal$financial<-0
testwal$financial[twal$DepartmentDescription=="FINANCIAL SERVICES"]<-1

testwal$frozenfoods<-0
testwal$frozenfoods[twal$DepartmentDescription=="FROZEN FOODS"]<-1

testwal$furntiure<-0
testwal$furntiure[twal$DepartmentDescription=="FURNITURE"]<-1

testwal$girls_wear<-0
testwal$girls_wear[twal$DepartmentDescription=="GIRLS WEAR, 4-6X  AND 7-14"]<-1

testwal$grocerydrygoods<-0
testwal$grocerydrygoods[twal$DepartmentDescription=="GROCERY DRY GOODS"]<-1

testwal$hardware<-0
testwal$hardware[twal$DepartmentDescription=="HARDWARE"]<-1

testwal$healthandbeauty<-0
testwal$healthandbeauty[twal$DepartmentDescription=="HEALTH AND BEAUTY AIDS"]<-1

testwal$homedecor<-0
testwal$homedecor[twal$DepartmentDescription=="HOME DECOR"]<-1

testwal$homemana<-0
testwal$homemana[twal$DepartmentDescription=="HOME MANAGEMENT"]<-1

testwal$horticuture<-0
testwal$horticuture[twal$DepartmentDescription=="HORTICULTURE AND ACCESS"]<-1

testwal$housechemical<-0
testwal$housechemical[twal$DepartmentDescription=="HOUSEHOLD CHEMICALS/SUPP"]<-1

testwal$household<-0
testwal$household[twal$DepartmentDescription=="HOUSEHOLD PAPER GOODS"]<-1

testwal$impulse<-0
testwal$impulse[twal$DepartmentDescription=="IMPULSE MERCHANDISE"]<-1

testwal$infantappearl<-0
testwal$infantappearl[twal$DepartmentDescription=="INFANT APPARE"]<-1

testwal$infant<-0
testwal$infant[twal$DepartmentDescription=="INFANT CONSUMABLE HARDLINES"]<-1

testwal$jewellery<-0
testwal$jewellery[twal$DepartmentDescription=="JEWELRY AND SUNGLASSES"]<-1

testwal$ladiessocks<-0
testwal$ladiessocks[twal$DepartmentDescription=="LADIES SOCKS"]<-1

testwal$ladieswear<-0
testwal$ladieswear[twal$DepartmentDescription=="LADIESWEAR"]<-1

testwal$lhousegoods<-0
testwal$lhousegoods[twal$DepartmentDescription=="LARGE HOUSEHOLD GOODS"]<-1

testwal$lawn<-0
testwal$lawn[twal$DepartmentDescription=="LAWN AND GARDEN"]<-1

testwal$lwb<-0
testwal$lwb[twal$DepartmentDescription=="LIQUOR,WINE,BEER"]<-1
testwal$meat_fresh<-0
testwal$meat_fresh[twal$DepartmentDescription=="MEAT - FRESH & FROZEN"]<-1
testwal$media_gaming<-0
testwal$media_gaming[twal$DepartmentDescription=="MEDIA AND GAMING"]<-1

testwal$mens_wear<-0
testwal$mens_wear[twal$DepartmentDescription=="MENS WEAR"]<-1

testwal$menswear<-0
testwal$menswear[twal$DepartmentDescription=="MENSWEAR"]<-1

testwal$nUll<-0
testwal$nUll[twal$DepartmentDescription=="NULL"]<-1

testwal$officesupplies<-0
testwal$officesupplies[twal$DepartmentDescription=="OFFICE SUPPLIES"]<-1

testwal$opticalframe<-0
testwal$opticalframe[twal$DepartmentDescription=="OPTICAL - FRAMES"]<-1

testwal$opticallenses<-0
testwal$opticallenses[twal$DepartmentDescription=="OPTICAL - LENSES"]<-1

testwal$otherdepa<-0
testwal$otherdepa[twal$DepartmentDescription=="OTHER DEPARTMENTS"]<-1

testwal$paint<-0
testwal$paint[twal$DepartmentDescription=="PAINT AND ACCESSORIES"]<-1

testwal$personalcare<-0
testwal$personalcare[twal$DepartmentDescription=="PERSONAL CARE"]<-1

testwal$petssupplies<-0
testwal$petssupplies[twal$DepartmentDescription=="PETS AND SUPPLIES"]<-1

testwal$pharmacyotc<-0
testwal$pharmacyotc[twal$DepartmentDescription=="PHARMACY OTC"]<-1

testwal$pharmacyrx<-0
testwal$pharmacyrx[twal$DepartmentDescription=="PHARMACY RX"]<-1

testwal$players<-0
testwal$players[twal$DepartmentDescription=="PLAYERS AND ELECTRONICS"]<-1

testwal$plus_maternity<-0
testwal$plus_maternity[twal$DepartmentDescription=="PLUS AND MATERNITY"]<-1

testwal$pre_packed<-0
testwal$pre_packed[twal$DepartmentDescription=="PRE PACKED DELI"]<-1

testwal$produce<-0
testwal$produce[twal$DepartmentDescription=="PRODUCE"]<-1

testwal$seafood<-0
testwal$seafood[twal$DepartmentDescription=="SEAFOOD"]<-1


testwal$seasonal<-0
testwal$seasonal[twal$DepartmentDescription=="SEASONAL"]<-1


testwal$service<-0
testwal$service[twal$DepartmentDescription=="SERVICE DELI"]<-1

testwal$sheer<-0
testwal$sheer[twal$DepartmentDescription=="SHEER HOSIERY"]<-1

testwal$shoes<-0
testwal$shoes[twal$DepartmentDescription=="SHOES"]<-1

testwal$sleepwear<-0
testwal$sleepwear[twal$DepartmentDescription=="SLEEPWEAR/FOUNDATIONS"]<-1

testwal$sportinggoods<-0
testwal$sportinggoods[twal$DepartmentDescription=="SPORTING GOODS"]<-1

testwal$swimwear<-0
testwal$swimwear[twal$DepartmentDescription=="SWIMWEAR/OUTERWEAR"]<-1

testwal$toys<-0
testwal$toys[twal$DepartmentDescription=="TOYS"]<-1

testwal$wireless<-0
testwal$wireless[twal$DepartmentDescription=="WIRELESS"]<-1

#####traindata####

twal$HR_PHOT<-0
twal$HR_PHOT[twal$DepartmentDescription=="1-HR PHOTO"]<-1

twal$accessories<-0
twal$accessories[twal$DepartmentDescription=="ACCESSORIES"]<-1

twal$automotive<-0
twal$automotive[twal$DepartmentDescription=="AUTOMOTIVE"]<-1

twal$bakery<-0
twal$bakery[twal$DepartmentDescription=="BAKERY"]<-1

twal$bath_shower<-0
twal$bath_shower[twal$DepartmentDescription=="BATH AND SHOWER"]<-1

twal$beauty<-0
twal$beauty[twal$DepartmentDescription=="BEAUTY"]<-1

twal$bedding<-0
twal$bedding[twal$DepartmentDescription=="BEDDING"]<-1


twal$books_magzines<-0
twal$books_magzines[twal$DepartmentDescription=="BOOKS AND MAGAZINES"]<-1

twal$boyswear<-0
twal$boyswear[twal$DepartmentDescription=="BOYS WEAR"]<-1

twal$bras<-0
twal$bras[twal$DepartmentDescription=="BRAS & SHAPEWEAR"]<-1

twal$camerassupplies<-0
twal$camerassupplies[twal$DepartmentDescription=="CAMERAS AND SUPPLIES"]<-1

twal$candytobacco<-0
twal$candytobacco[twal$DepartmentDescription=="CANDY, TOBACCO, COOKIES"]<-1

twal$celebration<-0
twal$celebration[twal$DepartmentDescription=="CELEBRATION"]<-1

twal$commbread<-0
twal$commbread[twal$DepartmentDescription=="COMM BREAD"]<-1

twal$conceptstores<-0
twal$conceptstores[twal$DepartmentDescription=="CONCEPT STORES"]<-1

twal$cookanddine<-0
twal$cookanddine[twal$DepartmentDescription=="COOK AND DINE"]<-1

twal$dairy<-0
twal$dairy[twal$DepartmentDescription=="DAIRY"]<-1

twal$dsdgrocery<-0
twal$dsdgrocery[twal$DepartmentDescription=="DSD GROCERY"]<-1

twal$electronics<-0
twal$electronics[twal$DepartmentDescription=="ELECTRONICS"]<-1

twal$fabrics<-0
twal$fabrics[twal$DepartmentDescription=="FABRICS AND CRAFTS"]<-1

twal$financial<-0
twal$financial[twal$DepartmentDescription=="FINANCIAL SERVICES"]<-1

twal$frozenfoods<-0
twal$frozenfoods[twal$DepartmentDescription=="FROZEN FOODS"]<-1

twal$furntiure<-0
twal$furntiure[twal$DepartmentDescription=="FURNITURE"]<-1

twal$girls_wear<-0
twal$girls_wear[twal$DepartmentDescription=="GIRLS WEAR, 4-6X  AND 7-14"]<-1

twal$grocerydrygoods<-0
twal$grocerydrygoods[twal$DepartmentDescription=="GROCERY DRY GOODS"]<-1

twal$hardware<-0
twal$hardware[twal$DepartmentDescription=="HARDWARE"]<-1

twal$healthandbeauty<-0
twal$healthandbeauty[twal$DepartmentDescription=="HEALTH AND BEAUTY AIDS"]<-1

twal$homedecor<-0
twal$homedecor[twal$DepartmentDescription=="HOME DECOR"]<-1

twal$homemana<-0
twal$homemana[twal$DepartmentDescription=="HOME MANAGEMENT"]<-1

twal$horticuture<-0
twal$horticuture[twal$DepartmentDescription=="HORTICULTURE AND ACCESS"]<-1

twal$housechemical<-0
twal$housechemical[twal$DepartmentDescription=="HOUSEHOLD CHEMICALS/SUPP"]<-1

twal$household<-0
twal$household[twal$DepartmentDescription=="HOUSEHOLD PAPER GOODS"]<-1

twal$impulse<-0
twal$impulse[twal$DepartmentDescription=="IMPULSE MERCHANDISE"]<-1

twal$infantappearl<-0
twal$infantappearl[twal$DepartmentDescription=="INFANT APPARE"]<-1

twal$infant<-0
twal$infant[twal$DepartmentDescription=="INFANT CONSUMABLE HARDLINES"]<-1

twal$jewellery<-0
twal$jewellery[twal$DepartmentDescription=="JEWELRY AND SUNGLASSES"]<-1

twal$ladiessocks<-0
twal$ladiessocks[twal$DepartmentDescription=="LADIES SOCKS"]<-1

twal$ladieswear<-0
twal$ladieswear[twal$DepartmentDescription=="LADIESWEAR"]<-1

twal$lhousegoods<-0
twal$lhousegoods[twal$DepartmentDescription=="LARGE HOUSEHOLD GOODS"]<-1

twal$lawn<-0
twal$lawn[twal$DepartmentDescription=="LAWN AND GARDEN"]<-1

twal$lwb<-0
twal$lwb[twal$DepartmentDescription=="LIQUOR,WINE,BEER"]<-1
twal$meat_fresh<-0
twal$meat_fresh[twal$DepartmentDescription=="MEAT - FRESH & FROZEN"]<-1
twal$media_gaming<-0
twal$media_gaming[twal$DepartmentDescription=="MEDIA AND GAMING"]<-1

twal$mens_wear<-0
twal$mens_wear[twal$DepartmentDescription=="MENS WEAR"]<-1

twal$menswear<-0
twal$menswear[twal$DepartmentDescription=="MENSWEAR"]<-1

twal$nUll<-0
twal$nUll[twal$DepartmentDescription=="NULL"]<-1

twal$officesupplies<-0
twal$officesupplies[twal$DepartmentDescription=="OFFICE SUPPLIES"]<-1

twal$opticalframe<-0
twal$opticalframe[twal$DepartmentDescription=="OPTICAL - FRAMES"]<-1

twal$opticallenses<-0
twal$opticallenses[twal$DepartmentDescription=="OPTICAL - LENSES"]<-1

twal$otherdepa<-0
twal$otherdepa[twal$DepartmentDescription=="OTHER DEPARTMENTS"]<-1

twal$paint<-0
twal$paint[twal$DepartmentDescription=="PAINT AND ACCESSORIES"]<-1

twal$personalcare<-0
twal$personalcare[twal$DepartmentDescription=="PERSONAL CARE"]<-1

twal$petssupplies<-0
twal$petssupplies[twal$DepartmentDescription=="PETS AND SUPPLIES"]<-1

twal$pharmacyotc<-0
twal$pharmacyotc[twal$DepartmentDescription=="PHARMACY OTC"]<-1

twal$pharmacyrx<-0
twal$pharmacyrx[twal$DepartmentDescription=="PHARMACY RX"]<-1

twal$players<-0
twal$players[twal$DepartmentDescription=="PLAYERS AND ELECTRONICS"]<-1

twal$plus_maternity<-0
twal$plus_maternity[twal$DepartmentDescription=="PLUS AND MATERNITY"]<-1

twal$pre_packed<-0
twal$pre_packed[twal$DepartmentDescription=="PRE PACKED DELI"]<-1

twal$produce<-0
twal$produce[twal$DepartmentDescription=="PRODUCE"]<-1

twal$seafood<-0
twal$seafood[twal$DepartmentDescription=="SEAFOOD"]<-1


twal$seasonal<-0
twal$seasonal[twal$DepartmentDescription=="SEASONAL"]<-1


twal$service<-0
twal$service[twal$DepartmentDescription=="SERVICE DELI"]<-1

twal$sheer<-0
twal$sheer[twal$DepartmentDescription=="SHEER HOSIERY"]<-1

twal$shoes<-0
twal$shoes[twal$DepartmentDescription=="SHOES"]<-1

twal$sleepwear<-0
twal$sleepwear[twal$DepartmentDescription=="SLEEPWEAR/FOUNDATIONS"]<-1

twal$sportinggoods<-0
twal$sportinggoods[twal$DepartmentDescription=="SPORTING GOODS"]<-1

twal$swimwear<-0
twal$swimwear[twal$DepartmentDescription=="SWIMWEAR/OUTERWEAR"]<-1

twal$toys<-0
twal$toys[twal$DepartmentDescription=="TOYS"]<-1

twal$wireless<-0
twal$wireless[twal$DepartmentDescription=="WIRELESS"]<-1

str(twal)
str(testwal)

#####removing extra columns####


segc1<-grep("Weekday",names(twal))
twal<-twal[,-segc1]

segc2<-grep("DepartmentDescription",names(twal))
twal<-twal[,-segc2]

segc3<-grep("Weekday",names(testwal))
testwal<-testwal[,-segc3]
segc4<-grep("DepartmentDescription",names(testwal))
testwal<-testwal[,-segc4]

ncol(twal)
ncol(testwal)
#####making factors#####

length(unique(twal$DepartmentDescription))
trip<-twal$TripType
twal$TripType<-as.factor(twal$TripType)





###imputing null values#####

twal$Upc[is.na(twal$Upc)]<-1
testwal$Upc[is.na(testwal$Upc)]<-1

twal$FinelineNumber[is.na(twal$FinelineNumber)]<- -1 
testwal$FinelineNumber[is.na(testwal$FinelineNumber)]<- -1



levels <- unique(c(twal$Upc, testwal$Upc))
twal$Upc <- as.integer(factor(twal$Upc, levels=levels))
testwal$Upc<- as.integer(factor(testwal$Upc,  levels=levels))


levels <- unique(c(twal$FinelineNumber, testwal$FinelineNumber))
twal$FinelineNumber <- as.integer(factor(twal$FinelineNumber, levels=levels))
testwal$FinelineNumber<- as.integer(factor(testwal$FinelineNumber,  
                                           levels=levels))




str(twal)
str(testwal)


# Feature engineering 
# Include ReturnCount column
twal$ReturnCount <- -twal$ScanCount
twal$ReturnCount[twal$ReturnCount < 0] <- 0
twal$ScanCount[twal$ScanCount < 0] <- 0
twal$ResultCount <- twal$ScanCount - twal$ReturnCount

# Calculate Scan and Return counts by VisitNumber
library(dplyr)
item.counts <- summarise(group_by(twal, VisitNumber),
                         TotalScan = sum(ScanCount), TotalReturn = sum(ReturnCount), TotalResult = sum(ResultCount))

# Include ReturnCount column
testwal$ReturnCount <- -testwal$ScanCount
testwal$ReturnCount[testwal$ReturnCount < 0] <- 0
testwal$ScanCount[testwal$ScanCount < 0] <- 0
testwal$ResultCount <- testwal$ScanCount - testwal$ReturnCount

# Calculate Scan and Return counts by VisitNumber
library(dplyr)
item.counts1 <- summarise(group_by(testwal, VisitNumber),
                         TotalScan = sum(ScanCount), TotalReturn = sum(ReturnCount), TotalResult = sum(ResultCount))



##
outcomes <- data.table(TripType = sort(unique(twal$TripType)))
outcomes$Index <- seq_along(outcomes$TripType) - 1

y <- plyr::mapvalues(twal$TripType, from = outcomes$TripType, to = outcomes$Index)

num.class <- length(unique(y))

# xgboost parameters
param <- list("objective" = "multi:softprob",    # multiclass classification 
              "num_class" = num.class,    # number of classes 
              "eval_metric" = "mlogloss",    # evaluation metric 
              "nthread" = 8,   # number of threads to be used 
              "max_depth" = 16,    # maximum depth of tree 
              "eta" = 0.3,    # step size shrinkage 
              "gamma" = 0,    # minimum loss reduction 
              "subsample" = 1,    # part of data instances to grow tree 
              "colsample_bytree" = 1,  # subsample ratio of columns when constructing each tree 
              "min_child_weight" = 12  # minimum sum of instance weight needed in a child 
)

train.matrix <- as.matrix(twal)
train.matrix <- as(train.matrix, "dgCMatrix") # conversion to sparse matrix
dtrain <- xgb.DMatrix(data = train.matrix, label = y)

set.seed(456)
cv.nround <- 50 # 200
cv.nfold <- 3 # 10
bst.cv <- xgb.cv(param=param, data=dtrain, 
                 nfold=cv.nfold, nrounds=cv.nround, prediction=TRUE) 
tail(bst.cv$dt)

# Index of minimum merror
min.error.index = which.min(bst.cv$dt[, test.mlogloss.mean]) 
min.error.index 

# Minimum error
bst.cv$dt[min.error.index, ]

## Model
nround = min.error.index # number of trees generated
bst <- xgboost(param = param, data = dtrain, nrounds = nround, verbose = TRUE)

model <- xgb.dump(bst, with.stats = T)
model[1:10]








library(randomForest)
mtry <- tuneRF(twal, twal$TripType, ntreeTry = 100,stepFactor = 2,
               improve = 0.05, trace = TRUE, plot = TRUE, doBest = FALSE)
###this gives  mtry =18 ####

###applying prediction####
randomForestP = randomForest(TripType ~ ., data=twal, ntree=100, do.trace=TRUE,
                             mtry=18,nodesize=40) 
###applying on test data###
randomForestPredictF = predict(randomForestP, newdata=testwal)
answer<-data.frame(VisitNumber=testwal$VisitNumber,
                   TripType=randomForestPredictF)

####reshaping the result in required format###
y <- reshape(answer, direction="wide", v.names="TripType", timevar="TripType", 
             idvar="VisitNumber")
#####save it in csv arrange the columns in incresing order there #####
write_csv(y,"E:/FS.csv")

####reload it ####
yy<-read_csv("E:/FS.csv")
yy[is.na(y)]<-0                 ####replacing null by zero ###
yy[yy>0]<-1                     ####replacing values by 1 as required 

###final result#####

write_csv(yy,"E:/finalwalmart.csv")