rm(list=ls(all=TRUE))

setwd("/Users/bikash/repos/kaggleCompetition1/winton")
library(readr)
library(dplyr)
library(tidyr)
library(Matrix)
library(ggplot2)
library(xgboost)

options(mc.cores = parallel::detectCores(),
        stringsAsFactors = FALSE,
        scipen = 10) 

#' coalesce function
`%||%` <- function(a, b) ifelse(!is.na(a), a, b)

train.full <- read_csv("data/train.csv")
test.full <- read_csv("data/test.csv")
sample.submission <- read_csv("data/submission.csv")

names(train.full)


##Test data
intra.ret <- test.full %>%
  select(Ret_2:Ret_120) %>%
  as.matrix()

intra.ret <- intra.ret+1
prod <- apply(intra.ret, 1, prod, na.rm=TRUE)

n.returns <- apply(!is.na(intra.ret), 1, sum)

df <- data_frame(total.gr.intra = prod,
                 n.returns.intra = n.returns) %>%
  mutate(return.intra = total.gr.intra^(1/n.returns)-1) %>%
  mutate(return.intra.day = (return.intra+1)^420-1)

test.imp <- cbind(test.full, df) %>%
  mutate(Feature_3 = Feature_3 %||% Feature_4,
         Feature_4 = Feature_4 %||% Feature_3,
         Feature_6 = Feature_6 %||% Feature_21,
         Feature_21 = Feature_21 %||% Feature_6,
         Feature_18 = Feature_18 %||% Feature_21,
         Feature_21 = Feature_21 %||% Feature_18) %>%
  #8 level: +++, ---, ++-, --+, -++, +--, +-+, -+-
  mutate(level_8 = ifelse(sign(Ret_MinusOne) == sign(Ret_MinusTwo) & 
                            sign(Ret_MinusOne) == sign(return.intra) & 
                            1 == sign(return.intra), 1, 0),
         level_8 = ifelse(sign(Ret_MinusOne) == sign(Ret_MinusTwo) & 
                            sign(Ret_MinusOne) == sign(return.intra) & 
                            -1 == sign(return.intra), 2, level_8),
         # ++-
         level_8 = ifelse(sign(Ret_MinusOne) == sign(Ret_MinusTwo) & 
                            sign(Ret_MinusOne) != sign(return.intra) & 
                            -1 == sign(return.intra), 3, level_8),
         # --+
         level_8 = ifelse(sign(Ret_MinusOne) == sign(Ret_MinusTwo) & 
                            sign(Ret_MinusOne) != sign(return.intra) & 
                            1 == sign(return.intra), 4, level_8),
         # -++
         level_8 = ifelse(sign(Ret_MinusOne) != sign(Ret_MinusTwo) & 
                            sign(Ret_MinusOne) == sign(return.intra) & 
                            1 == sign(return.intra), 5, level_8),
         # +--
         level_8 = ifelse(sign(Ret_MinusOne) != sign(Ret_MinusTwo) & 
                            sign(Ret_MinusOne) == sign(return.intra) & 
                            -1 == sign(return.intra), 6, level_8),
         # +-+
         level_8 = ifelse(sign(Ret_MinusOne) != sign(Ret_MinusTwo) & 
                            sign(Ret_MinusOne) != sign(return.intra) & 
                            1 == sign(return.intra), 7, level_8),
         # -+-
         level_8 = ifelse(sign(Ret_MinusOne) != sign(Ret_MinusTwo) & 
                            sign(Ret_MinusOne) != sign(return.intra) & 
                            -1 == sign(return.intra), 8, level_8)) %>%
#  replace_na(list(Feature_1=0,Feature_2=0,Feature_3=0,Feature_4=0,Feature_5=0,Feature_6=0,Feature_7=0,Feature_8=0,Feature_9=0,Feature_10=0,
#                  Feature_11=0,Feature_12=0,Feature_13=0,Feature_14=0,Feature_15=0,Feature_16=0,Feature_17=0,Feature_18=0,Feature_19=0,Feature_20=0,
#                  Feature_21=0,Feature_22=0,Feature_23=0,Feature_24=0,Feature_25=0))

intra.median <- test.full %>%
  select(Ret_2:Ret_120) %>%
  by_row(lift_vl(median), na.rm=TRUE, .labels=FALSE, .to='intra.median', .collate = "rows") %>%
  bind_cols(test.full %>% select(Id))
intra.mean <- test.full %>%
  select(Ret_2:Ret_120) %>%
  by_row(lift_vl(mean), na.rm=TRUE, .labels=FALSE, .to='intra.mean', .collate = "rows") %>%
  bind_cols(test.full %>% select(Id))

test.imp <- test.imp %>%
  inner_join(intra.median) %>%
  inner_join(intra.mean) %>%
  mutate(gr.day = ((1+Ret_MinusOne) * (1+Ret_MinusTwo) * (1+return.intra.day)),
         gr.day = gr.day^(1/3)-1)

##Train data
intra.ret <- train.full %>%
  select(Ret_2:Ret_120) %>%
  as.matrix()

intra.ret <- intra.ret+1
prod <- apply(intra.ret, 1, prod, na.rm=TRUE)

n.returns <- apply(!is.na(intra.ret), 1, sum)

df <- data_frame(total.gr.intra = prod,
                 n.returns.intra = n.returns) %>%
  mutate(return.intra = total.gr.intra^(1/n.returns)-1) %>%
  mutate(return.intra.day = (return.intra+1)^420-1)

train.imp <- cbind(train.full, df) %>%
  mutate(Feature_3 = Feature_3 %||% Feature_4,
         Feature_4 = Feature_4 %||% Feature_3,
         Feature_6 = Feature_6 %||% Feature_21,
         Feature_21 = Feature_21 %||% Feature_6,
         Feature_18 = Feature_18 %||% Feature_21,
         Feature_21 = Feature_21 %||% Feature_18) %>%
  #8 level: +++, ---, ++-, --+, -++, +--, +-+, -+-
  mutate(level_8 = ifelse(sign(Ret_MinusOne) == sign(Ret_MinusTwo) & 
                            sign(Ret_MinusOne) == sign(return.intra) & 
                            1 == sign(return.intra), 1, 0),
         level_8 = ifelse(sign(Ret_MinusOne) == sign(Ret_MinusTwo) & 
                            sign(Ret_MinusOne) == sign(return.intra) & 
                            -1 == sign(return.intra), 2, level_8),
         # ++-
         level_8 = ifelse(sign(Ret_MinusOne) == sign(Ret_MinusTwo) & 
                            sign(Ret_MinusOne) != sign(return.intra) & 
                            -1 == sign(return.intra), 3, level_8),
         # --+
         level_8 = ifelse(sign(Ret_MinusOne) == sign(Ret_MinusTwo) & 
                            sign(Ret_MinusOne) != sign(return.intra) & 
                            1 == sign(return.intra), 4, level_8),
         # -++
         level_8 = ifelse(sign(Ret_MinusOne) != sign(Ret_MinusTwo) & 
                            sign(Ret_MinusOne) == sign(return.intra) & 
                            1 == sign(return.intra), 5, level_8),
         # +--
         level_8 = ifelse(sign(Ret_MinusOne) != sign(Ret_MinusTwo) & 
                            sign(Ret_MinusOne) == sign(return.intra) & 
                            -1 == sign(return.intra), 6, level_8),
         # +-+
         level_8 = ifelse(sign(Ret_MinusOne) != sign(Ret_MinusTwo) & 
                            sign(Ret_MinusOne) != sign(return.intra) & 
                            1 == sign(return.intra), 7, level_8),
         # -+-
         level_8 = ifelse(sign(Ret_MinusOne) != sign(Ret_MinusTwo) & 
                            sign(Ret_MinusOne) != sign(return.intra) & 
                            -1 == sign(return.intra), 8, level_8)) %>%
  replace_na(list(Feature_1=0,Feature_2=0,Feature_3=0,Feature_4=0,Feature_5=0,Feature_6=0,Feature_7=0,Feature_8=0,Feature_9=0,Feature_10=0,
                  Feature_11=0,Feature_12=0,Feature_13=0,Feature_14=0,Feature_15=0,Feature_16=0,Feature_17=0,Feature_18=0,Feature_19=0,Feature_20=0,
                  Feature_21=0,Feature_22=0,Feature_23=0,Feature_24=0,Feature_25=0))

intra.median <- train.full %>%
  select(Ret_2:Ret_120) %>%
  by_row(lift_vl(median), na.rm=TRUE, .labels=FALSE, .to='intra.median', .collate = "rows") %>%
  bind_cols(train.full %>% select(Id))
intra.mean <- train.full %>%
  select(Ret_2:Ret_120) %>%
  by_row(lift_vl(mean), na.rm=TRUE, .labels=FALSE, .to='intra.mean', .collate = "rows") %>%
  bind_cols(train.full %>% select(Id))

train.imp <- train.imp %>%
  inner_join(intra.median) %>%
  inner_join(intra.mean) %>%
  mutate(gr.day = ((1+Ret_MinusOne) * (1+Ret_MinusTwo) * (1+return.intra.day)),
         gr.day = gr.day^(1/3)-1)

#set.seed(12345)
set.seed(252014)

test.data <- test.imp %>%
  select(matches("Feature"), Ret_MinusTwo, Ret_MinusOne, total.gr.intra, n.returns.intra, return.intra, return.intra.day, level_8,
         intra.median, intra.mean, gr.day) %>%
  as.matrix

train.data.model <- train.imp %>%
  select(matches("Feature"), Ret_MinusTwo, Ret_MinusOne, total.gr.intra, n.returns.intra, return.intra, return.intra.day, level_8,
         intra.median, intra.mean, gr.day) %>%
  as.matrix

train.data.model <- cbind(train.data.model, 0)
test.data <- cbind(test.data, 0)

train.data.model.y <- train.imp %>%
  select(Ret_PlusOne:Ret_PlusTwo) %>%
  as.matrix

dtrain.1 <- xgb.DMatrix(data = train.data.model, label = train.data.model.y[,1])
dtrain.2 <- xgb.DMatrix(data = train.data.model, label = train.data.model.y[,2])

reg.1 <- xgb.train(data=dtrain.1, max.depth=20, eta=.05, nround=200, 
                   eval.metric = "rmse",
                   nthread = 8, objective = "reg:linear")
reg.2 <- xgb.train(data=dtrain.2, max.depth=20, eta=.05, nround=200, 
                   eval.metric = "rmse",
                   nthread = 8, objective = "reg:linear")

pred.1 <- predict(reg.1, test.data)
pred.2 <- predict(reg.2, test.data)

pred.test <- cbind(test.imp$Id, pred.1, pred.2) %>%
  as.data.frame() %>%
  setNames(c('Id','61','62'))

summary(train.imp$Ret_MinusTwo)
summary(train.imp$Ret_MinusOne)

summary(pred.test$`62`)
summary(pred.test$`61`)

y.pred.inter <- pred.test %>%
  gather(Id.y, Predicted, -Id) %>%
  mutate(Id.merge = paste0(Id, "_", Id.y)) %>%
  select(Id=Id.merge, Predicted.inter = Predicted) %>%
  arrange(Id)

y.pred.intra <- test.imp %>%
  select(Id, return.intra, intra.mean, intra.median)

submission <- sample.submission %>%
  separate(Id, c('Id2', 'Day'), remove=FALSE, convert=TRUE) %>%
  left_join(y.pred.inter, by=c('Id'='Id')) %>%
  left_join(y.pred.intra, by=c('Id2'='Id'))

submission <- submission %>%
  mutate(Predicted = ifelse(Day>60, Predicted.inter, 0)) %>%
  mutate(Predicted = ifelse(Day<=60, return.intra, Predicted)) %>%
  select(Id, Predicted)

file <- paste0("winton-inter_gbm_200_mean1", ".csv.gz")
write.csv(submission, gzfile(file), row.names=FALSE)

###########################
#Blends

sub.1 <- read.csv("winton-inter_gbm_400_median1.csv.gz")
sub.2 <- read.csv("winton-inter_gbm_200_mean1.csv.gz")

summary(sub.1$Predicted)
summary(sub.2$Predicted)

blend <- sub.1 %>%
  rename(Predicted.1=Predicted) %>%
  inner_join(sub.2 %>% rename(Predicted.2=Predicted)) %>%
  rowwise() %>%
  mutate(Predicted=(mean(c(Predicted.1, Predicted.2))))

blend <- blend %>% select(Id, Predicted)

file <- paste0("winton-inter_gbm_blend1", ".csv.gz")
write.csv(blend, gzfile(file), row.names=FALSE)

blend <- sub.1 %>%
  rename(Predicted.1=Predicted) %>%
  inner_join(sub.2 %>% rename(Predicted.2=Predicted)) %>%
  rowwise() %>%
  mutate(avg=(mean(c(Predicted.1, Predicted.2))),
         avg.0=(mean(c(Predicted.1, 0)))) %>%
  separate(Id, c('Id2', 'Day'), remove=FALSE, convert=TRUE) %>%
  mutate(Predicted = ifelse(Day>60, avg, Predicted.1))

blend <- blend %>% select(Id, Predicted)

file <- paste0("winton-inter_gbm_blend2", ".csv.gz")
write.csv(blend, gzfile(file), row.names=FALSE)

blend <- sub.1 %>%
  rename(Predicted.1=Predicted) %>%
  inner_join(sub.2 %>% rename(Predicted.2=Predicted)) %>%
  rowwise() %>%
  mutate(avg=(mean(c(Predicted.1, Predicted.2))),
         avg.0=(mean(c(Predicted.1, 0)))) %>%
  separate(Id, c('Id2', 'Day'), remove=FALSE, convert=TRUE) %>%
  mutate(Predicted = ifelse(Day>60, avg, avg.0))

blend <- blend %>% select(Id, Predicted)

file <- paste0("winton-inter_gbm_blend2-0", ".csv.gz")
write.csv(blend, gzfile(file), row.names=FALSE)