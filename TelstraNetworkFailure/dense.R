library(magrittr)
library(data.table)
library(caret)
library(Matrix)
library(dplyr)
setwd("/Users/bikash/repos/kaggleCompetition1/TelstraNetworkFailure")
set.seed(12345)
time.start <- Sys.time()

logfile <- "data/xgblog.txt"

if (!file.exists(logfile)) {
  file.create(logfile)
}

#map <- plyr::mapvalues

d.event <- read.csv("data/event_type.csv") %>% data.table() %>% arrange(id) %>% rename(event = event_type)
d.feature <- read.csv("data/log_feature.csv") %>% data.table() %>% arrange(id) %>% rename(feature = log_feature)
d.resource <- read.csv("data/resource_type.csv") %>% data.table() %>% arrange(id) %>% rename(resource = resource_type)
d.severity <- read.csv("data/severity_type.csv") %>% data.table() %>% arrange(id) %>% rename(severity = severity_type)
d.test <- read.csv("data/test.csv") %>% data.table() %>% arrange(id)
d.train <- read.csv("data/train.csv") %>% data.table() %>% arrange(id) %>% rename(outcome = fault_severity)

trainindex <- data.table(id = sort(unique(d.train$id)))
trainindex$index <- seq_along(trainindex$id)

testindex <- data.table(id = sort(unique(d.test$id)))
testindex$index <- seq_along(testindex$id)

totrainindex <- function(id) {
  map(id, from = trainindex$id, to = trainindex$index, warn_missing = FALSE)
}

totestindex <- function(id) {
  map(id, from = testindex$id, to = testindex$index, warn_missing = FALSE)
}

d.event$event <- gsub("event_type ", "", d.event$event) %>% as.integer() %>% as.factor()
d.feature$feature <- gsub("feature ", "", d.feature$feature) %>% as.integer() %>% as.factor()
d.resource$resource <- gsub("resource_type ", "", d.resource$resource) %>% as.integer() %>% as.factor()
d.severity$severity <- gsub("severity_type ", "", d.severity$severity) %>% as.integer() %>% as.factor()
d.train$location <- gsub("location ", "", d.train$location) %>% as.integer() %>% as.factor()
d.test$location <- gsub("location ", "", d.test$location) %>% as.integer() %>% as.factor()

d.set <- rbind(d.train, mutate(d.test, outcome = -1)) %>% arrange(id)

## Only allow locations in d.test.location which are present in the training set
# d.location <- select(d.set, id, location) %>% filter(location %in% d.set$location[d.set$fault_severity != -1])
# d.train.location <- d.location[d.location$id %in% d.train$id]
# d.test.location <- d.location[d.location$id %in% d.test$id]

## d.event
# event 16 is not present in the dataset
m.event <- model.matrix(~ 0 + id + event, data = d.event) %>% as.data.table()
m.event <- aggregate(. ~ id, data = m.event, FUN = sum)

## d.feature
m.feature <- dcast(d.feature, id ~ feature, value.var = "volume", fun.aggregate = sum)
names(m.feature)[2:ncol(m.feature)] <- paste0("feature", names(m.feature)[2:ncol(m.feature)])

## d.resource
m.resource <- model.matrix(~ 0 + id + resource, data = d.resource) %>% as.data.table()
m.resource <- aggregate(. ~ id, data = m.resource, FUN = sum)

## d.severity
# severity may or may not be ordinal
m.severity <- model.matrix(~ 0 + id + severity, data = d.severity)

## d.set$location
m.location <- model.matrix(~ 0 + id + location + outcome, data = d.set) %>% as.data.frame()
m.location <- m.location[, colSums(m.location[d.set$outcome != -1, ]) != 0]

## m.set
m.set <- merge(m.event, m.feature, by = "id") %>%
  merge(m.resource, by = "id") %>%
  merge(m.severity, by = "id") %>%
  merge(m.location, by = "id") %>%
  arrange(id)

m.set <- m.set[, colSums(m.set[m.set$outcome != -1, ]) != 0]

m.train <- filter(m.set, outcome != -1) %>% dplyr::select(-c(id, outcome))
m.test <- filter(m.set, outcome == -1) %>% dplyr::select(-c(id, outcome))

## y
y <- d.train$outcome

## xgb
dtrain <- xgb.DMatrix(data = as.matrix(m.train),
                      label = y)

num.class <- length(unique(d.train$outcome))

param <- list("objective" = "multi:softprob",    # multiclass classification
              "num_class" = num.class,    # number of classes
              "eval_metric" = "mlogloss",    # evaluation metric
              "nthread" = 8,   # number of threads to be used
              "max_depth" = 10,    # maximum depth of tree
              "eta" = 0.3,    # step size shrinkage
              "gamma" = 0,    # minimum loss reduction
              "subsample" = 1,    # part of data instances to grow tree
              "colsample_bytree" = 1,  # subsample ratio of columns when constructing each tree
              "min_child_weight" = 2,  # minimum sum of instance weight needed in a child
              "verbose" = 2
)

set.seed(123534)

library(xgboost)
cv.nround <- 60
cv.nfold <- 10

bst.cv <- xgb.cv(param=param, data=dtrain,
                 nfold=cv.nfold, nrounds=cv.nround, prediction=TRUE)

tail(bst.cv$dt)

# Index of minimum merror
min.error.index = which.min(bst.cv$dt[, test.mlogloss.mean])
# sink(file = logfile, append = TRUE)
print(paste0("Min Error Index = ", min.error.index))

# Minimum error
print(bst.cv$dt[min.error.index, ])
# sink()

nround = min.error.index # number of trees generated
bst <- xgboost(param = param, data = dtrain, nrounds = nround, verbose = TRUE)

model <- xgb.dump(bst, with.stats = T)
# xgb.dump(bst, "dump.raw.txt", with.stats = T)
model[1:10]

# Compute feature importance matrix
feature.names <- names(m.set)[2:ncol(m.set) - 1]
importance_matrix <- xgb.importance(feature.names, model = bst)
xgb.plot.importance(importance_matrix[1:20,])

# Tree plot - not working
xgb.plot.tree(feature_names = feature.names, model = bst, n_first_tree = 2)

# Prediction
pred <- predict(bst, as.matrix(m.test))

# Decode prediction
pred <- matrix(pred, nrow=num.class, ncol=length(pred) / num.class)
pred <- data.frame(cbind(testindex$id, t(pred)))
names(pred) <- c("id", "predict_0", "predict_1", "predict_2")

plot(density(pred$predict_0), col = "green", ylim = c(0, 30))
lines(density(pred$predict_1), col = "orange")
lines(density(pred$predict_2), col = "red")

# output
filename <- "telstra_xgboost4.csv"
write.table(format(pred, scientific = FALSE), paste("output/", filename, sep = ""), row.names = FALSE, sep = ",", quote = FALSE)

save.image("xgbimage.RData")

time.end <- Sys.time()
time.end - time.start

## glmnet.cr - need full rank, etc.
library(glmnetcr)

load("xgbimage.RData")
p <- as.data.frame(as.matrix(train.data))
r <- as.data.frame(as.matrix(test.data))
y.factor <- factor(y, levels = c(0, 1, 2), ordered = TRUE)

set.seed(1234)

fit <- glmnet.cr(train.data, y.factor)
print(fit)

BIC.step <- select.glmnet.cr(fit)
BIC.step

# AIC.step <- select.glmnet.cr(fit, which = "AIC")
# AIC.step
#
# coefficients <- coef(fit, s = BIC.step)
# coefficients$a0
#
# sum(coefficients$beta != 0)
#
nonzero.glmnet.cr(fit, s = BIC.step)
#
# plot(fit)


f <- fitted.glmnet.cr(object = fit, newx = test.data, s = AIC.step)
pred.glm <- data.frame(cbind(testindex$id, f$probs))


# output
filename <- "telstra_xgboost5.csv"
names(pred.glm) <- c("id", "predict_0", "predict_1", "predict_2")
write.table(format(pred.glm, scientific = FALSE), paste("./output/", filename, sep = ""), row.names = FALSE, sep = ",", quote = FALSE)
