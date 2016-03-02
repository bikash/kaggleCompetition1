library(httr)


makeReadable <- function(x){
  num <- sapply(strsplit(x, " "), `[`, 2)
  c <- str_sub(x, 1, 1)
  return(paste0(c,num))}

makeNumeric <- function(x){
  num <- sapply(strsplit(x, " "), `[`, 2)
  return(as.numeric(num))}

mc_logloss <- function (data, lev = NULL, model = NULL) {
  if (!all(levels(data[, "pred"]) == levels(data[, "obs"])))
    stop("levels of observed and predicted data do not match")
  eps=1e-15
  n <- nrow(data)
  out <- rep(0.0, n)
  for (i in 1:n) {
    out[i] <- max(min(data[i, as.character(data$obs[i])], 1-eps), eps)
  }
  out <- -1.0*sum(log(out))/n
  names(out) <- c("mlogloss")
  return(out)
}

notify_android <- function(event, msg)
{
  body <- list(
    apikey = "71bfd9cfea5a4ffc48153647177d818e211f40daca35043b", 
    application = "RStudio", 
    event = event,
    description = msg)
  
  r <- POST(url = "https://www.notifymyandroid.com/publicapi/notify", body = body, encode = "form")
}