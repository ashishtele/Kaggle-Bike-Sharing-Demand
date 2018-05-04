rm(list = ls())

# Loaing the libraries
load_lb <- function()
{
  suppressPackageStartupMessages(library(doMC))
  registerDoMC(cores = 8)
  suppressPackageStartupMessages(library(readxl))
  suppressPackageStartupMessages(library(tidyr))
  suppressPackageStartupMessages(library(dplyr))
  suppressPackageStartupMessages(library(caret))
  suppressPackageStartupMessages(library(rpart))
  suppressPackageStartupMessages(library(tree))
  suppressPackageStartupMessages(library(MASS))
  suppressPackageStartupMessages(library(mice))
  suppressPackageStartupMessages(require(xgboost))
  suppressPackageStartupMessages(require(data.table))
  suppressPackageStartupMessages(require(Matrix))
  suppressPackageStartupMessages(require(ggplot2))
  suppressPackageStartupMessages(require(lubridate))
}

load_lb()

# Importing the train and test files

bike_train <- fread("E:/Study/R Projects/Common files/Bike_train.csv")
bike_test <- fread("E:/Study/R Projects/Common files/Bike_test.csv")

# Glimpse and summary of files

glimpse(bike_train)
summary(bike_train)
glimpse(bike_test)
summary(bike_test)

# EDA and Feature engineering

# 'Count' distribution

ggplot(bike_train,aes(x=count, fill = "red")) +
  geom_density() +
  ggtitle("Original count plot")


library(moments)
skewness(bike_train$count)
skewness(log(bike_train$count+1))
skewness(sqrt(bike_train$count))
kurtosis(bike_train$count)

ggplot(bike_train,aes(x=log(count+1), fill = "red")) +
  geom_density() +
  ggtitle("Count plot: transformed")

## New fields creation
library(fasttime)

bike_train %>%
  mutate(datetime = fastPOSIXct(datetime,"GMT")) %>%
  mutate(hour = hour(datetime),
         month = month(datetime),
         year = year(datetime),
         wkday = wday(datetime))-> bike

bike_test %>%
  mutate(datetime = fastPOSIXct(datetime,"GMT")) %>%
  mutate(hour = hour(datetime),
         month = month(datetime),
         year = year(datetime),
         wkday = wday(datetime))-> test

head(bike,3)

## Correlation

corl <- cor(bike[,-1])
corrplot::corrplot(corl, method = "ellipse", type = c("upper"))


bike$count <- log(bike$count+1)


# Matrix cration for xgboost

rmv <- c("datetime","casual","registered","count")
bike[,!names(bike) %in% rmv] %>% as.matrix() -> x_train
y_train <- bike$count
test[,!names(test) %in% "datetime"] %>% as.matrix() -> x_test


d_train <- xgb.DMatrix(x_train,label = y_train)
d_test <- xgb.DMatrix(x_test)
model1 <- xgb.train(data = d_train, nround = 1000, max_depth = 5,
                    eta=0.1, subsample = 0.9)
xgb.importance(feature_names = colnames(bike), model1) %>% xgb.plot.importance()



pred <- predict(model1, x_test)
pred <- exp(pred)

solution <- data.frame(datetime = test$datetime, count = pred)

## tunning

# Default parameters

xgbparams <- list(
  booster = "gbtree",
  objective = 'reg:linear',
  colsample_bytree=1, # no of features supplied to a tree (0.5 to 0.9)
  eta=0.1, # learning rate (0.01 to 0.3)
  min_child_weight=1,
  max_depth = 5,
  alpha=0.3, #L1
  lambda=0.8, #L2
  gamma=0.5, # prevent overfit (regularixation)
  # gamma brings improvement when we want to use shallow (low max_depth) trees
  subsample=0.8, # no of observations supplied to a tree (0.5 to 0.8)
  silent=TRUE,
  eval_metrics = 'rmse' 
)

set.seed(123)
xgb_cv2 <- xgb.cv(params = xgbparams, 
                  data = d_train,
                  nrounds =1000, 
                  nfold = 5, # the original dataset is randomly partitioned into nfold equal size subsamples.
                  stratified =T,
                  print_every_n = 50, verbose = T, showsd = T, prediction = T,
                  early_stopping_rounds = 20, maximize = F)
# Best iteration: 535

xgb1 <- xgb.train (params = xgbparams, 
                   data = d_train, 
                   nrounds = 535,
                   print_every_n = 50, 
                   early_stop_round = 20, 
                   maximize = F)
pred2 <- predict(xgb1,d_test)
pred2 <- exp(pred2)

## MLR

library(parallelMap)
library(mlr)

set.seed(931992)

bike1 <- bike
test1 <- test

# Factor conversion
fct_col <- c("season","holiday","workingday","weather","hour","month",
             "year","wkday")

bike1[,fct_col] <- lapply(bike1[,fct_col],factor)
glimpse(bike1)
bike1[,c("datetime","casual","registered")] <- NULL

test1[,fct_col] <- lapply(test1[,fct_col],factor)
glimpse(test1)
test1[,c("datetime")] <- NULL

# One hot encoding
bike1 <- createDummyFeatures(bike1,target = "count")
test1 <- createDummyFeatures(test1)

# create tasks (train)
t_task <- makeRegrTask(data=bike1, target = "count")

# 
lerner <- makeLearner("regr.xgboost",
                      nrounds = 800,
                      nthread =1, base_score = mean(bike1$count))
getParamSet("regr.xgboost")

# Parameter space
ps <- makeParamSet(
  makeNumericParam("eta", lower = 0.01, upper = 0.08),
  makeNumericParam("subsample", lower = 0.7, upper = 1),
  makeNumericParam("colsample_bytree", lower = 0.5, upper = 1),
  makeIntegerParam("max_depth", lower = 5, upper = 12),
  makeIntegerParam("min_child_weight", lower = 1, upper = 50)
      )

# Search strategy
ctrl = makeTuneControlRandom(maxit = 50)
# Resampling strategy
rdesc = makeResampleDesc("CV", iters = 5)

# Parameter tuning
res = tuneParams(learner = lerner, 
                 task = t_task, resampling = rdesc, measures = rmse, 
                 par.set = ps, control = ctrl)

# tuning result
res

# Set hyperparameters
lrn_t <- setHyperPars(lerner, par.vals = res$x)

# Train model
xgmodel <- mlr::train(lrn_t,t_task)

# Model prediction
pred3 <- predict(xgmodel, newdata = test1)
pred4 <- expm1(getPredictionResponse(pred3))
