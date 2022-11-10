library(caret)
library(xgboost)
library(stackgbm)
ecoli_attr <- read.csv("Data/Processed/ecoli_attributed.csv")

ecoli_235 <- ecoli_attr[,2:9]

#make this example reproducible
set.seed(0)

#split into training (80%) and testing set (20%)
parts <- createDataPartition(ecoli_235$ecoli_235, p = .8, list = F)

train <- ecoli_235[parts, ]
test <- ecoli_235[-parts, ]

train_x <- data.matrix(train[, -8])
train_y <- train[,8]

test_x <- data.matrix(test[, -8])
test_y <- test[, 8]

xgb_train <- xgb.DMatrix(data = train_x, label = train_y)
xgb_test <- xgb.DMatrix(data = test_x, label = test_y)
watchlist <- list(train=xgb_train, test=xgb_test)
model <- xgb.train(data = xgb_train, max.depth = 3, watchlist=watchlist, nrounds = 200,objective = "binary:logistic")

finalmodel <- xgboost(data = xgb_train, max.depth = 3, nrounds = 140, verbose = 0)

pred <- predict(finalmodel, as.matrix(test_x))
pred <-  as.numeric(pred > 0.45)
confusionMatrix(factor(pred),factor(test_y))

###################################### Breaking the model #####################
ecoli_attr <- read.csv("Data/Processed/ecoli_attributed.csv")

ecoli_235 <- ecoli_attr2[,2:9]

#make this example reproducible
set.seed(0)

#split into training (80%) and testing set (20%)
parts <- createDataPartition(ecoli_235$ecoli_235, p = .8, list = F)

train <- ecoli_235[parts, ]
test<-ecoli_235[-parts, ]

# Change the value here. Previous 30 precip was the most strongly correlated to the test.
# Altering this value significantly changed the results. 
# This means that the older model appears much more trustworthy.

test$Previous30Precip <- test$Previous30Precip + 2

train_x <- data.matrix(train[, -8])
train_y <- train[,8]

test_x <- data.matrix(test[, -8])
test_y <- test[, 8]

xgb_train <- xgb.DMatrix(data = train_x, label = train_y)
xgb_test <- xgb.DMatrix(data = test_x, label = test_y)
watchlist <- list(train=xgb_train, test=xgb_test)
model <- xgb.train(data = xgb_train, max.depth = 3, watchlist=watchlist, nrounds = 200,objective = "binary:logistic")

finalmodel <- xgboost(data = xgb_train, max.depth = 3, nrounds = 140, verbose = 0)

pred <- predict(finalmodel, as.matrix(test_x))
pred <-  as.numeric(pred > 0.45)
confusionMatrix(factor(pred),factor(test_y))

###################################### Breaking the model 2 #####################
ecoli_attr <- read.csv("Data/Processed/ecoli_attributed.csv")

ecoli_235 <- ecoli_attr2[,2:9]

#make this example reproducible
set.seed(0)

#split into training (80%) and testing set (20%)
parts <- createDataPartition(ecoli_235$ecoli_235, p = .8, list = F)

train <- ecoli_235[parts, ]
test<-ecoli_235[-parts, ]

# Discharge value changed to reflect a different stage. Added 20 cfs so represent reasonably larger flow (but not flooding)
# Altering this value significantly changed the results. 
# This means that the older model appears much more trustworthy.

test$Previous30Precip <- test$Discharge_CFS + 20

train_x <- data.matrix(train[, -8])
train_y <- train[,8]

test_x <- data.matrix(test[, -8])
test_y <- test[, 8]

xgb_train <- xgb.DMatrix(data = train_x, label = train_y)
xgb_test <- xgb.DMatrix(data = test_x, label = test_y)
watchlist <- list(train=xgb_train, test=xgb_test)
model <- xgb.train(data = xgb_train, max.depth = 3, watchlist=watchlist, nrounds = 200,objective = "binary:logistic")

finalmodel <- xgboost(data = xgb_train, max.depth = 3, nrounds = 1, verbose = 0)

pred <- predict(finalmodel, as.matrix(test_x))
pred <-  as.numeric(pred > 0.45)
confusionMatrix(factor(pred),factor(test_y))

###################################### Breaking the model 3#####################
ecoli_attr <- read.csv("Data/Processed/ecoli_attributed.csv")

ecoli_235 <- ecoli_attr2[,2:9]

#make this example reproducible
set.seed(0)

#split into training (80%) and testing set (20%)
parts <- createDataPartition(ecoli_235$ecoli_235, p = .8, list = F)

train <- ecoli_235[parts, ]
test<-ecoli_235[-parts, ]

# What if I take previous30 precip out of the equation
train_matrix <- data.matrix(train)

train_x <- data.matrix(train[, 2:7])
train_y <- train[,8]

test_x <- data.matrix(test[, 2:7])
test_y <- test[, 8]

xgb_train <- xgb.DMatrix(data = train_x, label = train_y)
xgb_test <- xgb.DMatrix(data = test_x, label = test_y)
watchlist <- list(train=xgb_train, test=xgb_test)
model <- xgb.train(data = xgb_train, max.depth = 3, watchlist=watchlist, nrounds = 200,objective = "binary:logistic")

finalmodel <- xgboost(data = xgb_train, max.depth = 3, nrounds = 60, verbose = 0)

# Without the previous precip the accuracy drops 5 - 10% 

pred <- predict(finalmodel, as.matrix(test_x))
pred <-  as.numeric(pred > 0.45)
confusionMatrix(factor(pred),factor(test_y))


##############################
library(tidyverse)
folds = createFolds(train_x, k = 10)
cv <- lapply(folds, function(x) {
  # we stick our XGBoost classifier in here
  classifier = xgboost(data = as.matrix(train_x), label = train_y, nrounds = 10)
  y_pred = predict(classifier, newdata = as.matrix(test_x)) # again need a matrix
  y_pred = (y_pred >= 0.5) # here we are setting up the binary outcome of 0 or 1
  cm = table(test_y, y_pred)
  accuracy = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])
  return(accuracy)
})
accuracy = mean(as.numeric(cv))
print(accuracy)
