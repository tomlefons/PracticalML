Practical ML Course Project
========================================================

# Executive Summary

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. 
Data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants were used. Participants were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). 

The supervised learning task of the project is to predict how well subjects performed weight lifting excercises based on the above described data. The outcome is classified into five different categories. 

The target of choosing and quantifying a model that correctly classifies 20 samples provided as a testing set was achieved with a random forest model.

## Data loading

```r
train = read.csv("pml-training.csv")
test = read.csv("pml-testing.csv")
```


## Feature Selection
Several features are obviously useless for the predecition especially time related data. These might need to be reinstated, if accuracy is too low. All columns with a large number of NAs in the test or training data set were deleted. There was no need to impute any cases. The size of the data material is more than sufficient. 


```r
str(train)
train <- train[-c(1:7)]
test <- test[-c(1:7)]

NA_col_overview <- data.frame(train = apply(train, 2, function(x) {
    sum(is.na(x))
}), test = apply(test, 2, function(x) {
    sum(is.na(x))
}))
NA_col_overview
NA_col_train <- which(NA_col_overview$train > 0)
NA_col_test <- which(NA_col_overview$test > 0)
to_be_deleted <- union(NA_col_train, NA_col_test)
train <- train[, -to_be_deleted]
test <- test[, -to_be_deleted]

dim(train)
length(complete.cases(train))
```


## Creating training and test datasets

```r
library(caret)
set.seed(1)
inTrain <- createDataPartition(train$classe, p = 0.6, list = FALSE)
test_internal <- train[-inTrain, ]
train_full <- train[inTrain, ]
```


## Model selection via Accuracy
Very important. The model has to accurately predict 20 test cases. 


```r
set.seed(1)
train <- train_full[sample(nrow(train_full), 3000), ]
```


### Decision Tree

```r
mod <- train(train$classe ~ ., data = train, method = "rpart")
cM <- confusionMatrix(predict(mod, newdata = test_internal), test_internal$classe)
cM$overall[1]
```

Accuracy: 0.54

### LDA

```r
mod <- train(train$classe ~ ., data = train, method = "lda")
cM <- confusionMatrix(predict(mod, newdata = test_internal), test_internal$classe)
cM$overall[1]
```

Accuracy: 0.70 


### Random Forest

```r
mod <- train(train$classe ~ ., method = "rf", data = train, trControl = trainControl(method = "cv", 
    number = 4), importance = TRUE)
cM <- confusionMatrix(predict(mod, newdata = test_internal), test_internal$classe)
cM$overall[1]
```

Accuracy: 0.96 

# Results with Random Forest 
Random Forest is the best model regarding accuracy of the selected ones.
The Out of Sample error with 4 fold cross-validation is 0.04 or 4% i.e. (1- Accuracy).

The standard random forest model shows already with only 3000 rows 100% accuracy for the provided test set.
All the 20 test cases were correct. 

The accuracy could be further increased by using a bigger training set and by optimizing the model.


## Creation of the prediction files of the test set for the grading task 

```r
answers = predict(mod, newdata = test)
pml_write_files = function(x) {
    n = length(x)
    for (i in 1:n) {
        filename = paste0("problem_id_", i, ".txt")
        write.table(x[i], file = filename, quote = FALSE, row.names = FALSE, 
            col.names = FALSE)
    }
}
pml_write_files(answers)
```

