## KNN Numeric


library(class)

# 1. Predictor columns (numeric)
predictor_cols <- c(
  "Age",
  "Academic.Pressure",
  "Work.Pressure",
  "CGPA",
  "Study.Satisfaction",
  "Job.Satisfaction",
  "Work.Study.Hours",
  "Financial.Stress"
)

# 2. Build KNN train/test from rf_data and train_idx
knn_train <- rf_data[train_idx, ]
knn_test  <- rf_data[-train_idx, ]

X_train <- knn_train[, predictor_cols]
X_test  <- knn_test[, predictor_cols]

# all predictors are numeric 
X_train <- data.frame(lapply(X_train, function(z) as.numeric(as.character(z))))
X_test  <- data.frame(lapply(X_test,  function(z) as.numeric(as.character(z))))

y_train <- knn_train$Depression
y_test  <- knn_test$Depression

# 3. Drop any rows with NA
train_complete <- complete.cases(X_train) & !is.na(y_train)
test_complete  <- complete.cases(X_test)  & !is.na(y_test)

X_train <- X_train[train_complete, , drop = FALSE]
y_train <- y_train[train_complete]

X_test  <- X_test[test_complete, , drop = FALSE]
y_test  <- y_test[test_complete]

# 4. 5-fold cross-validation on TRAIN 
set.seed(123)

n_train <- nrow(X_train)
K_folds <- 5

fold_id <- sample(rep(1:K_folds, length.out = n_train))

k_values <- c(5, 10)   
cv_acc <- numeric(length(k_values))

for (j in seq_along(k_values)) {
  k <- k_values[j]
  fold_acc <- numeric(K_folds)
  
  for (f in 1:K_folds) {
    val_idx <- which(fold_id == f)
    tr_idx  <- setdiff(1:n_train, val_idx)
    
    X_tr_f <- X_train[tr_idx, , drop = FALSE]
    X_val_f <- X_train[val_idx, , drop = FALSE]
    y_tr_f <- y_train[tr_idx]
    y_val_f <- y_train[val_idx]
    
    # Scale predictors using only the training part of this fold
    means_f <- apply(X_tr_f, 2, mean)
    sds_f   <- apply(X_tr_f, 2, sd)
    
    X_tr_f_sc  <- scale(X_tr_f,  center = means_f, scale = sds_f)
    X_val_f_sc <- scale(X_val_f, center = means_f, scale = sds_f)
    
    # KNN on this fold
    pred_val <- knn(
      train = X_tr_f_sc,
      test  = X_val_f_sc,
      cl    = y_tr_f,
      k     = k
    )
    
    tab_val <- table(Predicted = pred_val, Actual = y_val_f)
    fold_acc[f] <- sum(diag(tab_val)) / sum(tab_val)
  }
  
  cv_acc[j] <- mean(fold_acc)
}

# 5. CV results and best k 
knn_cv_results <- data.frame(
  k = k_values,
  cv_accuracy = cv_acc
)
knn_cv_results  

best_idx <- which.max(cv_acc)
best_k <- k_values[best_idx]
best_k

best_cv_accuracy <- cv_acc[best_idx]
best_cv_accuracy

# 6. Fit final KNN on FULL TRAIN set with best k and evaluate on TEST set

train_means <- apply(X_train, 2, mean)
train_sds   <- apply(X_train, 2, sd)

X_train_scaled <- scale(X_train, center = train_means, scale = train_sds)
X_test_scaled  <- scale(X_test,  center = train_means, scale = train_sds)

best_pred <- knn(
  train = X_train_scaled,
  test  = X_test_scaled,
  cl    = y_train,
  k     = best_k
)

best_tab <- table(Predicted = best_pred, Actual = y_test)
best_tab

best_test_accuracy <- sum(diag(best_tab)) / sum(best_tab)
best_test_accuracy

# 7. Plot: 5-fold CV accuracy for k = 5 and 10
plot(
  k_values, cv_acc, type = "b",
  xlab = "Number of Neighbors (k)",
  ylab = "5-fold CV Accuracy",
  main = "KNN 5-fold CV Accuracy (k = 5 vs 10)"
)



## KNN Numeric + Categorical


##-------------------------Munisa's KNN with numeric + categorical--------------------------------

library(class)

# New data frame
df_all <- read.csv("student_depression_dataset.csv")

# 1. Select numeric + categorical predictors + response
knn2_data <- df_all[, c(
  "Age",
  "Academic.Pressure",
  "Work.Pressure",
  "CGPA",
  "Study.Satisfaction",
  "Job.Satisfaction",
  "Work.Study.Hours",
  "Financial.Stress",
  "Gender",
  "Sleep.Duration",
  "Dietary.Habits",
  "Have.you.ever.had.suicidal.thoughts..",
  "Family.History.of.Mental.Illness",
  "Depression"
)]

# 2. Drop NA
knn2_data <- na.omit(knn2_data)

# 3. Mark categorical predictors and response as factors
knn2_data$Gender <- as.factor(knn2_data$Gender)
knn2_data$Sleep.Duration <- as.factor(knn2_data$Sleep.Duration)
knn2_data$Dietary.Habits <- as.factor(knn2_data$Dietary.Habits)
knn2_data$Have.you.ever.had.suicidal.thoughts.. <- as.factor(knn2_data$Have.you.ever.had.suicidal.thoughts..)
knn2_data$Family.History.of.Mental.Illness <- as.factor(knn2_data$Family.History.of.Mental.Illness)
knn2_data$Depression <- as.factor(knn2_data$Depression)

# 4. Create dummy variables for categoricals using model.matrix
X_full <- model.matrix(Depression ~ ., data = knn2_data)[, -1]  # drop intercept
y_full <- knn2_data$Depression

# 5. Train/test split (80/20)
set.seed(123)
n2 <- nrow(X_full)
train_idx2 <- sample(1:n2, size = floor(0.8 * n2))

X2_train <- X_full[train_idx2, , drop = FALSE]
X2_test  <- X_full[-train_idx2, , drop = FALSE]
y2_train <- y_full[train_idx2]
y2_test  <- y_full[-train_idx2]

# No NA
train_ok <- complete.cases(X2_train) & !is.na(y2_train)
test_ok  <- complete.cases(X2_test)  & !is.na(y2_test)

X2_train <- X2_train[train_ok, , drop = FALSE]
y2_train <- y2_train[train_ok]

X2_test  <- X2_test[test_ok, , drop = FALSE]
y2_test  <- y2_test[test_ok]

# 6. 5-fold CV 
set.seed(123)
n_train2 <- nrow(X2_train)
K_folds <- 5
fold_id2 <- sample(rep(1:K_folds, length.out = n_train2))

k_values2 <- c(5, 10)
cv_acc2 <- numeric(length(k_values2))

for (j in seq_along(k_values2)) {
  k <- k_values2[j]
  fold_acc <- numeric(K_folds)
  
  for (f in 1:K_folds) {
    val_idx <- which(fold_id2 == f)
    tr_idx  <- setdiff(1:n_train2, val_idx)
    
    X_tr_f <- X2_train[tr_idx, , drop = FALSE]
    X_val_f <- X2_train[val_idx, , drop = FALSE]
    y_tr_f <- y2_train[tr_idx]
    y_val_f <- y2_train[val_idx]
    
    # Scale within each fold
    means_f <- apply(X_tr_f, 2, mean)
    sds_f   <- apply(X_tr_f, 2, sd)
    
    X_tr_f_sc  <- scale(X_tr_f,  center = means_f, scale = sds_f)
    X_val_f_sc <- scale(X_val_f, center = means_f, scale = sds_f)
    
    pred_val <- knn(
      train = X_tr_f_sc,
      test  = X_val_f_sc,
      cl    = y_tr_f,
      k     = k
    )
    
    tab_val <- table(Predicted = pred_val, Actual = y_val_f)
    fold_acc[f] <- sum(diag(tab_val)) / sum(tab_val)
  }
  
  cv_acc2[j] <- mean(fold_acc)
}

# 7. CV results and best k 
knn2_cv_results <- data.frame(
  k = k_values2,
  cv_accuracy = cv_acc2
)
knn2_cv_results

best_idx2 <- which.max(cv_acc2)
best_k2 <- k_values2[best_idx2]
best_k2

best_cv_accuracy2 <- cv_acc2[best_idx2]
best_cv_accuracy2

# 8. Final KNN with numeric + categorical on full train, evaluate on test

train_means2 <- apply(X2_train, 2, mean)
train_sds2   <- apply(X2_train, 2, sd)

X2_train_sc <- scale(X2_train, center = train_means2, scale = train_sds2)
X2_test_sc  <- scale(X2_test,  center = train_means2, scale = train_sds2)

best_pred2 <- knn(
  train = X2_train_sc,
  test  = X2_test_sc,
  cl    = y2_train,
  k     = best_k2
)

best_tab2 <- table(Predicted = best_pred2, Actual = y2_test)
best_tab2

best_test_accuracy2 <- sum(diag(best_tab2)) / sum(best_tab2)
best_test_accuracy2

# 9. Simple barplot: CV accuracy for k = 5 vs 10
barplot(
  cv_acc2, names.arg = k_values2,
  xlab = "Number of Neighbors (k)",
  ylab = "5-fold CV Accuracy",
  main = "KNN with numeric + categorical (k = 5 vs 10)"
)

