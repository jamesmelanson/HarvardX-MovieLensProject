#Run the HarvardX script to retrieve the MovieLens data and perform an initial cleaning of the data.
source('~/test/CreateTrainandValidationSets.R')

#Load packages, installing if needed.
if(!require(caret)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
library(caret)
library(lubridate)
library(tidyverse)


#Subsetting of the edx dataframe into training and test sets:

#Taking the edx set that we are allowed to work with, and subsetting it into training and validation sets.
#The training set created from the edx dataframe will contain approximately 90% the original dataframe's data; with 10% of the data
#being held out to test the accuracy of our model.

set.seed(1, sample.kind = "Rounding")
test_index <- createDataPartition(edx$rating, times = 1, p = 0.1, list = FALSE)
train <- edx[-test_index,]
temp <- edx[test_index,]

#Making sure to remove users and movies from the test set that don't appear in the training set.
test_set <- temp %>%
semi_join(train, by = "movieId") %>%
semi_join(train, by = "userId")

#Adding the users and movies removed from the test set back into the training set.
removed <- temp %>% anti_join(test_set)
train <- rbind(train, removed)
rm(removed, temp, test_index)

#Defining a function to calculate the root mean squared error (RMSE)
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings-predicted_ratings)^2))
}

#Model 1: Predict the average user's rating
mu <- mean(train$rating)
predicted_ratings <- mutate(test_set, pred = mu)
rmse_model_1 <- RMSE(predicted_ratings$rating, predicted_ratings$pred)

rmse_results <- tibble(Method = "Predict the average", RMSE = rmse_model_1, "RMSE - Target RMSE" = rmse_model_1 - 0.86490)
rmse_results %>% knitr::kable()

#Model 2: Movie effect
mu <- mean(train$rating)

b_movie <- train %>%
    group_by(movieId) %>%
    summarize(b_movie = mean(rating-mu))
  
predicted_ratings <- test_set %>%
    left_join(b_movie, by = "movieId") %>%
    mutate(pred = mu + b_movie)
rmse_model_2 <- RMSE(predicted_ratings$pred, predicted_ratings$rating)

rmse_results <- bind_rows(rmse_results, tibble(Method = "Movie effect", RMSE = rmse_model_2, "RMSE - Target RMSE" = rmse_model_2 - 0.86490))

#Model 3: Movie effect (regularized)
mu <- mean(train$rating)
lambdas <- seq(0, 10, 1)

#Picking a lambda by searching for the minimum RMSE using this data set.
rmses <- sapply(lambdas, function(l){
  b_movie <- train %>%
    group_by(movieId) %>%
    summarize(b_movie = sum(rating-mu)/(n()+l))
  
  predicted_ratings <- test_set %>%
    left_join(b_movie, by = "movieId") %>%
    mutate(pred = mu + b_movie)
  
  RMSE(predicted_ratings$pred, predicted_ratings$rating)
})

lambda <- lambdas[which.min(rmses)]
rmse_model_3 <- min(rmses)

rmse_results <- bind_rows(rmse_results, tibble(Method = "Movie effect (regularized)", RMSE = rmse_model_3, "RMSE - Target RMSE" = rmse_model_3 - 0.86490))

#Model 4: Movie effects + User effects (Regularized)
mu <- mean(train$rating)
lambdas <- seq(0, 10, 1)

#Picking a lambda by searching for the minimum using this data set.
rmses <- sapply(lambdas, function(l){
b_movie <- train %>%
  group_by(movieId) %>%
  summarize(b_movie = sum(rating-mu)/(n()+l))

b_user <- train %>% 
  left_join(b_movie, by = "movieId") %>% 
  group_by(userId) %>%
  summarize(b_user = sum(rating-b_movie-mu)/(n()+l))

predicted_ratings <- test_set %>%
  left_join(b_movie, by = "movieId") %>%
  left_join(b_user, by = "userId") %>%
  mutate(pred = mu + b_movie + b_user)

RMSE(predicted_ratings$pred, predicted_ratings$rating)
})

lambda <- lambdas[which.min(rmses)]
rmse_model_4 <- min(rmses)

rmse_results <- bind_rows(rmse_results, tibble(Method = "Movie and user effects (regularized)", RMSE = rmse_model_4, "RMSE - Target RMSE" = rmse_model_4 - 0.86490))

#Model 5: Movie effects + User effects (regularized with separate lambdas for movie effects and user effects)
mu <- mean(train$rating)
lambdas <- expand.grid(l1 = seq(0,10,2), l2 = seq(0,10,2))

#Picking a lambda by searching for the minimum using this data set.
rmses <- mapply(function(l1, l2){
  b_movie <- train %>%
    group_by(movieId) %>%
    summarize(b_movie = sum(rating-mu)/(n()+l1))
  
  b_user <- train %>% 
    left_join(b_movie, by = "movieId") %>% 
    group_by(userId) %>%
    summarize(b_user = sum(rating-b_movie-mu)/(n()+l2))
  
  predicted_ratings <- test_set %>%
    left_join(b_movie, by = "movieId") %>%
    left_join(b_user, by = "userId") %>%
    mutate(pred = mu + b_movie + b_user)
  
  RMSE(predicted_ratings$pred, predicted_ratings$rating)
}, lambdas$l1, lambdas$l2)

lambda <- lambdas[which.min(rmses),]
rmse_model_5 <- min(rmses)

rmse_results <- bind_rows(rmse_results, tibble(Method = "Movie and user effects (regularized, separate lambdas for movie effects and user effects)", RMSE = rmse_model_5, "RMSE - Target RMSE" = rmse_model_5 - 0.86490))

#Model 6: Movie effects + User effects + Genre effects (regularized with separate lambdas for movie, user, and genre effects)
mu <- mean(train$rating)
lambdas <- expand.grid(l1 = seq(2,6,2), l2 = seq(2,6,2), l3 = seq(2,6,2))

#Picking a lambda by searching for the minimum using this data set.
rmses <- mapply(function(l1, l2, l3){
  b_movie <- train %>%
    group_by(movieId) %>%
    summarize(b_movie = sum(rating-mu)/(n()+l1))
  
    b_user <- train %>% 
    left_join(b_movie, by = "movieId") %>% 
    group_by(userId) %>%
    summarize(b_user = sum(rating-b_movie-mu)/(n()+l2))
  
  b_genre <- train %>%
    left_join(b_movie, by = "movieId") %>%
    left_join(b_user, by = "userId") %>%
    group_by(genres) %>%
    summarize(b_genre = sum(rating-b_movie-b_user-mu)/(n()+l3))
  
  predicted_ratings <- test_set %>%
    left_join(b_movie, by = "movieId") %>%
    left_join(b_user, by = "userId") %>%
    left_join(b_genre, by = "genres")
  
  predicted_ratings <- predicted_ratings %>%
    mutate(pred = mu + b_movie + b_user + b_genre)
  
  RMSE(predicted_ratings$pred, predicted_ratings$rating)
}, lambdas$l1, lambdas$l2, lambdas$l3)

lambda <- lambdas[which.min(rmses),]
rmse_model_6 <- min(rmses)

rmse_results <- bind_rows(rmse_results, tibble(Method = "Movie, user, and genre effects (regularized, seperate lambdas for movie, user, and genre effects)", RMSE = rmse_model_6, "RMSE - Target RMSE" = rmse_model_6 - 0.86490))

#Model 7: Movie effects + User effects + Genre effects + Week effects (Regularized)
mu <- mean(train$rating)
lambdas <- seq(4,8,2)

#Picking a lambda by searching for the minimum using this data set.
rmses <- sapply(lambdas, function(l){
  b_movie <- train %>%
    group_by(movieId) %>%
    summarize(b_movie = sum(rating-mu)/(n()+l))
  
    b_user <- train %>% 
    left_join(b_movie, by = "movieId") %>% 
    group_by(userId) %>%
    summarize(b_user = sum(rating-b_movie-mu)/(n()+l))
  
  b_genre <- train %>%
    mutate(date = as_datetime(timestamp)) %>%
    mutate(week = round_date(date, unit = "week")) %>%
    left_join(b_movie, by = "movieId") %>%
    left_join(b_user, by = "userId") %>%
    group_by(genres) %>%
    summarize(b_genre = sum(rating-b_movie-b_user-mu)/(n()+l))
  
  b_week <- train %>%
    left_join(b_movie, by = "movieId") %>%
    left_join(b_user, by = "userId") %>%
    left_join(b_genre, by = "genres") %>%
    mutate(date = as_datetime(timestamp)) %>%
    mutate(week = round_date(date, unit = "week")) %>%
    group_by(week) %>%
    summarize(b_week = sum(rating-b_movie-b_user-b_genre-mu)/(n()+l))
  
  predicted_ratings <- test_set %>%
    mutate(date = as_datetime(timestamp)) %>%
    mutate(week = round_date(date, unit = "week")) %>%
    left_join(b_movie, by = "movieId") %>%
    left_join(b_user, by = "userId") %>%
    left_join(b_genre, by = "genres") %>%
    left_join(b_week, by = "week") %>%
    mutate(pred = mu + b_movie + b_user + b_genre + b_week)
  
  RMSE(predicted_ratings$pred, predicted_ratings$rating)
})

lambda <- lambdas[which.min(rmses)]
rmse_model_7 <- min(rmses)

rmse_results <- bind_rows(rmse_results, tibble(Method = "Movie, user, genre, and week effects (regularized)", RMSE = rmse_model_7, "RMSE - Target RMSE" = rmse_model_7 - 0.86490))

#Model 8: Movie effects + User effects + Genre effects + Week effects + Year released effects (Regularized) 
mu <- mean(train$rating)
lambdas <- seq(4,6,2)

##Picking a lambda by searching for the minimum using this data set.
rmses <- sapply(lambdas, function(l){
  b_movie <- train %>%
    group_by(movieId) %>%
    summarize(b_movie = sum(rating-mu)/(n()+l))
  
  b_user <- train %>% 
    left_join(b_movie, by = "movieId") %>% 
    group_by(userId) %>%
    summarize(b_user = sum(rating-b_movie-mu)/(n()+l))
  
  b_genre <- train %>%
    left_join(b_movie, by = "movieId") %>%
    left_join(b_user, by = "userId") %>%
    group_by(genres) %>%
    summarize(b_genre = sum(rating-b_movie-b_user-mu)/(n()+l))
  
  b_week <- train %>%
    left_join(b_movie, by = "movieId") %>%
    left_join(b_user, by = "userId") %>%
    left_join(b_genre, by = "genres") %>%
    mutate(date = as_datetime(timestamp)) %>%
    mutate(week = round_date(date, unit = "week")) %>%
    group_by(week) %>%
    summarize(b_week = sum(rating-b_movie-b_user-b_genre-mu)/(n()+l))
  
  b_year_released <- train %>%
    left_join(b_movie, by = "movieId") %>%
    left_join(b_user, by = "userId") %>%
    left_join(b_genre, by = "genres") %>%
    mutate(date = as_datetime(timestamp)) %>%
    mutate(week = round_date(date, unit = "week")) %>%
    left_join(b_week, by = "week") %>%
    mutate(year = str_extract(title, "\\(\\d\\d\\d\\d\\)")) %>%
    mutate(year = as.numeric(str_extract(year, "\\d\\d\\d\\d"))) %>%
    group_by(year) %>%
    summarize(b_year_released = sum(rating-b_movie-b_user-b_genre-b_week-mu)/(n()+l))

  predicted_ratings <- test_set %>%
    left_join(b_movie, by = "movieId") %>%
    left_join(b_user, by = "userId") %>%
    left_join(b_genre, by = "genres") %>%
    mutate(date = as_datetime(timestamp)) %>%
    mutate(week = round_date(date, unit = "week")) %>%
    left_join(b_week, by = "week") %>%
    mutate(year = str_extract(title, "\\(\\d\\d\\d\\d\\)")) %>%
    mutate(year = as.numeric(str_extract(year, "\\d\\d\\d\\d"))) %>%
    left_join(b_year_released, by = "year") %>%
    mutate(pred = mu + b_movie + b_user + b_genre + b_week + b_year_released)
  
  RMSE(predicted_ratings$pred, predicted_ratings$rating)
})
lambda <- lambdas[which.min(rmses)]
rmse_model_8 <- min(rmses)

rmse_results <- bind_rows(rmse_results, tibble(Method = "Movie, user, genre, week, and year released effects (regularized)", RMSE = rmse_model_8, "RMSE - Target RMSE" = rmse_model_8 - 0.86490))

#Model 9.1: Movie effects + User effects + Genre effects + Week effects + Year released + Time Post-Movie Release effects (Regularized) 
mu <- mean(train$rating)
lambdas <- seq(5,7,0.25)

##Picking a lambda by searching for the minimum using this data set.
rmses <- sapply(lambdas, function(l){
  b_movie <- train %>%
    group_by(movieId) %>%
    summarize(b_movie = sum(rating-mu)/(n()+l))

  b_user <- train %>% 
    left_join(b_movie, by = "movieId") %>% 
    group_by(userId) %>%
    summarize(b_user = sum(rating-b_movie-mu)/(n()+l))
  
  b_genre <- train %>%
    left_join(b_movie, by = "movieId") %>%
    left_join(b_user, by = "userId") %>%
    group_by(genres) %>%
    summarize(b_genre = sum(rating-b_movie-b_user-mu)/(n()+l))
  
  b_week <- train %>%
    left_join(b_movie, by = "movieId") %>%
    left_join(b_user, by = "userId") %>%
    left_join(b_genre, by = "genres") %>%
    mutate(date = as_datetime(timestamp)) %>%
    mutate(week = round_date(date, unit = "week")) %>%
    group_by(week) %>%
    summarize(b_week = sum(rating-b_movie-b_user-b_genre-mu)/(n()+l))
  
  b_year_released <- train %>%
    left_join(b_movie, by = "movieId") %>%
    left_join(b_user, by = "userId") %>%
    left_join(b_genre, by = "genres") %>%
    mutate(date = as_datetime(timestamp)) %>%
    mutate(week = round_date(date, unit = "week")) %>%
    left_join(b_week, by = "week") %>%
    mutate(year = str_extract(title, "\\(\\d\\d\\d\\d\\)")) %>%
    mutate(year = as.numeric(str_extract(year, "\\d\\d\\d\\d"))) %>%
    group_by(year) %>%
    summarize(b_year_released = sum(rating-b_movie-b_user-b_genre-b_week-mu)/(n()+l))
  
  b_time <- train %>%
    left_join(b_movie, by = "movieId") %>%
    left_join(b_user, by = "userId") %>%
    left_join(b_genre, by = "genres") %>%
    mutate(date = as_datetime(timestamp)) %>%
    mutate(week = round_date(date, unit = "week")) %>%
    left_join(b_week, by = "week") %>%
    mutate(year = str_extract(title, "\\(\\d\\d\\d\\d\\)")) %>%
    mutate(year = as.numeric(str_extract(year, "\\d\\d\\d\\d"))) %>%
    left_join(b_year_released, by = "year") %>%
    mutate(yearreviewed = year(as_datetime(timestamp))) %>%
    mutate(yearselapsed = yearreviewed-year) %>%
    group_by(yearselapsed) %>%
    summarize(b_time = sum(rating-b_movie-b_user-b_genre-b_week-b_year_released-mu)/(n()+l))
  
  predicted_ratings <- test_set %>%
    left_join(b_movie, by = "movieId") %>%
    left_join(b_user, by = "userId") %>%
    left_join(b_genre, by = "genres") %>%
    mutate(date = as_datetime(timestamp)) %>%
    mutate(week = round_date(date, unit = "week")) %>%
    left_join(b_week, by = "week") %>%
    mutate(year = str_extract(title, "\\(\\d\\d\\d\\d\\)")) %>%
    mutate(year = as.numeric(str_extract(year, "\\d\\d\\d\\d"))) %>%
    left_join(b_year_released, by = "year") %>%
    mutate(yearreviewed = year(as_datetime(timestamp))) %>%
    mutate(yearselapsed = yearreviewed-year) %>%
    left_join(b_time, by = "yearselapsed") %>%
    mutate(pred = mu + b_movie + b_user + b_genre + b_week + b_year_released + b_time)
  
  RMSE(predicted_ratings$pred, predicted_ratings$rating)
})
lambda <- lambdas[which.min(rmses)]
rmse_model_9_1 <- min(rmses)

data.frame(lambdas, rmses) %>% ggplot(aes(x = lambdas, y = rmses)) + geom_point() + labs(title = "Model 9.1 - RMSEs versus Lambdas")
rmse_results <- bind_rows(rmse_results, tibble(Method = "Movie, user, genre, week, year released, and time post-movie release effects (regularized, version 1)", RMSE = rmse_model_9_1, "RMSE - Target RMSE" = rmse_model_9_1 - 0.86490))

## Can we further refine the lambda identified in the previous model? Attempting to get better resolution on
## the bottom of the RMSE versus lambda curve, by reducing the lambda search range and reducing the step size between
## lambdas.

#Model 9.2: Movie effects + User effects + Genre effects + Week effects + Year released + Time Post-Movie Release effects (Regularized) 
mu <- mean(train$rating)
lambdas <- seq(5.1, 5.3, 0.05)

#Picking a lambda by searching for the minimum using this data set.
rmses <- sapply(lambdas, function(l){
  b_movie <- train %>%
    group_by(movieId) %>%
    summarize(b_movie = sum(rating-mu)/(n()+l))
  
  b_user <- train %>% 
    left_join(b_movie, by = "movieId") %>% 
    group_by(userId) %>%
    summarize(b_user = sum(rating-b_movie-mu)/(n()+l))
  
  b_genre <- train %>%
    left_join(b_movie, by = "movieId") %>%
    left_join(b_user, by = "userId") %>%
    group_by(genres) %>%
    summarize(b_genre = sum(rating-b_movie-b_user-mu)/(n()+l))
  
  b_week <- train %>%
    left_join(b_movie, by = "movieId") %>%
    left_join(b_user, by = "userId") %>%
    left_join(b_genre, by = "genres") %>%
    mutate(date = as_datetime(timestamp)) %>%
    mutate(week = round_date(date, unit = "week")) %>%
    group_by(week) %>%
    summarize(b_week = sum(rating-b_movie-b_user-b_genre-mu)/(n()+l))
  
  b_year_released <- train %>%
    left_join(b_movie, by = "movieId") %>%
    left_join(b_user, by = "userId") %>%
    left_join(b_genre, by = "genres") %>%
    mutate(date = as_datetime(timestamp)) %>%
    mutate(week = round_date(date, unit = "week")) %>%
    left_join(b_week, by = "week") %>%
    mutate(year = str_extract(title, "\\(\\d\\d\\d\\d\\)")) %>%
    mutate(year = as.numeric(str_extract(year, "\\d\\d\\d\\d"))) %>%
    group_by(year) %>%
    summarize(b_year_released = sum(rating-b_movie-b_user-b_genre-b_week-mu)/(n()+l))
  
  b_time <- train %>%
    left_join(b_movie, by = "movieId") %>%
    left_join(b_user, by = "userId") %>%
    left_join(b_genre, by = "genres") %>%
    mutate(date = as_datetime(timestamp)) %>%
    mutate(week = round_date(date, unit = "week")) %>%
    left_join(b_week, by = "week") %>%
    mutate(year = str_extract(title, "\\(\\d\\d\\d\\d\\)")) %>%
    mutate(year = as.numeric(str_extract(year, "\\d\\d\\d\\d"))) %>%
    left_join(b_year_released, by = "year") %>%
    mutate(yearreviewed = year(as_datetime(timestamp))) %>%
    mutate(yearselapsed = yearreviewed-year) %>%
    group_by(yearselapsed) %>%
    summarize(b_time = sum(rating-b_movie-b_user-b_genre-b_week-b_year_released-mu)/(n()+l))
  
  predicted_ratings <- test_set %>%
    left_join(b_movie, by = "movieId") %>%
    left_join(b_user, by = "userId") %>%
    left_join(b_genre, by = "genres") %>%
    mutate(date = as_datetime(timestamp)) %>%
    mutate(week = round_date(date, unit = "week")) %>%
    left_join(b_week, by = "week") %>%
    mutate(year = str_extract(title, "\\(\\d\\d\\d\\d\\)")) %>%
    mutate(year = as.numeric(str_extract(year, "\\d\\d\\d\\d"))) %>%
    left_join(b_year_released, by = "year") %>%
    mutate(yearreviewed = year(as_datetime(timestamp))) %>%
    mutate(yearselapsed = yearreviewed-year) %>%
    left_join(b_time, by = "yearselapsed") %>%
    mutate(pred = mu + b_movie + b_user + b_genre + b_week + b_year_released + b_time)
  
  RMSE(predicted_ratings$pred, predicted_ratings$rating)
})

model_9_2_lambda <- lambdas[which.min(rmses)]
rmse_model_9_2 <- min(rmses)

rmse_results <- bind_rows(rmse_results, tibble(Method = "Movie, user, genre, week, year released, and time post-movie release effects (regularized, version 2)", RMSE = rmse_model_9_2, "RMSE - Target RMSE" = rmse_model_9_2 - 0.86490))

#Validation of the model's accuracy against the hold-out test set
mu <- mean(train$rating)
lambdas <- model_9_2_lambda

rmse_validation <- sapply(lambdas, function(l){
  b_movie <- train %>%
    group_by(movieId) %>%
    summarize(b_movie = sum(rating-mu)/(n()+l))
  
  b_user <- train %>% 
    left_join(b_movie, by = "movieId") %>% 
    group_by(userId) %>%
    summarize(b_user = sum(rating-b_movie-mu)/(n()+l))
  
  b_genre <- train %>%
    left_join(b_movie, by = "movieId") %>%
    left_join(b_user, by = "userId") %>%
    group_by(genres) %>%
    summarize(b_genre = sum(rating-b_movie-b_user-mu)/(n()+l))
  
  b_week <- train %>%
    left_join(b_movie, by = "movieId") %>%
    left_join(b_user, by = "userId") %>%
    left_join(b_genre, by = "genres") %>%
    mutate(date = as_datetime(timestamp)) %>%
    mutate(week = round_date(date, unit = "week")) %>%
    group_by(week) %>%
    summarize(b_week = sum(rating-b_movie-b_user-b_genre-mu)/(n()+l))
  
  b_year_released <- train %>%
    left_join(b_movie, by = "movieId") %>%
    left_join(b_user, by = "userId") %>%
    left_join(b_genre, by = "genres") %>%
    mutate(date = as_datetime(timestamp)) %>%
    mutate(week = round_date(date, unit = "week")) %>%
    left_join(b_week, by = "week") %>%
    mutate(year = str_extract(title, "\\(\\d\\d\\d\\d\\)")) %>%
    mutate(year = as.numeric(str_extract(year, "\\d\\d\\d\\d"))) %>%
    group_by(year) %>%
    summarize(b_year_released = sum(rating-b_movie-b_user-b_genre-b_week-mu)/(n()+l))
  
  b_time <- train %>%
    left_join(b_movie, by = "movieId") %>%
    left_join(b_user, by = "userId") %>%
    left_join(b_genre, by = "genres") %>%
    mutate(date = as_datetime(timestamp)) %>%
    mutate(week = round_date(date, unit = "week")) %>%
    left_join(b_week, by = "week") %>%
    mutate(year = str_extract(title, "\\(\\d\\d\\d\\d\\)")) %>%
    mutate(year = as.numeric(str_extract(year, "\\d\\d\\d\\d"))) %>%
    left_join(b_year_released, by = "year") %>%
    mutate(yearreviewed = year(as_datetime(timestamp))) %>%
    mutate(yearselapsed = yearreviewed-year) %>%
    group_by(yearselapsed) %>%
    summarize(b_time = sum(rating-b_movie-b_user-b_genre-b_week-b_year_released-mu)/(n()+l))
  
  predicted_ratings <- validation %>%
    left_join(b_movie, by = "movieId") %>%
    left_join(b_user, by = "userId") %>%
    left_join(b_genre, by = "genres") %>%
    mutate(date = as_datetime(timestamp)) %>%
    mutate(week = round_date(date, unit = "week")) %>%
    left_join(b_week, by = "week") %>%
    mutate(year = str_extract(title, "\\(\\d\\d\\d\\d\\)")) %>%
    mutate(year = as.numeric(str_extract(year, "\\d\\d\\d\\d"))) %>%
    left_join(b_year_released, by = "year") %>%
    mutate(yearreviewed = year(as_datetime(timestamp))) %>%
    mutate(yearselapsed = yearreviewed-year) %>%
    left_join(b_time, by = "yearselapsed") %>%
    mutate(pred = mu + b_movie + b_user + b_genre + b_week + b_year_released + b_time)
  
  RMSE(predicted_ratings$pred, predicted_ratings$rating)
})

rmse_results <- bind_rows(rmse_results, tibble(Method = "Validation of model 9.2 using hold-out test set", RMSE = rmse_validation, "RMSE - Target RMSE" = rmse_validation - 0.86490))
