## ----echo = FALSE, message = FALSE-----------------------------------------------------------------------------------------------
library(tidyverse)
library(scales)
library(ISLR)
library(knitr)



## --------------------------------------------------------------------------------------------------------------------------------
library(tidymodels) ## load library

## load the data in
ads <- read_csv("../data/Advertising.csv", col_select = -1) 

## fit the model
lm_spec <- linear_reg() |>
  set_mode("regression") |>
  set_engine("lm")

slr_fit <- lm_spec |>
  fit(sales ~ TV, data = ads)

slr_fit |>
  pluck("fit") |>
  summary()


## --------------------------------------------------------------------------------------------------------------------------------
# mlr_fit <- lm_spec |> fit(sales ~ TV + radio + newspaper, data = ads) 
mlr_fit <- lm_spec |> 
  fit(sales ~ ., data = ads) 

mlr_fit |>
  pluck("fit") |>
  summary()


## --------------------------------------------------------------------------------------------------------------------------------
# model with TV, radio, and newspaper
mlr_fit |> pluck("fit") |> summary()


## --------------------------------------------------------------------------------------------------------------------------------
# model without newspaper
lm_spec |> fit(sales ~ TV + radio, data = ads) |>
  pluck("fit") |> summary()


## ---- fig.height=2---------------------------------------------------------------------------------------------------------------
ggplot() +
  geom_point(aes(mlr_fit$fit$fitted.values, mlr_fit$fit$residuals))


## --------------------------------------------------------------------------------------------------------------------------------
head(mpg)


## ---- message = FALSE, fig.height=9, cache=TRUE----------------------------------------------------------------------------------
library(GGally)

mpg %>% 
  select(-model) %>% # too many models
  ggpairs() # plot matrix


## --------------------------------------------------------------------------------------------------------------------------------
lm_spec |>
  fit(hwy ~ displ + cty + drv, data = mpg) |>
  pluck("fit") |>
  summary()


## --------------------------------------------------------------------------------------------------------------------------------
lm_spec |>
  fit(sales ~ TV + radio + TV*radio, data = ads) |>
  pluck("fit") |>
  summary()


## --------------------------------------------------------------------------------------------------------------------------------
ggplot(data = mpg, aes(displ, hwy)) +
  geom_point() +
  geom_smooth(method = "lm", colour = "red") +
  geom_smooth(method = "loess", colour = "blue")


## --------------------------------------------------------------------------------------------------------------------------------
lm_spec |>
  fit(hwy ~ displ + I(displ^2), data = mpg) |>
  pluck("fit") |> summary()


## ---- fig.show="hold", out.width = "33%", fig.height = 7-------------------------------------------------------------------------
set.seed(445) #reproducibility

## generate data
x <- rnorm(100, 4, 1) # pick some x values
y <- 0.5 + x + 2*x^2 + rnorm(100, 0, 2) # true relationship
df <- data.frame(x = x, y = y) # data frame of training data

knn_spec <- nearest_neighbor(mode = "regression")
for (k in seq(2, 10, by = 2)) {
  knn_spec |>
    fit(y ~ x, data = df, neighbors = k) |>
    augment(new_data = df) |>
    ggplot() +
    geom_point(aes(x, y)) +
    geom_line(aes(x, .pred), colour = "red") +
    ggtitle(paste("KNN, k = ", k)) +
    theme(text = element_text(size = 30)) -> p
    
  print(p)
}

lm_spec |>
  fit(y ~ x, df) |>
  augment(new_data = df) |>
  ggplot() +
    geom_point(aes(x, y)) +
    geom_line(aes(x, .pred), colour = "red") +
    ggtitle("Simple Linear Regression") +
    theme(text = element_text(size = 30)) # slr plot


