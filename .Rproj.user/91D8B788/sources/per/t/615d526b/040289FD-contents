## Basic R ----------------------------------------------------------------
# 1. Use the `rep()` function to construct the following vector: `1 1 2 2 3 3 4 4 5 5`
rep(1:5, each = 2)

# 2. Use `rep()` to construct this vector: `1 2 3 4 5 1 2 3 4 5 1 2 3 4 5`
rep(1:5, times = 3)

# 3. Create a vector of 1300 values evenly spaced between 1 and 100.
x <- seq(1, 100, length.out = 1300)

# 4. How many of these values are greater than 91? (Hint: see `sum()` as a helpful functions.)
sum(x > 91)

# Using the vector you created of 1300 values evenly spaced between 1 and 100,
# 5. Modify the elements greater than 90 to equal 9999.
x[x > 90] <- 9999

# 6. View (not modify) the first 10 values in your vector.
head(x, 10)

# 7. View (not modify) the last 10 values in your vector.
tail(x, 10)

## Data Frames --------------------------------------------------------------
# 1. Make a data frame with column 1: `1,2,3,4,5,6` and column 2: `a,b,a,b,a,b`
df <- data.frame(nums = 1:6,
                 lets = rep(c("a", "b"), times = 3))

# 2. Select only rows with value "a" in column 2 using logical vector
df[df$lets == "a", ]

# 3. `mtcars` is a built-in data set like `iris`: Extract the 4th row of the mtcars data.
mtcars[4, ]

## Functions ----------------------------------------------------------------
# 1. Make a function called `my_mean()` that takes a vector of numbers as input and returns the mean of the vector.
my_mean <- function(x) {
  sum(x)/length(x)
}

# 2. Alter your `my_mean()` function to take a second argument (`na.rm`) with default value `FALSE` that removes `NA` values if `TRUE`.
my_mean <- function(x, na.rm = FALSE) {
  if(na.rm) {
    sum(na.omit(x))/length(x)
  } else {
    sum(x)/length(x)
  }
}

# 3. Add checks to your function to make sure the input data is either numeric or logical. If it is logical convert it to numeric.
my_mean <- function(x, na.rm = FALSE) {
  if(!(is.numeric(x) | is.logical(x))) stop("x must be numeric or logical")
  if(is.logical(x)) x <- as.numeric(x)
  
  if(na.rm) {
    sum(na.omit(x))/length(x)
  } else {
    sum(x)/length(x)
  }
}

# 4. The diamonds data set is included in the `ggplot2` package (not by default in `R`). It can be read into your environment with the following function. Loop over the columns of the diamonds data set and apply your mean function to all of the numeric columns (Hint: look at the `class()` function).
data("diamonds", package = "ggplot2")
for(col_name in colnames(diamonds)) {
  col <- diamonds[, col_name]
  if(is.numeric(col)) {
    print(paste(col_name, my_mean(col, na.rm = TRUE)))
  }
}
