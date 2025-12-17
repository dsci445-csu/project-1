# Project Title: Student Depression Classification
This repository contains the R Markdown file (Real Document.Rmd) used for the DSCI 445 Final Project, comparing four machine learning models (LASSO, Logistic Regression, Random Forest, and K-Nearest Neighbors) to predict depression status in a student population.

## 1. Data Source
The project relies on a single dataset, which must be placed in the root directory of this project folder:

File Name: student_depression_dataset.csv

Source Link: https://www.kaggle.com/datasets/adilshamim8/student-depression-dataset?resource=download

Scroll down to where the download button is adjacent to the data set name: student_depression_dataset.csv


Please ensure this file is present in the same directory as the R Markdown file.

## 2. Required R Packages
All packages used in the analysis must be installed before running the script. You can install all necessary packages by running the following command in your R console:

install.packages(c("readr", "dplyr", "stringr", "tidymodels", "rsample", "tune", "randomForest", "caret", "class", "tidyverse", "forcats", "knitr", "ggplot2")

## 3. Steps for Reproduction
Follow these steps to reproduce the entire analysis, model tuning, and final results:
Open Project: Open the R Markdown file named Real Document.Rmd in RStudio.
Ensure Data is Present: Verify that student_depression_dataset.csv is in the working directory.
Run All Code Chunks: Use the "Run All Chunks" button in RStudio (or select Run $\rightarrow$ Run All from the menu).
Verify Results: The script will output all key steps directly to the console or the knitted document, including: Final Test Set Accuracy for each model (LASSO, LR, RF, KNN). The confusion matrices for the final selected models. The LASSO tuning plot showing the selection of the optimal $\lambda$.

## 4. Key Reproducibility Note
A consistent seed (set.seed(445)) is used at the beginning of all modeling procedures to ensure that the data splitting (Train/Test) and subsequent random processes (like Random Forest initialization and Cross-Validation folds) are identical upon every execution.

## 5. Project Structure
Real Document.Rmd: The primary R Markdown file containing all data cleaning, EDA, model tuning, and final evaluation code.

student_depression_dataset.csv: The raw dataset used for the analysis (must be in the root directory).

Paper-DSCI-Refined.pdf: The formal written report based on the analysis.

Presentation-DSCI-445: The visual presentation used for the project defense.

