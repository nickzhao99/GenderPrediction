#install libraries if needed
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(glmnet)) install.packages("glmnet", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(readr)) install.packages("readr", repos = "http://cran.us.r-project.org")
if(!require(glmnet)) install.packages("glmnet", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(knitr)) install.packages("knitr", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(rpart)) install.packages("rpart", repos = "http://cran.us.r-project.org")

# Load libraries
library(tidyverse)
library(caret)
library(data.table)
library(dplyr)
library(readr)
library(glmnet)
library(randomForest)
library(knitr)
library(ggplot2)
library(rpart)

# Download dataset
download.file(
  "https://github.com/nickzhao99/genderpred/raw/master/data_masked.csv", "~/df.csv")

# Import csv file as df
df <- read_csv("~/df.csv")
# Set seed as 1 to ensure same results across all iterations
set.seed(1)


# finds most common name and sort in descending order in kable format
df %>% group_by(Name) %>% count() %>% arrange(desc(n)) %>% head() %>% kable()

# counts each gender and present in kable format
df %>% group_by(Gender) %>% count() %>% kable()

# Creates a dummy variable for female
df <- df %>% mutate(female = case_when(Gender =="Female" ~1, Gender =="Male"~0))

# Converts female to a factor variable.
df$female <- as.factor(df$female)

# Converts names to lower case
df$Name <- tolower(df$Name)

# demonstration of match function, which converts alphabet to letters
match("a", letters)
match("b", letters)
match("c", letters)

#############################################################################
# Creates a new column for each letter in the Name column, up to eight.
# The letters are then converted to numbers for future possible Naive Bayes
# algorithms or decision rees.
df$one <- substr(df$Name, 1,1)
df$one <- match(df$one, letters)

df$two <- substr(df$Name, 2,2)
df$two <- match(df$two, letters)

df$three <- substr(df$Name, 3,3)
df$three <- match(df$three, letters)

df$four <- substr(df$Name, 4,4)
df$four <- match(df$four, letters)

df$five <- substr(df$Name, 5,5)
df$five <- match(df$five, letters)

df$six <- substr(df$Name, 6,6)
df$six <- match(df$six, letters)

df$seven <- substr(df$Name, 7,7)
df$seven <- match(df$seven, letters)

df$eight <- substr(df$Name, 8,8)
df$eight <- match(df$eight, letters)
#################################################################################
# Fills NA values with 0 in cases where number of letters in a name < 8
df[is.na(df)] <- 0


#split data into training and test data sets
indxTrain <- createDataPartition(y=df$female, p = 0.8, list = FALSE)
training <- df[indxTrain,]
testing <- df[-indxTrain,]

boxplot(df$one, df$two, df$three, df$four,
        main = "Boxplot of first to fourth letters",
        names = c("first","second", "third","fourth"),
        notch = TRUE)



# create x, a matrix consisting of first letter - eighth letter
x = as.matrix(training[,c(4:11)])
# create outcome y column
y = training$female



# our first model will be fit using naive bayes
model.nb1 <- naivebayes::naive_bayes(x,y)
# fits model using naive bayes with x as input, y as output
Predict <- predict(model.nb1,newdata = x)
# used to predict the training set
confusionMatrix(Predict, training$female)
# creates confusion matrix for the training set for the given algorithm.
# With naive bayes classifier, The accuracy is only 60.15%, a little better 
# than just guessing.

model.nb2 = naivebayes::nonparametric_naive_bayes(x,y)
# fits model using nonparametric naive bayes
Predict <- predict(model.nb2,newdata = x)
# used to predict the training set
confusionMatrix(Predict, training$female)
# creates confusion matrix for the training set for the given algorithm.

# With non-parametric naive bayes classifier, our accuracy 
# increased to 70.47%. But, can we do any better?

model.logit <- glmnet(x, y, family = "binomial") #apply logistic equation
Predict <- predict(model.logit,newx = x) # apply predict function
predicted.classes <- ifelse(Predict > 0.5, "1", "0") #if p > 0.5, female, else male
observed.classes <- training$female # checks observed classes
mean(predicted.classes == observed.classes)# calculates accuracy

#Logistic regression only has an accuracy of 57.61%! Time to move on.

# fit model using classification tree
model.ct <- rpart(female ~ one + two + three + four + five + six + seven + eight, data=training, method = 'class')
# make prediction on training dataset
Predict <- predict(model.ct, newdata= training, type = 'class')
mean(Predict == training$female) # calculates accuracy
#With classification tree, our accuracy increased to 82.86%.


model.rf = randomForest(x,y)
# fits model using randomForest
Predict <- predict(model.rf,newdata = x)
# used to predict the training set
confusionMatrix(Predict, training$female)
# creates confusion matrix for the training set for the given algorithm.


# With RandomForest, our accuracy increased to 96.77%! 
# We will apply this to the testing dataset.


# apply alg to testing dataset
Predict <- predict(model.rf,newdata = as.matrix(testing[,c(4:11)]))
# create confusion matrix on testing dataset
confusionMatrix(Predict, testing$female)


# As we can see, our ﬁnal accuracy is 96%. This shows that 
# we did not overtrain our algorithm. 

