# Setting up the directory
setwd("C:/Users/mishrsat/Desktop/Automation/codes/Edelweiss_Hackathon_Machine_Learning/Dataset")

# Calling the required Libraries
library(dplyr)
library(stringr)
library(tidyr)
library(data.table)
library(car)
library(Hmisc)
library(ROCR)
library(DMwR)
library(GGally)
library(ranger)
library(MASS)
library(caTools)
library(caret)
library(lubridate)
library(Information)
library(xgboost)
library(dummies)
library(Matrix)

# Preventing scientific notation
options(scipen=999)

# Loading the raw file.
train_set <- read.csv("train_foreclosure.csv", header = TRUE, sep = ',', stringsAsFactors = FALSE)
test_data <- read.csv("test_foreclosure.csv", header = TRUE, sep = ',', stringsAsFactors = FALSE)

# Adding the from file status and then combining the files.
train_set$from_set <- "Train"
test_data$from_set <- "Test"

combined_set <- rbind(train_set, test_data)

#-------------------------------------------
# Loading the other data files.
cust_demo <- read.csv("Customer_Demographics.csv", header = TRUE, sep = ',',
                      stringsAsFactors = FALSE, na.strings = c("","NA"))
cust_tran <- read.csv("Customer_Transaction.csv", header = TRUE, sep = ',',
                      stringsAsFactors = FALSE, na.strings = c("", "NA"))

#-------------------------------------------
# Checking for NA's in cust demo
sapply(cust_demo, function(x) sum(is.na(x)))

# We will remove profession and occupation column entierly as all of the entries are NA
cust_demo <- cust_demo[, c(1:3,5:9,11:15)]
cust_demo$CUSTOMERID <- as.character(cust_demo$CUSTOMERID)

# We don't have agreement id in cust_demo so we will take them from cust_tran
# Getting the unique customer id and agreement id's
storing_bothid <- unique(cust_tran[ , c(1:2)])


# Now taking all other variables from cust_demo dataset
storing_bothid <- left_join(storing_bothid, cust_demo, by.x = "CUSTOMERID", by.y = "CUSTOMERID", all.x = TRUE)

# Checking the NA's in storing bothid
sapply(storing_bothid, function(x) sum(is.na(x)))
# We see a lot of NA's in all of the columns, we will go ahead and join the cust_tran

#-------------------------------------------

# Changing the last receipt date to date format from character, we are doing so as to sort
# them on a descending order so as to get only the latest transaction from cust_tran
cust_tran$LAST_RECEIPT_DATE <- parse_date_time(cust_tran$LAST_RECEIPT_DATE, orders = "dmy")

# Creating a year column
cust_tran$Yearof_lastreceipt <- year(cust_tran$LAST_RECEIPT_DATE)

# Sorting the cust_tran on agreement id and last receipt date
cust_tran <- arrange(cust_tran, AGREEMENTID, desc(LAST_RECEIPT_DATE))

# Creating a column to store system date
cust_tran$system_date <- Sys.Date()

# Now we will substract last receipt date from system date
cust_tran$LAST_RECEIPT_DATE <- as.Date(cust_tran$LAST_RECEIPT_DATE)
cust_tran$days <- cust_tran$LAST_RECEIPT_DATE - cust_tran$system_date

# Now lets have a rank column
cust_tran$Ranking <- transform(cust_tran, Rank = ave(days, AGREEMENTID, FUN = function(x)
  rank (-x, ties.method ="first")))

# Ranking is stored as a dataframe inside a dataframe, so we will create a new dataframe
# and use that.
main_data <- cust_tran$Ranking

# Now subsetting wherever the rank is 1
main_data <- subset(main_data, Rank == 1)

# Removing column 40 to 42
main_data <- main_data[, c(1:39)]

# Now lets add all the columns into from storing both id file.
storing_bothid <- storing_bothid[, -2]

main_data <- left_join(main_data, storing_bothid, by.x = "AGREEMENTID", by.y = "AGREEMENT", all.x = TRUE)

# Now we will combine the combined set and main data to get the foreclousere and from set
main_data <- left_join(main_data, combined_set,  by.x = "AGREEMENTID", by.y = "AGREEMENT", all.x = TRUE)

# Now we will remove the ID columns except Agreement ID and date columns
# because we don't need them for our modelling purpose.
main_data <- main_data[, c(1,3:4,6:10,12:26,28:34,38:53)]

# Checking the number of NA's in each variable.
check_na <- as.data.frame(sapply(main_data, function(x) sum(is.na(x))))

#-------------------------------------------
# We don't need branch code as we already have city, so lets remove it.
main_data <- main_data[, -44]

#-------------------------------------------
# Feature Engineering
#-------------------------------------------
# We have our main dataset ready, now lets look at doing feature engineering


# Checking the structure
str(main_data)

# Changing loan amount and net amount to numeric
main_data$LOAN_AMT <- as.numeric(gsub(",","",main_data$LOAN_AMT, fixed = TRUE))
main_data$NET_DISBURSED_AMT <- as.numeric(gsub(",","", main_data$NET_DISBURSED_AMT, fixed = TRUE))

# As we will be building a logistic regression we will look at who we can
# treat the categorical variables first

#------------City Variable

levels(factor(main_data$CITY))
# There are 316 cities, we either have to take the state for this variable or
# have to cluster them into similar buckets.
# Lets cluster them

# To run k-means we need only numeric variables
for_city <- main_data[, c(2:8,10:29,31,9)]

# Lets scale the variables
for_city[, -c(29)] <- scale(for_city[, -c(29)])

# Creating a new dataframe
for_city1 <- for_city[, -29]

# Replacing the variables mean  
for(i in 1:ncol(for_city1)){
  for_city1[is.na(for_city1[,i]), i] <- mean(for_city1[,i], na.rm = TRUE)
}

# Now checking if there are any NA's
sapply(for_city1, function(x) sum(is.na(x))) # No NA's found

# Finding the optimal value of K
r_sq<- rnorm(20)

# Here we are creating a function to understand how many clusters will be good for accounts dataset
for (number in 1:20){clus <- kmeans(for_city1, centers = number, nstart = 50)
r_sq[number]<- clus$betweenss/clus$totss
}

# betweenss is the inter-cluster sum of squares of the distance
# and totss is measures the total spread in the data by calculating
# the total sum of squares of distance

plot(r_sq)
# We see that forming clusters with 10 would be good as that is at the bend.

# Creating the cluster
cluster10_city <- kmeans(for_city1, centers = 10, iter.max = 50, nstart = 50)

# Now lets add the cluster id to the main data
main_data <- cbind(main_data, cluster10_city$cluster)
colnames(main_data)[46]<- "City_ClusterID"

# Now we will remove the city column
main_data <- main_data[, -9]

#------------Product Variable

str(main_data)

# Checking the summary
summary(factor(main_data$PRODUCT))
#HL   LAP  STHL STLAP 
#5770 10306 12195  5083

# We will leave this as is

#------------Sex Variable

# Checking the summary
summary(factor(main_data$SEX))
#F       M    NA's 
# 1188  8418 23748

# We will change the NA's to missing
main_data$SEX[is.na(main_data$SEX)] <- "Missing"

#------------Marital Status Variable

# Checking the summary
summary(factor(main_data$MARITAL_STATUS))
# M       S    NA's 
# 8798   806 23750

# We will change the NA's to missing
main_data$MARITAL_STATUS[is.na(main_data$MARITAL_STATUS)] <- "Missing"

#------------Qualification Variable

# Checking the summary
summary(factor(main_data$QUALIFICATION))

# We will make null and NA's as missing
main_data$QUALIFICATION <- factor(gsub("null", "Missing", main_data$QUALIFICATION))
main_data$QUALIFICATION[is.na(main_data$QUALIFICATION)] <- "Missing"

main_data$QUALIFICATION <- as.character(main_data$QUALIFICATION)

#------------NO_OF_DEPENDENT Variable

# Checking the summary
summary(factor(main_data$NO_OF_DEPENDENT))

# We will change the NA's to missing
main_data$NO_OF_DEPENDENT[is.na(main_data$NO_OF_DEPENDENT)] <- ""

# Converting the variable to character
main_data$NO_OF_DEPENDENT <- as.numeric(main_data$NO_OF_DEPENDENT)

#------------Position Variable

# Checking the summary
summary(factor(main_data$POSITION))

main_data$POSITION <- as.factor(main_data$POSITION)

# Merging the levels to appropriate buckets
levels(main_data$POSITION)[levels(main_data$POSITION) %in% 
                             c('C','CEO','COO','VP','SRMGMT','SGM','PRSD','MD','DIR','DIR1','DGM','AVP','AGM')] <- "Executive_Level"

levels(main_data$POSITION)[levels(main_data$POSITION) %in% 
                             c('AST','ASTM','DM','JRMGMT','MGR','SM','TL')] <- "Middle_Level"

levels(main_data$POSITION)[levels(main_data$POSITION) %in% 
                             c('CP1','ED1','ED4','HEADCLER','JROF','OFF','OFFCADRE','POSITION','PROP','SROF','TRST1')] <- "Entry_Level"

main_data$POSITION <- as.character(main_data$POSITION)

# We will change the NA's to missing
main_data$POSITION[is.na(main_data$POSITION)] <- "Missing"

#------------PRE_JOBYEARS Variable

# Checking the summary
summary(factor(main_data$PRE_JOBYEARS))

# We will change the NA's to blank
main_data$PRE_JOBYEARS[is.na(main_data$PRE_JOBYEARS)] <- ""
# The bucketing part will be taken care with WOE values.

# Changing to numeric
main_data$PRE_JOBYEARS <- as.numeric(main_data$PRE_JOBYEARS)


#------------CUST_CONSTTYPE_ID Variable

# Checking the summary
summary(factor(main_data$CUST_CONSTTYPE_ID))

# We will change the NA's to missing
main_data$CUST_CONSTTYPE_ID[is.na(main_data$CUST_CONSTTYPE_ID)] <- ""

# Changing to numeric
main_data$CUST_CONSTTYPE_ID <- as.numeric(main_data$CUST_CONSTTYPE_ID)

#------------CUST_CATEGORYID Variable

# Checking the summary
summary(factor(main_data$CUST_CATEGORYID))

# We will change the NA's to missing
main_data$CUST_CATEGORYID[is.na(main_data$CUST_CATEGORYID)] <- ""

# Changing to numeric
main_data$CUST_CATEGORYID <- as.numeric(main_data$CUST_CATEGORYID)

#------------Yearof_lastreceipt Variable

# Checking the summary
summary(factor(main_data$Yearof_lastreceipt))

# We will change the NA's to missing
main_data$Yearof_lastreceipt[is.na(main_data$Yearof_lastreceipt)] <- ""

# Changing to numeric
main_data$Yearof_lastreceipt <- as.numeric(main_data$Yearof_lastreceipt)

#---------------------------------------------
# Information Value and WOE
#---------------------------------------------
# First we will take out the information value
# Second we will get the weight of evidence attached to each variable
# Third We will segregate the categorical variable from others
# Forth we will bucket the categorical variables accordingly and replace them with woe values
# Fifth we will scale the numerical variables
# Add back the dataset as one dataset and then run the GLM

# Its time to split the data now
train_main <- subset(main_data, from_set == "Train")
train_main <- train_main[, c(1:42,45,43)]
test_main <- subset(main_data, from_set == "Test")
test_main <- test_main[, c(1:42,45,43)]

train_main$City_ClusterID <- as.character(train_main$City_ClusterID)
test_main$City_ClusterID <- as.character(test_main$City_ClusterID)

# Information value is a useful technique to select important variables in a predictive model.
# It helps to rank variables on the basis of their importance.
IV <- create_infotables(data = train_main[, -1], y = "FORECLOSURE", ncore = 2)

# Creating a dataframe containing IV values of all the variables
IV_dataframe <- IV$Summary
str(IV_dataframe)

# checking the WOE values for bucketing
woe_list <- list()

typeof(IV$Tables)
length(IV$Tables)

# Extracting only the bins and the woe values of each bin
for(i in 1:length(IV$Tables)) {
  woe_list[[i]] = cbind(IV$Tables[[i]][1],IV$Tables[[i]][4])
}

woe_list


# Re-combining the datasets so that we can bucket values in certain variables according to WOE values
train_main$Status <- "Train"
test_main$Status <- "Test"
recom_set <- rbind(train_main,test_main)


# Now lets group the character variables looking their WOE values
# City Cluster
recom_set$City_ClusterID[which(recom_set$City_ClusterID >= 1  & recom_set$City_ClusterID <= 3)] <- '1_3'
recom_set$City_ClusterID[which(recom_set$City_ClusterID == "4")] <- '4_6'
recom_set$City_ClusterID[which(recom_set$City_ClusterID == "5")] <- '4_6'
recom_set$City_ClusterID[which(recom_set$City_ClusterID == "6")] <- '4_6'

#Age
recom_set$AGE[which(recom_set$AGE >= 18 & recom_set$AGE <= 28)] <- '18_28'
recom_set$AGE[which(recom_set$AGE >= 29 & recom_set$AGE <= 31)] <- '29_31'
recom_set$AGE[which(recom_set$AGE >= 32 & recom_set$AGE <= 34)] <- '32_34'
recom_set$AGE[which(recom_set$AGE >= 35 & recom_set$AGE <= 37)] <- '35_37'
recom_set$AGE[which(recom_set$AGE >= 38 & recom_set$AGE <= 39)] <- '38_39'
recom_set$AGE[which(recom_set$AGE >= 40 & recom_set$AGE <= 42)] <- '40_42'
recom_set$AGE[which(recom_set$AGE >= 43 & recom_set$AGE <= 45)] <- '43_45'
recom_set$AGE[which(recom_set$AGE >= 46 & recom_set$AGE <= 48)] <- '46_48'
recom_set$AGE[which(recom_set$AGE >= 49 & recom_set$AGE <= 53)] <- '49_53'
recom_set$AGE[which(recom_set$AGE >= 54 & recom_set$AGE <= 76)] <- '54_76'
recom_set$AGE[is.na(recom_set$AGE)] <- "Missing"

#Pre_jobyear ---Note run these four lines first
recom_set$PRE_JOBYEARS[which(recom_set$PRE_JOBYEARS >= 0 & recom_set$PRE_JOBYEARS <= 1)] <- '0_1'
recom_set$PRE_JOBYEARS[which(recom_set$PRE_JOBYEARS >= 2 & recom_set$PRE_JOBYEARS <= 3)] <- '2_3'
recom_set$PRE_JOBYEARS[which(recom_set$PRE_JOBYEARS >= 5 & recom_set$PRE_JOBYEARS <= 8)] <- '5_8'
recom_set$PRE_JOBYEARS[which(recom_set$PRE_JOBYEARS >= 13 & recom_set$PRE_JOBYEARS <= 42)] <- '13_42'

recom_set$PRE_JOBYEARS[which(recom_set$PRE_JOBYEARS == "9")] <- '9_12'
recom_set$PRE_JOBYEARS[which(recom_set$PRE_JOBYEARS == "10")] <- '9_12'
recom_set$PRE_JOBYEARS[which(recom_set$PRE_JOBYEARS == "11")] <- '9_12'
recom_set$PRE_JOBYEARS[which(recom_set$PRE_JOBYEARS == "12")] <- '9_12'
recom_set$PRE_JOBYEARS[is.na(recom_set$PRE_JOBYEARS)] <- "Missing"

# No of dependant
recom_set$NO_OF_DEPENDENT[which(recom_set$NO_OF_DEPENDENT >= 2 & recom_set$NO_OF_DEPENDENT <= 10)] <- '2_10'
recom_set$NO_OF_DEPENDENT[is.na(recom_set$NO_OF_DEPENDENT)] <- "Missing"

# CUST_CATEGORYID
recom_set$CUST_CATEGORYID[which(recom_set$CUST_CATEGORYID >= 5 & recom_set$CUST_CATEGORYID <= 8)] <- '5_8'
recom_set$CUST_CATEGORYID[is.na(recom_set$CUST_CATEGORYID)] <- "Missing"

# CUST_CONSTTYPE_ID
recom_set$CUST_CONSTTYPE_ID[which(recom_set$CUST_CONSTTYPE_ID >= 2 & recom_set$CUST_CONSTTYPE_ID <= 6)] <- '2_6'
recom_set$CUST_CONSTTYPE_ID[is.na(recom_set$CUST_CONSTTYPE_ID)] <- "Missing"

# Yearof_lastreceipt
recom_set$Yearof_lastreceipt[which(recom_set$Yearof_lastreceipt >= 1974 & recom_set$Yearof_lastreceipt <= 2016)] <- '1974_2016'
recom_set$Yearof_lastreceipt[which(recom_set$Yearof_lastreceipt == "2018")] <- '2018_2019'
recom_set$Yearof_lastreceipt[which(recom_set$Yearof_lastreceipt == "2019")] <- '2018_2019'
recom_set$Yearof_lastreceipt[is.na(recom_set$Yearof_lastreceipt)] <- "Missing"

# Checking for NA values now.
sapply(recom_set, function(x) sum(is.na(x)))

# we will remove Gross_Income and NETTAKEHOMEINCOME as the number of NA are very
# We will not like to replace the NA's with mean, not a good idea
# and anyways the information value that they have is bothering, so to be removed.
recom_set <- recom_set[, c(1:39,41,43:45)]

# Now we will replace the NA's in LAST_RECEIPT_AMOUNT and DPD with variable mean
recom_set$LAST_RECEIPT_AMOUNT <- ifelse(is.na(recom_set$LAST_RECEIPT_AMOUNT), mean(recom_set$LAST_RECEIPT_AMOUNT, na.rm=TRUE), recom_set$LAST_RECEIPT_AMOUNT)
recom_set$DPD <- ifelse(is.na(recom_set$DPD), mean(recom_set$DPD, na.rm=TRUE), recom_set$DPD)

# First lets rearrange the dataset
recom_set <- recom_set[, c(1:28,30,29,31:41,42:43)]

# Storing the categorical variables in a list
cols <- c("PRODUCT","Yearof_lastreceipt","CUST_CONSTTYPE_ID","CUST_CATEGORYID","AGE","SEX",
          "MARITAL_STATUS","QUALIFICATION","NO_OF_DEPENDENT","POSITION","PRE_JOBYEARS","City_ClusterID")

# Now lets convert all of them to factor
recom_set[cols] <- lapply(recom_set[cols], factor)

# We will now do one hot encoding for the categorical variables
recom_set <- dummy.data.frame(recom_set, names = c("PRODUCT","Yearof_lastreceipt","CUST_CONSTTYPE_ID",
                                                   "CUST_CATEGORYID","AGE","SEX",
                                                   "MARITAL_STATUS","QUALIFICATION","NO_OF_DEPENDENT",
                                                   "POSITION","PRE_JOBYEARS","City_ClusterID"))

# Now we are felt with scaling, lets scale the numerical variables.
recom_set_num <- recom_set[, c(2:29)]
recom_set_num <- as.data.frame(scale(recom_set_num))

# Keeping categorical variables, dependant and from set
all_others <- recom_set[, c(30:90)]

# Now we will column bind all the variables to make the final dataset
final_recom_set <- cbind(recom_set[,1],recom_set_num,all_others)
names(final_recom_set)[1] <- "AGREEMENTID"

# Now lets divide the dataset into train and test datasets.
final_train <- subset(final_recom_set, Status == "Train") 
final_train <- final_train[, -90]
final_test <- subset(final_recom_set, Status == "Test")
final_test <- final_test[, c(1:88)]

#------------------------------------------
# First we will run random forest and take out the important variables from the
# random forest model, we will then use these variables in building a logistic regression
#------------------------------------------
# Lets try with random forest
library(randomForest)

# Setting up the seed
set.seed(123)

final_train$FORECLOSURE <- as.factor(final_train$FORECLOSURE)

# Running a basic RF model
model_rf <- randomForest(FORECLOSURE ~., ntree = 1000, data = final_train[,-1])

print(model_rf)

varImpPlot(model_rf, sort = T, n.var=30, main="Top 20 - Variable Importance")

#--------------------------------------------
# We have the top 20 important variables from random forest algorithm, now lets
# build a logistic regression.

# Approaching with 20 variables.

# Lets take the top 20 variables and form the dataset for GLM
for_glm_train <- final_train[, c(1,4:7,9:10,12:13,20:21,23,25:26,29:32,34:36,89)]

# without balancing the dataset and 20 variables
best_model <- glm(FORECLOSURE ~ ., data = for_glm_train[,-1], family = "binomial")
summary(best_model)

# Smote Set, for balancing with 20 variables taken
train_smote <- SMOTE(FORECLOSURE ~ ., data = for_glm_train, perc.over = 100, perc.under=200)

best_model_smote <- glm(FORECLOSURE ~ ., data = train_smote[,-1], family = "binomial")
summary(best_model)


for_glm_test <- final_test[, c(1,4:7,9:10,12:13,20:21,23,25:26,29:32,34:36)]

# For imbalanced data
test_pred <- predict(best_model, for_glm_test[,-1], type = "response")

# For balanced data through smote
test_pred_smote <- predict(best_model_smote, for_glm_test[,-1], type = "response")

final_output <- as.data.frame(cbind(for_glm_test[,1], test_pred))
final_output_smote <- as.data.frame(cbind(for_glm_test[,1], test_pred_smote))

write.csv(final_output,"test_prediction.csv",row.names = FALSE)
write.csv(final_output_smote,"test_prediction_smote.csv",row.names = FALSE)

#--------------------------------------------
# Lets approach this by taking 28 significant variables from RF to build GLM
top_28_train <- final_train[, c(1:7,9:10,12:13,18:26,28:36,89)]

top_28_smote <- SMOTE(FORECLOSURE ~ ., data = top_28_train, perc.over = 100, perc.under=200)

model_1 <- glm(FORECLOSURE ~ ., data = top_28_smote[,-1], family = "binomial")
summary(model_1)

model_2 <- stepAIC(model_1, direction = "both")
summary(model_2)

top_28_test <- final_test[, c(1:7,9:10,12:13,18:26,28:36)]

test_pred_top_28 <- predict(model_2, top_28_test[,-1], type = "response")

final_output_top_28 <- as.data.frame(cbind(top_28_test[,1], test_pred_top_28))

write.csv(final_output_top_28,"test_prediction_top_28.csv",row.names = FALSE)































































































