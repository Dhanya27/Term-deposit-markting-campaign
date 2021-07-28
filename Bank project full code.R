
# Prediction of Term deposit Subscription in Marketing Campaign by different models
# Objective is to predict term deposit subscription by different models and comparing accuracy
# across different models, thereby choosing the best model with highest accuracy.

# Installing required packages for the project
if(!require(dplyr)) install.packages("dplyr")
if(!require(tidyverse)) install.packages("tidyverse")
if(!require(caret)) install.packages("caret")
if(!require(data.table)) install.packages("data.table")
if(!require(rminer)) install.packages("rminer")

# Loading required libraries and data
library(dplyr)
library(tidyverse)
library(caret)
library(data.table)
library(rminer)

# Downloading data from uci repository dataset
# https://archive.ics.uci.edu/ml/machine-learning-databases/00222/
# Downloaded from https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip
dl <- tempfile()
download.file("https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip",dl)
# Unzip bank-additional-full.csv from bank-additional zip file
data <- read.csv2(unzip(dl,"bank-additional/bank-additional-full.csv"),header = TRUE,
                  stringsAsFactors = TRUE)


# Number of observations in the dataset.
dim(data)
# There are 41188 rows and 21 columns in the dataset.

# Overview of the dataset with initial 6 rows
head(data)
# It consists of 20 descriptive attributes(client and campaign information) and 1 target attribute(y)
# Output variable y has two values yes or no, which denotes whether client has subscribed term deposit or not?

# Structure of the dataset
str(data)

# Summary of the dataset
summary(data)

# Data Cleaning
# Scanning for NAs
colSums(is.na(data))
# It is proven that all attributes in the dataset doesn't have NA values. So, there is no problem 
# in future to transform NAs into mean or median of the attribute to get a real predictive model.

# Splitting the dataset into training(bank_data) and testing(validation) sets
# The validation set will be 10% of bank marketing data
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = data$y, times = 1, p = 0.1, list = FALSE)
bank_data <- data[-test_index,]
bank_temp <- data[test_index,]
# Make sure euribor3m in validation set are also in bank_data set
validation <- bank_temp %>%
  semi_join(bank_data, by = "euribor3m")
# Bring back removed rows from validation to bank_data set 
removed <- anti_join(bank_temp, validation)
bank_data <- rbind(bank_data, removed)
rm(dl, test_index, bank_temp, removed) #Clear out unwanted files
# Validation set should only be used at the end of the final model.

##  Data Exploration and Visualization
# General overview of the dataset:

# Number of observations(rows and columns) in the dataset
dim(bank_data)
dim(validation)
# There are 37071 rows and 21 columns in the training dataset
# There are 4117 observations(about 10% of training set) in the testing dataset

# Glimpse of the dataset
glimpse(bank_data)

# Summary of the dataset
summary(bank_data)

# Has the client subscribed a deposit?(Target variable)
unique(bank_data$y) # This is the output variable. It has two categories: yes and no
table(bank_data$y)
# Table lists how many clients subscribed(yes) term deposit or not(no).
# Distribution of y:
term_y <- bank_data %>%
  group_by(y) %>% summarize(count = n()) %>% arrange(desc(count))
term_y
term_y %>% 
  ggplot(aes(x = y, y = count)) + 
  geom_bar(stat="identity") +
  ggtitle("Distribution of term deposit subscription")
# clients who does not subscribe term deposit are higher in count compared to clients
# who subscribed term deposit in this campaign.

# Exploration of age attribute:
unique(bank_data$age)
length(unique(bank_data$age))
min(bank_data$age)
max(bank_data$age)
table(bank_data$age)
# In the marketing campaign, minimum age person contacted is 17 whereas maximum age is 98
# There are 78 different ages of person who are client to banks.
# The table shows how many times the same type of age person is contacted during the campaign

# What are the different type of jobs of client?
unique(bank_data$job)
n_distinct(bank_data$job)
table(bank_data$job)
# There are 12 different types of job for client are available in the dataset.
# Table lists how many clients are in the same job type in the campaign.

# What are the distinct marital status of client?
unique(bank_data$marital)
n_distinct(bank_data$marital)
# There are 4 categories in the marital status of client.
# Married, single, divorced and unknown
sum(bank_data$marital=="unknown")
# There are 72 unknown entries of clients for marital status.
table(bank_data$marital)

# Basic Educational details of client:
unique(bank_data$education)
n_distinct(bank_data$education)
# There are 8 categories of Education status namely, basic.4y, high.school, basic.6y, basic.9y           
# professional.course, unknown, university.degree and illiterate
sum(bank_data$education=="unknown")
# There are 1541 unknown entries of client education details.
table(bank_data$education)
# This table lists number of clients who are in the same education category.

# What are the different ways bankers make contact with client for the campaign?
unique(bank_data$contact)
# By two ways - telephone and cellular
table(bank_data$contact)
# Table shows number of clients who have been contacted by cellular and telephone

# What are the different results of the last contact marketing campaign?
unique(bank_data$poutcome)
# It has three results - success, failure and nonexistent(no previous contact)
sum(bank_data$poutcome=="nonexistent")
# There are 32032 nonexistent entries of client's previous marketing campaign results.
table(bank_data$poutcome)

# Feature selection
# The attributes which affects prediction of deposit subscription is chosen in this section 
# for building a better model to predict whether client subscribes deposit or not.

# Does age play a role in predicting output variable, y?
bank_data %>% group_by(age) %>% summarize(count = n()) %>% arrange(desc(count))
# Histogram of the Clients Age Distribution
bank_data %>% 
  ggplot(aes(age,fill = y)) +
  geom_histogram(bins = 30, color = "black") +
  ggtitle("Histogram of the Clients Age Distribution")
# Plot shows clients around the age 30 to 40 are higher in number
# This implies banking systems choose more people around 30 to 40 for marketing campaigns
# compared to others for coercing clients for term deposit subscription 
# And one can see decrease in number after age 60 drastically
# This means clients age>60 does not subscribe term deposit much in number.
ggplot(bank_data, aes(age, fill =y)) + geom_histogram(aes(y=..count..)) +
  facet_grid(~y) +
  ggtitle("Histogram of the Clients Age Subscription")
# The age group 30 to 40 subscribes term deposit(yes) most, although all age group people subscribes.
# Similarly, the same age group(30 - 40) have higher count in who did not subscribe term deposit(no)
# This leads to the fact that this is the most sought after group due to its highest proportion in total.


# Distribution of Job types Of Clients
jobs <- bank_data %>% select(job,y) %>% 
  group_by(job,y) %>%
  summarize(count = n()) %>% arrange(desc(count))
jobs
# Highest type of jobs found among clients are admin.(y = "no")(8153) and blue-collar(y="no")(7757)
jobs %>%
  ggplot(aes(x = reorder(job,count), y = count, fill = y)) +
  geom_bar(stat="identity", position ="dodge") +
  xlab("job") +
  theme(axis.text.x=element_text(angle=45,hjust=1)) +
  ggtitle("Distribution of Client's job types")
# From the plot, it is clear unknown and student are lower in number compared to others.
# People with admin. job who does not subscribe are larger in number (8153) in the dataset
# This implies admin.job people are more active in deposit subscription in both "yes"(1221) & "no"(8153)
# People in admin subscribes term deposit more followed by technician(660) and blue-collar(565).
# From the plot, it is obvious due to higher proportion in total

# To confirm if job attribute contributes to term deposit subscription, lets run chi-squared test and see,
c_job <- table(bank_data$job,bank_data$y)
c_job
chisq_job <- chisq.test(c_job)
chisq_job 
# Since p-value is less than 0.05, it is clear that age is an important factor in predicting term deposit subscription.
# p-value < 2.2e-16 denotes results are considered statistically significant
# i.e, attribute job is significant(dependent) to attribute y(term deposit subscription)
# thus job feature should be included in predicting the model.


# Distribution of Marital Status of clients
mar <- bank_data %>%
  select(marital,y) %>% 
  group_by(marital,y) %>%
  summarize(count = n()) %>%
  arrange(desc(count))
mar  
# Unknown clients with subscription "yes" are lower in number(12) followed by unknown(y="no")
mar %>% ggplot(aes(reorder(marital,count),count)) +
  geom_bar(aes(fill = y),stat = "identity") + 
  xlab("Marital status") +
  ggtitle("Barplot of Marital status of clients")
# Married people avail the highest number of term deposits followed by single people.
# Plot shows Married people who does not subscribe term deposits are also higher in number(20157) 
# followed by single people(y = "no") of 8922 respectively.
# This result is also due to higher proportion of people in that category

# To check if marital attribute contributes to term deposit subscription, lets run chi-squared test,
c_mar <- table(bank_data$marital,bank_data$y)
c_mar
chisq_mar <- chisq.test(c_mar)
chisq_mar 
# It is clear, p-value < 0.05(i.e, p < 2.2e-16) implies that marital is statistically significant(dependent)
# to attribute y(term deposit subscription)
# marital details of client is an important feature in term deposit subscription.


# Distribution of Education details of clients
edu <- bank_data %>%
  select(education,y) %>%
  group_by(education,y) %>%
  summarize(count = n()) %>%
  arrange(desc(count))
edu
# Illiterate people have lower term deposit subscription compared to others.
edu %>%
  ggplot(aes(x = reorder(education,count), y = count, fill = y)) +
  geom_bar(stat="identity", position ="dodge") +
  xlab("Education") +
  theme(axis.text.x=element_text(angle=45,hjust=1)) +
  ggtitle("Distribution of Education of the client")
# University degree clients subscribe term deposit higher (1510) than others(High school-936
# Professional course - 535)
# Plot shows Clients who have University Degree who does not subscribe are also higher in number.

# To check if education of the client relates to term deposit subscription, run chi-squared test,
c_edu <- table(bank_data$education,bank_data$y)
c_edu
chisq_edu <- chisq.test(c_edu,simulate.p.value = TRUE)
chisq_edu 
# One can see p-value is less than 0.05 above for education chi-squared test with y.
# It is clear that education feature plays a role in subscribing term deposit subscription
# because education feature is dependent to deposit subscription in the marketing campaign.


# Does client have a credit default? (fails to pay debt)
unique(bank_data$default)
table(bank_data$default)
# It have three status as yes,no and unknown
# yes - client who has credit default , no - who does not have credit default
# unknown - no information about default for that client
# Table shows how many clients are in the same category of default status in the marketing campaign.
sum(bank_data$default == "unknown")
# We don't know credit default status for 7757 clients in the campaign.
# How does credit default affects deposit subscription?
def <- bank_data %>%
  group_by(default,y) %>%
  summarize(count = n()) %>%
  arrange(desc(count))
def
# The clients who have credit default are lower in number.
def %>%
  ggplot(aes(x = reorder(default,count), y = count, fill = y)) +
  geom_bar(stat="identity", position ="dodge") +
  xlab("Credit Default Status") +
  facet_grid(~y) +
  ggtitle("Barplot of distribution of Credit Default of clients towards subscription")
# From the plot, it is clear, client with credit default does not subscribe term deposit much.
# They are lesser (3) in count compared to others.
# Clients who does not have credit default avails term deposit higher compared to others
# This may be due to high ratio in the total count as well.

# Chi-squared test is run below to check whether default is an important variable in deciding 
# term deposit subscription.
c_def <- table(bank_data$default,bank_data$y)
c_def
chisq_def <- chisq.test(c_def,simulate.p.value = TRUE)
chisq_def
# Chi-squared test's statistical significant result shows default is an important feature in 
# deciding term deposit subscription(y) preference
# Because p-value is 0.0004998 which is less than 0.05.
# So, default status of client is dependent to output variable,y.


# Does client take any housing loans in this or any banks?
unique(bank_data$housing)
# It have 3 status - "yes" for taking house loan, "no" for no house loan
# and "unknown" for no known information about housing loan for that client
sum(bank_data$housing == "unknown")
# There are 909 unknown entries for housing loan details
# How does housing loan affects term deposit subscription?
house <- bank_data %>%
  group_by(housing,y) %>%
  summarize(count = n()) %>%
  arrange(desc(count))
house
# Clients with unknown information of housing loans are the least in number(809 for y="no" and 100 for y="yes")
house %>% 
  ggplot(aes(x = reorder(housing,count), y = count, fill = y)) +
  geom_bar(stat="identity", position ="dodge") +
  xlab("Housing Loan Status") +
  facet_grid(~y) +
  ggtitle("Barplot of distribution of Housing loan status of clients")
# It appears people who take housing loans subscribes term deposit more compared to others
# followed by people who don't take hosing loans 
# Plot shows clients with housing loans who does not subscribe term deposits are also higher in number
# This is due to the fact that housing loan has large proportion in count.

# Chi-squared test is run below to check whether housing loan is an important variable in deciding 
# term deposit subscription.
c_house <- table(bank_data$housing,bank_data$y)
c_house
chisq_house <- chisq.test(c_house)
chisq_house
# chi-squared test's statistical result shows housing loan feature is not significant to y
# That is, housing loan status is not an important variable in deciding term deposit subscription
# preference because p-value is greater than 0.05(p-value is 0.05763).
# Lets analyze this feature in the model results and check whether chi-squared test result is true. 


# Does client take any personal loans?
unique(bank_data$loan)
table(bank_data$loan)
# It shows three status as yes, no and unknown
# "yes" for taking personal loan, "no" for no personal loan
# and "unknown" for no known information about personal loan for that client
# Table lists how many clients belongs to the same criteria of personal loan details in the marketing campaign.
per_loan <- bank_data %>%
  group_by(loan,y) %>%
  summarize(count = n()) %>%
  arrange(desc(count))
per_loan
# Clients with unknown details about personal loan are the ones who subscribe term deposit in low number
per_loan %>% 
  ggplot(aes(x = reorder(loan,count), y = count, fill = y)) +
  geom_bar(stat="identity", position ="dodge") +
  xlab("Personal Loan Status") +
  facet_grid(~y) +
  ggtitle("Barplot of distribution of Personal loan status of clients")
# From the plot, it appears that person who does not take personal loans avails term deposit in large 
# number followed by clients who take personal loans 
# Due to huge proportion in total count, clients with personal loan who does not subscribe term deposit 
# are also higher in number

# Chi-squared test is run below:
c_loan <- table(bank_data$loan,bank_data$y)
c_loan
chisq_loan <- chisq.test(c_loan)
chisq_loan 
# Since p-value is 0.6555(greater than 0.05), personal loan feature does not decide subscription 
# preference of clients. Lets analyze this in our model section.
# Thus personal loan does not contribute to term deposit subscription.


# Distribution of contact status with client
con <- bank_data %>%
  select(contact,y) %>%
  group_by(contact,y) %>%
  summarize(count = n()) %>%
  arrange(desc(count))
con
# Clients contacted by telephone and subscribes term deposit are the least in the marketing campaign
con %>% 
  ggplot(aes(x = reorder(contact,count), y = count, fill = y)) +
  geom_bar(stat="identity", position ="dodge") +
  xlab("Contact with clients") +
  facet_grid(~y) +
  ggtitle("Barplot of distribution of Contact details with clients for campaign")
# Clients who have been contacted by cellular subscribes term deposit higher in number.
# This is same with cellular contact who does not subscribe term deposit due to its large 
# proportion in number

# To check if contact attribute contributes to term deposit subscription, lets run chi-squared test and see,
c_contact <- table(bank_data$contact,bank_data$y)
c_contact
chisq_contact <- chisq.test(c_contact)
chisq_contact 
# From the chi-squared test, it is clear that contact variable is an important factor in deciding
# term deposit subscription as p-value < 0.05
# Contact attribute is statistically significant to term deposit subscription
# So,we should consider this feature while predicting the model.


# Which month client subscribes term deposit much?
unique(bank_data$month)
table(bank_data$month)
# There are 10 different months clients have been contacted by bankers throughout the campaign 
# ranging from month March to December. 
# Table shows how many clients has been contacted under same month in the campaign.
mon <- bank_data %>%
  group_by(month,y) %>%
  summarize(count = n()) %>%
  arrange(desc(count))
mon
# December month is the one client subscribes term deposit in low number
mon %>%
  ggplot(aes(x = reorder(month,count), y = count, fill = y)) +
  geom_bar(stat="identity", position ="dodge") +
  xlab("Month") +
  theme(axis.text.x=element_text(angle=45,hjust=1)) +
  ggtitle("Barplot of distribution of month in the campaign")
# Plot shows in month may, clients subscribe term deposit more compared to other months in the campaign
# followed by august and july month
# This figure is due to high proportion of count in may month
# From this plot, we get to know which month client is subscribing more and which month not
# This helps us to target customers during the marketing campaign.
# We can increase the number of calls and clients in month may and august

# Chi-squared test is run below,
c_mon <- table(bank_data$month,bank_data$y)
c_mon
chisq_mon <- chisq.test(c_mon)
chisq_mon 
# From the chi-squared test, it is clear that month variable is an important factor in deciding
# term deposit subscription as p-value < 0.05
# Month attribute is statistically significant to term deposit subscription


# Which day of week client subscribes term deposit much?
table(bank_data$day_of_week)
# Table shows how many clients have been contacted in the same day in the campaign.
day <- bank_data %>%
  group_by(day_of_week,y) %>%
  summarize(count = n()) %>%
  arrange(desc(count))
day
# Friday is the day client subscribes term deposit in lesser number
day %>% ggplot(aes(reorder(day_of_week,count),count)) +
  geom_bar(aes(fill = y),stat = "identity") + 
  xlab("Day of week") +
  ggtitle("Barplot of distribution of Day of week in the campaign")
# Plot shows Thursday is the day of week clients subscribe term deposit more compared to other days.
# followed by monday and wednesday
# There is not much difference in the days where client subscribes term deposit 
# The total numbers for day of week are closer to each other

# Chi-squared test is run below to check whether day of week is an important variable in deciding 
# term deposit subscription.
c_day <- table(bank_data$day_of_week,bank_data$y)
c_day
chisq_day <- chisq.test(c_day)
chisq_day 
# Chi-squared test tells p-value as 0.000288 which is less than 0.05
# So, day of week variable affects term deposit subscription preference


# Distribution of duration of last contact
length(unique(bank_data$duration))
min(bank_data$duration)
max(bank_data$duration)
table(bank_data$duration)
# There are 1513 unique duration of phone calls during the campaign
# Minimum duration of phone call is 0 whereas maximum duration is 4918
# If duration is 0, then there is no term deposit subscription as clients does not attend the call 
# Table shows how many clients have the same duration as other clients
dur <- bank_data %>%
  group_by(duration,y) %>%
  summarize(count = n()) %>%
  arrange(desc(count))
dur
dur %>% 
  ggplot(aes(duration,count)) + 
  geom_point(aes(color = y)) + 
  ggtitle("Distribution of duration of last contact")
# Plot shows duration of over 200 to 750 subscribes term deposit more compared to others
# It is also clear longer duration of phone call guarantees term deposit subscription with some exceptions
# Almost all people who does not subscribe term deposit decide within first 4 minutes
# and people who wish to subscribe sometimes take little longer in getting convinced and deciding.

# Chi-squared test is run below,
c_dur <- table(bank_data$duration,bank_data$y)
c_dur
chisq_dur <- chisq.test(c_dur,simulate.p.value = TRUE)
chisq_dur 
# From the chi-squared test, it is clear that duration variable is an important factor in deciding
# term deposit subscription as p-value < 0.05
# Duration attribute is statistically significant to term deposit subscription


# Distribution of outcome of the previous campaign
out <- bank_data %>%
  group_by(poutcome,y) %>%
  summarize(count = n()) %>%
  arrange(desc(count))
out
out %>% 
  ggplot(aes(x = reorder(poutcome,count), y = count, fill = y)) +
  geom_bar(stat="identity", position ="dodge") +
  xlab("Outcome of previous campaign") +
  facet_grid(~y) +
  ggtitle("Barplot of distribution of outcome of previous campaign")
# From the plot, it is clear success outcome of the previous campaign is the least in number compared to others 
# The nonexistent outcome of the previous campaign subscribes term deposit more followed by failure
# Failure outcome of previous campaign clients who does not subscribe deposit can be 
# targeted next year to coerce them into subscription.

# Chi-squared test is run below,
c_out <- table(bank_data$poutcome,bank_data$y)
c_out
chisq_out <- chisq.test(c_out)
chisq_out 
# One can see p-value is less than 0.05 above for previous outcome of campaign with y.
# It is clear that previous outcome feature plays a role in subscribing term deposit subscription.


# Exploration of number of contacts performed during this campaign
unique(bank_data$campaign)
length(unique(bank_data$campaign))
min(bank_data$campaign)
max(bank_data$campaign)
table(bank_data$campaign)
# 42 different number of contacts performed during the campaign
# Minimum number of contact performed for the client during the campaign is 1
# whereas maximum number of contact for client is 56
# table shows how many clients are in the same number of contacts category
# Distribution of contacts during this campaign
camp <- bank_data %>%
  group_by(campaign,y) %>% summarize(count = n()) %>% arrange(desc(count))
camp
# Barplot of the number of contacts for campaign Distribution
camp %>% 
  ggplot(aes(reorder(campaign,count),count)) +
  geom_bar(aes(fill = y),stat = "identity") + 
  xlab("Campaign") +
  ggtitle("Barplot of distribution of number of contacts during this campaign")
# From the plot, it is clear that one contact with the client subscribes term deposit more
# followed by 2 and 3. This is due to large proportion of number for one contact during the campaign
# This plays an important role in deciding term deposit subscription
# Chi-squared test is not applicable for numeric categorical factors.


# Exploration of number of days after last contact from previous campaign
unique(bank_data$pdays)
length(unique(bank_data$pdays))
min(bank_data$pdays)
max(bank_data$pdays)
table(bank_data$pdays)
# There are 26 different number of days after last contact from previous campaign
# Minimum number of days after last contact is 0 whereas maximum number of days are 999
# 999 means client was not previously contacted
# Distribution of number of days after last contact
num_of_days <- bank_data %>%
  group_by(pdays,y) %>% summarize(count = n()) %>% arrange(desc(count))
num_of_days
# Barplot of the number of days after last contact Distribution
num_of_days %>% 
  ggplot(aes(reorder(pdays,count),count)) +
  geom_bar(aes(fill = y),stat = "identity") +
  xlab("Number of days after last contact") +
  ggtitle("Barplot of the number of days after last contact Distribution")
# From the plot,it is clear that client who was not contacted(999) during previous campaign
# subscribes term deposit more as they did not subscribe last year
# This is also due to the fact that 999 has large proportion of number in the campaign
# pdays act as an important factor in term deposit subscription as many new clients who have 
# not contacted during last campaign  subscribes term deposit during this campaign


# Exploration of number of contacts for previous campaign for this client
unique(bank_data$previous)
length(unique(bank_data$previous))
min(bank_data$previous)
max(bank_data$previous)
table(bank_data$previous)
# There are eight different number of contacts for previous campaign for the client
# Minimum number of contact for the client during previous campaign is 0
# whereas maximum number of contact for client is 7
# Distribution of number of contacts for previous campaign for this client
prev <- bank_data %>%
  group_by(previous,y) %>% summarize(count = n()) %>% arrange(desc(count))
prev
# Barplot of the number of contacts before this campaign Distribution
prev %>% 
  ggplot(aes(reorder(previous,count),count)) +
  geom_bar(aes(fill = y),stat = "identity") +
  xlab("Number of contacts during previous campaign") +
  ggtitle("Baarplot of the number of contacts for previous campaign Distribution")
# From the plot, it is clear that clients who have zero contact during previous campaign
# subscribes term deposit more followed by one,two etc.,
# Obviously, clients who have not contacted last campaign will be contacted this campaign to 
# coerce them into term deposit subscription as well as due to high proportion of number
# Also clients who have been contacted last year but did not subscribe last campaign subscribes
# term deposit this campaign with frequent marketing strategies and awareness
# So, previous is an important term for term deposit subscription


# Distribution of Employment variation rate
unique(bank_data$emp.var.rate)
length(unique(bank_data$emp.var.rate))
table(bank_data$emp.var.rate)
# There are 10 different types of employment variation rate for clients in the campaign
# Table lists how many clients have same employment variation rate in the campaign
emp <- bank_data %>%
  group_by(emp.var.rate,y) %>% summarize(count = n()) %>% arrange(desc(count))
emp
# Barplot of the Employment variation rate Distribution
emp %>% 
  ggplot(aes(reorder(emp.var.rate,count),count)) +
  geom_bar(aes(fill = y),stat = "identity") +
  xlab("Employment variation rate") +
  ggtitle("Histogram of the Employment variation rate Distribution")
# Plot shows employment variation rate of 1.4 subscribes term deposit much followed by -1.8 and 1.1
# This figure may be due to large proportion of number 
# Employment variation rate of -0.2 subscribes term deposit in low number


# Distribution of Consumer price index
unique(bank_data$cons.price.idx)
length(unique(bank_data$cons.price.idx))
table(bank_data$cons.price.idx)
# There are 26 different types of consumer price index for clients in the campaign
# Table lists how many clients have same consumer price index in the campaign
cons_price <- bank_data %>%
  group_by(cons.price.idx,y) %>% summarize(count = n()) %>% arrange(desc(count))
cons_price
# Barplot of the Consumer price index Distribution
cons_price %>% 
  ggplot(aes(reorder(cons.price.idx,count),count)) +
  geom_bar(aes(fill = y),stat = "identity") +
  xlab("Consumer price index") +
  ggtitle("Barplot of the Consumer price index Distribution")
# Plot shows consumer price index of 92.893 subscribes term deposit much followed by 93.095.
# Consumer price index of 92.756 subscribes term deposit in low number
# From the figure one can observe changes in consumer price index increases term deposit subscription


# Distribution of Consumer Confidence index
unique(bank_data$cons.conf.idx)
length(unique(bank_data$cons.conf.idx))
table(bank_data$cons.price.idx)
# There are 26 different types of consumer confidence index for clients in the campaign
# Table lists how many clients have same consumer confidence index in the campaign
cons_conf <- bank_data %>% 
  group_by(cons.conf.idx,y) %>% summarize(count = n()) %>% arrange(desc(count))
cons_conf
# Barplot of the Consumer Confidence index Distribution
cons_conf %>% 
  ggplot(aes(reorder(cons.conf.idx,count),count)) +
  geom_bar(aes(fill = y),stat = "identity") +
  xlab("Consumer Confidence index") +
  ggtitle("Barplot of the Consumer Confidence index Distribution")
# Plot shows consumer price index of -46.02 subscribes term deposit much followed by -47.1.
# Consumer price index of -40.4 subscribes term deposit in low number
# From the figure one can observe changes in consumer confidence index increases term deposit subscription


# Distribution of Euribor 3 month rate
unique(bank_data$euribor3m)
length(unique(bank_data$euribor3m))
table(bank_data$euribor3m)
# There are 316 different types of euribor 3month rate for clients in the campaign
# Table lists how many clients have same euribor 3month rate in the campaign
eur <- bank_data %>%
  group_by(euribor3m,y) %>% summarize(count = n()) %>% arrange(desc(count))
eur
# Barplot of Euribor 3 month rate Distribution
eur %>% 
  ggplot(aes(reorder(euribor3m,count),count)) +
  geom_bar(aes(fill = y),stat = "identity") +
  xlab("Euribor 3 month rate") +
  ggtitle("Histogram of Euribor 3 month rate Distribution")
# From the plot, one can observe changes in changes in euribor 3month rate increases term deposit subscription
# As well as low euribor rate has low term deposit subscription.

# Distribution of number of employees
unique(bank_data$nr.employed)
length(unique(bank_data$nr.employed))
table(bank_data$nr.employed)
# Table shows how many clients belong to the same number of employees in the campaign.
num_emp <- bank_data %>%
  group_by(nr.employed,y) %>% summarize(count = n()) %>% arrange(desc(count))
num_emp
# Histogram of number of employees Distribution
bank_data %>% 
  ggplot(aes(nr.employed,fill = y)) +
  geom_histogram(stat = "count",bins = 30, color = "black") +
  ggtitle("Histogram of number of employees Distribution")
# From the plot, it is clear that 5099.1 no. of employees subscribes term deposit higher compared to others

# Model Analysis and Results

# To compare different models with their accuracies in predicting term deposit subscription,
# split further bank_data into training and testing sets as bank_train_data & bank_validation_data
# The model which gives highest accuracy is taken and tested against validation set(final holdout test)
# to determine accuracy and to evaluate model performance.

set.seed(1, sample.kind="Rounding")
test_index1 <- createDataPartition(y = bank_data$y, times = 1, p = 0.1, list = FALSE)
bank_train_data <- bank_data[-test_index1,]
bank_temp_data <- bank_data[test_index1,]
# Make sure euribor3m in bank_validation_data set are also in bank_train_data set
bank_validation_data <- bank_temp_data %>%
  semi_join(bank_train_data, by = "euribor3m")
# Bring back removed rows from bank_validation_data to bank_train_data set 
removed <- anti_join(bank_temp_data, bank_validation_data)
bank_train_data <- rbind(bank_train_data, removed)
rm( test_index1, bank_temp_data, removed) #Clear out unwanted files

# Number of observations(rows and columns) in the dataset
dim(bank_train_data)
dim(bank_validation_data)
# There are 33366 rows and 21 columns in training set
# There are 3705 rows and 21 columns in test set

# Summary of the dataset
summary(bank_train_data)

# Regression model
# Data Transformation
# For regression model, the output variable(y) should be numeric for calculation purposes.
# But, bank marketing dataset has factor as a datatype. Factor type is used for classification model
# So, target variable(y) has been converted from factor to numeric using as.numeric() function.
# This is applicable for both training and testing sets(bank_train_data and bank_validation_data)
bank_train_data <- bank_train_data %>% mutate(y = as.numeric(bank_train_data$y=="yes"))
bank_validation_data <- bank_validation_data %>%
  mutate(y = as.numeric(bank_validation_data$y=="yes"))
str(bank_train_data$y)
str(bank_validation_data$y)

# Logistic Regression model

# Since, duration attribute is not known before a call is performed, it should be removed from
# the dataset to have a realistic predictive model.

bank_glm <- glm(y ~. -duration, data = bank_train_data, family = "binomial")
# To remove warnings from the above model, we should remove NAs from dataset
# # We can see which attributes have NAs by summary() function
summary(bank_glm)
# Consumer confidence index and Consumer price index attributes have NAs. 
# So, remove those attributes and fit the model once again and see whether they give a warning.
f_glm <- glm(y ~. -duration -cons.price.idx - cons.conf.idx , data = bank_train_data, family = binomial)
# Now there is no warning.
summary(f_glm)
p_h <- predict(f_glm, newdata = bank_validation_data, type = "response")
y_hat <- factor(ifelse(p_h> 0.5, 1, 0))
c1 <- confusionMatrix( y_hat,as.factor(bank_validation_data$y)) 
acc_log_reg_base <- c1$overall["Accuracy"]
c1$byClass
# Lets tabulate the results
Regression_Accuracy_results <- tibble(Model = "Base Logistic Regression model",
                                      Accuracy = acc_log_reg_base)
Regression_Accuracy_results %>% knitr::kable()
# It produces accuracy of about 0.9053. This model has high sensitivity value(0.9811378)
# 98% sensitivity will identify 90% of clients who subscribes term deposit subscription.
# F1 score is 0.9483899. F1 score > 0.90 is considered as a good model.

# Now lets remove all variables whose p value is >0.05 using step function.
fit=step(f_glm)
# After running step function, 17 attributes are further reduced to 10 attributes
# which primarily contributes in predicting term deposit subscription.
# These attributes are then modeled once again to get better prediction
# The final set of attributes are:
print(fit$call)
fit_final = glm(y ~ default + contact + month + day_of_week + campaign + 
                  pdays + previous + poutcome + emp.var.rate + euribor3m,  
                data = bank_train_data, family = "binomial")
summary(fit_final)
p_hat_glm <- predict(fit_final, bank_validation_data, type="response")
y_hat_gl2 <- factor(ifelse(p_hat_glm > 0.5, 1, 0))
c <- confusionMatrix(y_hat_gl2,as.factor(bank_validation_data$y)) 
acc_log_reg_step <- c$overall["Accuracy"]
c$byClass
# Lets tabulate the results
Regression_Accuracy_results <- bind_rows(Regression_Accuracy_results,
                                         tibble(Model = "Step Logistic Regression model",
                                                Accuracy = acc_log_reg_step))
Regression_Accuracy_results %>% knitr::kable()
# The accuracy is 0.9039136. It has high sensitivity value(0.9808336)
# F1 score is 0.9476778.
# Though this model has 90% accuracy and F1 score(0.947), this model takes 2 hours for execution
# It is computationally time inefficient and step logistic regression's accuracy and F1 score 
# is lower compared to Base Logistic Regression model.

# Classification model
# By rminer package
# 14 rminer classification models

# Data transformation

# Since, regression model converts output variable into numeric, classification model can't work.
# Because, classification model is based on factor levels.
# So, convert numeric to factor data type for target variable(y). It involves two steps:
# 1. Convert numeric to character and 2. Convert character to factor

bank_train_data$y <- ifelse(bank_train_data$y == 1, "yes","no")
str(bank_train_data$y) # character
bank_train_data <- mutate(bank_train_data, y = as.factor(y))
str(bank_train_data$y) # factor

bank_validation_data$y <- ifelse(bank_validation_data$y == 1, "yes","no")
str(bank_validation_data$y) #character
bank_validation_data <- mutate(bank_validation_data, y = as.factor(y))
str(bank_validation_data$y) # factor

# Another transformation:
# Since R cannot handle categorical predictors more than 53 in fit() function for some models,
# check all the attributes factor levels with the help of str() function.
# We found attribute euribor3m is a factor with 316 levels of categories.
# Hence, convert factor to numeric for variable euribor3m in the data set using as.numeric function.

str(bank_train_data)
bank_train_data <- bank_train_data %>% mutate(euribor3m = as.numeric(as.character(euribor3m)))
bank_validation_data <- bank_validation_data %>% mutate(euribor3m = as.numeric(as.character(euribor3m)))
str(bank_train_data$euribor3m)
str(bank_validation_data$euribor3m)
# variable euribor3m of training and validation sets has been converted from factor to numeric type.
# 

bank_data <- bank_data %>% mutate(euribor3m = as.numeric(as.character(euribor3m)))
str(bank_data$euribor3m)
validation <- validation %>% mutate(euribor3m = as.numeric(as.character(euribor3m)))
str(validation$euribor3m)

# In rminer, holdout() splits the data into training and test sets and computes indexes.
# fit() consists of 16 classification and 18 regression methods inside it under the same coherent function structure.
# mmetric function computes classification or regression error metrics.

# 1. NaiveBayes Model
H=holdout(bank_data$y,ratio=2/3)
M=fit(y~., bank_data[H$tr,],model="naiveBayes")
P=predict(M,bank_data[H$ts,])
print("AUC of ROC curve:")
print(mmetric(bank_data$y[H$ts],P,"AUC")[[1]]) # ROC
print(mmetric(bank_data$y[H$ts],P,"CONF")) # Confusion matrix
print("All metrics:")
print(mmetric(bank_data$y[H$ts],P,"ALL"))
print("Accuracy:")
round(mmetric(bank_data$y[H$ts],P,metric="ACC"),2) # Accuracy
# Accuracy is around 86. But attribute duration highly affects the output target 
# (e.g., if duration=0 then y='no'). 
# Since, duration attribute is not known before a call is performed, it should be removed from
# the dataset to have a realistic predictive model.

M1 = fit(y ~. -duration,bank_data[H$tr,], model = "naiveBayes")
P1 = predict(M1,bank_data[H$ts,])
print("AUC of ROC curve:")
auc <- print(mmetric(bank_data$y[H$ts],P1,"AUC")[[1]]) #ROC
print(mmetric(bank_data$y[H$ts],P1,"CONF"))
print("All metrics:")
all_naive <- print(mmetric(bank_data$y[H$ts],P1,"ALL"))
print("F1 score: ")
f1_naive <- print(mmetric(bank_data$y[H$ts],P1,"macroF1"))
acc <- round(mmetric(bank_data$y[H$ts],P1,metric="ACC"),2) # Accuracy
# Lets tabulate the results
Accuracy_results <- tibble(Model = "Naive Bayes Model",
                           Accuracy = acc,
                           AUC = auc,
                           F1_score = f1_naive)
Accuracy_results %>% knitr::kable()
# After removing duration attribute, accuracy is slightly decreased in this model. 
# The overall area under the curve of ROC is around 0.78.
# F1 score is around 69% and accuracy is around 85%. Still can do better.

# 2. KNN(K-Nearest Neighbor) model
H=holdout(bank_data$y,ratio=2/3)
M2 = fit(y ~. -duration,bank_data[H$tr,], model = "kknn")
P2 = predict(M2,bank_data[H$ts,])
print("AUC of ROC:")
auc_knn <- print(mmetric(bank_data$y[H$ts],P2,"AUC")[[1]]) 
print(mmetric(bank_data$y[H$ts],P2,"CONF"))
print("All metrics")
all_knn <- print(mmetric(bank_data$y[H$ts],P2,"ALL"))
print("F1 score: ")
f1_knn <- print(mmetric(bank_data$y[H$ts],P2,"macroF1"))
acc_knn <- round(mmetric(bank_data$y[H$ts],P2,metric="ACC"),2)
# Lets tabulate the results
Accuracy_results <- bind_rows(Accuracy_results,tibble(Model = "KNN Model",
                                                      Accuracy = acc_knn,
                                                      AUC = auc_knn,
                                                      F1_score = f1_knn))
Accuracy_results %>% knitr::kable()
# knn model has improved accuracy to around 88 compared to naive bayes model.
# But F1 score is decreased to around 66% compared to previous one.
# Though this model has increased accuracy, its F1 score and AUC is reduced.

# 3.1 Conditional inference Tree model(cTree)
H=holdout(bank_data$y,ratio=2/3)
M3 = fit(y~.-duration,bank_train_data,model="ctree")
plot(M3@object)
P3=predict(M3,bank_validation_data)
print(mmetric(bank_validation_data$y,P3,"CONF"))
print("AUC of ROC:")
auc_ctree <- print(mmetric(bank_validation_data$y,P3,"AUC")[[1]]) 
print("All metrics:")
all_ctree <- print(mmetric(bank_validation_data$y,P3,"ALL"))
print("F1 score: ")
f1_ctree <- print(mmetric(bank_validation_data$y,P3,"macroF1"))
acc_ctree <- round(mmetric(bank_validation_data$y,P3,metric="ACC"),2) 
# Lets tabulate the results
Accuracy_results <- bind_rows(Accuracy_results,
                              tibble(Model = "cTree Model",
                                     Accuracy = acc_ctree,
                                     AUC = auc_ctree,
                                     F1_score = f1_ctree))
Accuracy_results %>% knitr::kable()
# ctree model greatly improve the accuracy of the model to around 91.55% compared to naivebayes model.
# Also, area under the curve of ROC is 0.94391928
# F1 score is around 77%.

# 3.2 Conditional inference Tree model(cTree) by internal validation
mint=c("kfold",10,123) # internal validation method
s=list(search=mparheuristic("ctree",n=8,lower=0.1,upper=0.99),method=mint,metric = "AUC")
M_heu=fit(y~. -duration,bank_train_data,model="ctree",search=s,fdebug=TRUE)
print("Heurestic parameter object:")
print(M_heu@mpar)
P_heu=predict(M_heu,bank_validation_data)
print(mmetric(bank_validation_data$y,P_heu,"CONF"))
print("AUC of ROC:")
auc_ctreeval <- print(mmetric(bank_validation_data$y,P_heu,"AUC")[[1]]) 
print("All metrics:")
print(mmetric(bank_validation_data$y,P_heu,"ALL"))
print("F1 score: ")
f1_ctreeval <- print(mmetric(bank_validation_data$y,P_heu,"macroF1"))
acc_ctreeval <- round(mmetric(bank_validation_data$y,P_heu,metric="ACC"),2) 
# Lets tabulate the results
Accuracy_results <- bind_rows(Accuracy_results,
                              tibble(Model = "cTree Model by 10-fold internal validation",
                                     Accuracy = acc_ctreeval,
                                     AUC = auc_ctreeval,
                                     F1_score = f1_ctreeval))
Accuracy_results %>% knitr::kable()
# By internal validation, accuracy has been further increased to 91.71 in ctree model. 
# F1 score is slightly improved to around 78.
# AUC of ROC is 0.9434025

# 4. eXtremeGradientBoosting(Tree) model
H=holdout(bank_data$y,ratio=2/3)
M4=fit(y~. -duration,bank_data[H$tr,],model="xgboost",verbose=1)
P4=predict(M4,bank_data[H$ts,]) # nrounds=2, that is default value
print("Overall area under the curve of ROC:")
auc_xgb <- print(mmetric(bank_data[H$ts,]$y,P4,"AUC")) # 0.77782
print(mmetric(bank_data$y[H$ts],P4,"CONF"))
print("F1 score: ")
f1_xgb <- print(mmetric(bank_data$y[H$ts],P4,"macroF1"))
acc_xgb <- round(mmetric(bank_data[H$ts,]$y,P4,metric="ACC"),2) 
# Lets tabulate the results
Accuracy_results <- bind_rows(Accuracy_results,
                              tibble(Model = "eXtremeGradientBoosting model",
                                     Accuracy = acc_xgb,
                                     AUC = auc_xgb,
                                     F1_score = f1_xgb))
Accuracy_results %>% knitr::kable()
# The accuracy of the xgboost model is 89. 
# F1 score is around 63.
# AUC can differ because of the random stochaistic nature of the model.

# To check if accuracy improves, when number of decision trees in the model is changed to 3.
# Lets implement the model below:
M_round = fit(y~. -duration,bank_data[H$tr,],model="xgboost",nrounds=3,verbose=1) # nrounds=3, show rounds:
P_round=predict(M_round,bank_data[H$ts,])
print("Overall area under the curve of ROC:")
auc_nr <- print(mmetric(bank_data[H$ts,]$y,P_round,"AUC")) 
print("F1 score: ")
f1_xgb_round <- print(mmetric(bank_data$y[H$ts],P_round,"macroF1"))
acc_nr <- round(mmetric(bank_data[H$ts,]$y,P_round,metric="ACC"),2)
# Lets tabulate the results
Accuracy_results <- bind_rows(Accuracy_results,
                              tibble(Model = "XGBoost Model(when nrounds = 3)",
                                     Accuracy = acc_nr,
                                     AUC = auc_nr,
                                     F1_score = f1_xgb_round))
Accuracy_results %>% knitr::kable()
# One can see, after changing nrounds parameter as 3 in XGBoost model, 
# there is a very slight improvement of accuracy in the model.
# Also, AUC of ROC is slightly increased but F1 score is decreased to around 63%.

# 5. Support Vector Machine model
H=holdout(bank_data$y,ratio=2/3)
M5=fit(y~. -duration,bank_data[H$tr,],model="ksvm",task="class")
P5=predict(M5,bank_data[H$ts,]) # classes
print(mmetric(bank_data$y[H$ts],P5,"CONF"))
print("All metrics:")
print(mmetric(bank_data$y[H$ts],P5,"ALL"))
print("F1 score: ")
f1_svm <- print(mmetric(bank_data$y[H$ts],P5,"macroF1"))
acc_svm <- round(mmetric(bank_data$y[H$ts],P5,metric="ACC"),2) #90.1
# Lets tabulate the results
Accuracy_results <- bind_rows(Accuracy_results,
                              tibble(Model = "SVM model with classes",
                                     Accuracy = acc_svm,
                                     F1_score = f1_svm))
Accuracy_results %>% knitr::kable()
# SVM model prediction has slight varied accuracy when compared to XGBoost model.
# It has increased F1 score compared to Xgboost models.

# When svm uses probability as a task parameter, does accuracy improves for predicting
# term deposit subscription? It can be implemented as:
M7=fit(y~. -duration,bank_data[H$tr,],model="ksvm",task="prob")
P7=predict(M7,bank_data[H$ts,])
print(mmetric(bank_data$y[H$ts],P7,"CONF"))
print("All metrics:")
print(mmetric(bank_data$y[H$ts],P7,"ALL"))
print("Overall area under the curve of ROC:")
auc_svm_prob <- print(mmetric(bank_data[H$ts,]$y,P7,"AUC")) 
print("F1 score: ")
f1_svm_prob <- print(mmetric(bank_data$y[H$ts],P7,"macroF1"))
acc_svm_prob <- round(mmetric(bank_data$y[H$ts],P7,metric="ACC"),2) #91.22
# Lets tabulate the results
Accuracy_results <- bind_rows(Accuracy_results,
                              tibble(Model = "SVM model with probability",
                                     Accuracy = acc_svm_prob,
                                     AUC = auc_svm_prob,
                                     F1_score = f1_svm_prob))
Accuracy_results %>% knitr::kable()
# The accuracy of the model has very slightly varied when task becomes probability compared to classes.
# F1 score is smaller than the svm with classes.
# AUC of ROC is lesser compared to knn and naive bayes model

# 6. LSSVM model
H=holdout(bank_data$y,ratio=2/3)
M6=fit(y~. -duration,bank_data[H$tr,],model="lssvm") # default task="class" is assumed
P6=predict(M6,bank_data[H$ts,]) # classes
print(mmetric(bank_data$y[H$ts],P6,"CONF"))
print("All metrics:")
print(mmetric(bank_data$y[H$ts],P6,"ALL"))
print("F1 score: ")
f1_lssvm <- print(mmetric(bank_data$y[H$ts],P6,"macroF1"))
acc_lssvm <- round(mmetric(bank_data$y[H$ts],P6,metric="ACC"),2) 
# Lets tabulate the results
Accuracy_results <- bind_rows(Accuracy_results,
                              tibble(Model = "LSSVM model with classes",
                                     Accuracy = acc_lssvm,
                                     F1_score = f1_lssvm))
Accuracy_results %>% knitr::kable()


# 7.1 Random Forest
M8=fit(y~. -duration,bank_train_data,model="randomForest")
P8=predict(M8,bank_validation_data)
print(mmetric(bank_validation_data$y,P8,"CONF"))
print("All metrics:")
print(mmetric(bank_validation_data$y,P8,"ALL"))
print("Overall area under the curve of ROC:")
auc_ran <- print(mmetric(bank_validation_data$y,P8,"AUC")) 
print("F1 score: ")
f1_ran <- print(mmetric(bank_validation_data$y,P8,"macroF1"))
acc_ran <- round(mmetric(bank_validation_data$y,P8,metric="ACC"),2) 
# Lets tabulate the results
Accuracy_results <- bind_rows(Accuracy_results,
                              tibble(Model = "Random Forest",
                                     Accuracy = acc_ran,
                                     AUC = auc_ran,
                                     F1_score = f1_ran))
Accuracy_results %>% knitr::kable()
# Random forest produces accuracy around 90%. 
# AUC is around 0.78 and F1 score is around 69%
# This model is a better model because with improved accuracy, AUC and F1 score is also increased
# compared to svm and knn models.

# 7.2 Tuning Random Forest
# search for mtry and ntree
s=list(smethod="grid",search=list(mtry=c(1,2,3),ntree=c(100,200,500)),
       convex=0,metric="AUC",method=c("kfold",3,12345))
print(s)
M10 = fit(y~. -duration,bank_train_data,model="randomForest",search=s,fdebug=TRUE)
print(M10@mpar)
P10=predict(M10,bank_validation_data)
print("AUC: ")
auc_ran_tune <- print(mmetric(bank_validation_data$y,P10,"AUC"))
print(mmetric(bank_validation_data$y,P10,"CONF"))
print("All metrics: ")
print(mmetric(bank_validation_data$y,P10,"ALL"))
print("F1 score: ")
f1_ran_tune <- print(mmetric(bank_validation_data$y,P10,"macroF1"))
acc_ran_tune <- round(mmetric(bank_validation_data$y,P10,metric="ACC"),2) #90.33
# Lets tabulate the results
Accuracy_results <- bind_rows(Accuracy_results,
                              tibble(Model = "Random Forest Tuning",
                                     Accuracy = acc_ran_tune,
                                     AUC = auc_ran_tune,
                                     F1_score = f1_ran_tune))
Accuracy_results %>% knitr::kable()
# Random forest by tuning slightly improve the model performance by increasing accuracy,AUC
# of ROC and F1 scores compared to standard random forest.

# 8. LDA model
H=holdout(bank_data$y,ratio=2/3)
M11 = fit(y ~. -duration,bank_data[H$tr,], model = "lda")
P11 = predict(M11,bank_data[H$ts,])
print("AUC: ")
auc_lda <- print(mmetric(bank_data$y[H$ts],P11,"AUC")[[1]]) #0.936
print(mmetric(bank_data$y[H$ts],P11,"CONF"))
print("All metrics: ")
print(mmetric(bank_data$y[H$ts],P11,"ALL"))
print("F1 score: ")
f1_lda <- print(mmetric(bank_data$y[H$ts],P11,"macroF1"))
acc_lda <- round(mmetric(bank_data$y[H$ts],P11,metric="ACC"),2) #88.59
# Lets tabulate the results
Accuracy_results <- bind_rows(Accuracy_results,
                              tibble(Model = "LDA model",
                                     Accuracy = acc_lda,
                                     AUC = auc_lda,
                                     F1_score = f1_lda))
Accuracy_results %>% knitr::kable()
# The accuracy of the model is lower compared to random forest and svm model. 
# It is a dimensionality reduction model. Therefore,we can't expect to improve the model quite high. 
# Though accuracy is smaller, F1 scores are higher for LDA model compared to svm and lssvm model. 

# 9. Generalized Linear model(GLM)
H=holdout(bank_data$y,ratio=2/3)
M12 =fit(y~. -duration,bank_data[H$tr,],model="cv.glmnet") #probabilities
P12 = predict(M12,bank_data[H$ts,])
print("AUC: ")
auc_glm <- print(mmetric(bank_data$y[H$ts],P12,"AUC")[[1]]) #0.933
print(mmetric(bank_data$y[H$ts],P12,"CONF"))
print("All metrics: ")
print(mmetric(bank_data$y[H$ts],P12,"ALL"))
print("F1 score: ")
f1_glm <- print(mmetric(bank_data$y[H$ts],P12,"macroF1"))
acc_glm <- round(mmetric(bank_data$y[H$ts],P12,metric="ACC"),2) #89.90
# Lets tabulate the results
Accuracy_results <- bind_rows(Accuracy_results,
                              tibble(Model = "GLM model",
                                     Accuracy = acc_glm,
                                     AUC = auc_glm,
                                     F1_score = f1_glm))
Accuracy_results %>% knitr::kable()
# GLM model slightly improves the accuracy compared to LDA model.
# F1 score is low
# AUC of ROC is higher than random forest and lda.

# To check if GLM model'saccuracy increases by k-fold validation method, lets implement that below:
M14=fit(y~. -duration,bank_data[H$tr,],model="cv.glmnet",nfolds=3) 
plot(M14@object) # show cv.glmnet object
P14=predict(M14,bank_data[H$ts,])
print("AUC: ")
auc_glm_fold <- print(mmetric(bank_data$y[H$ts],P14,"AUC")[[1]])
print(mmetric(bank_data$y[H$ts],P14,"CONF"))
print("All metrics: ")
print(mmetric(bank_data$y[H$ts],P14,"ALL"))
print("F1 score: ")
f1_glm_fold <- print(mmetric(bank_data$y[H$ts],P14,"macroF1"))
acc_glm_fold <- round(mmetric(bank_data$y[H$ts],P14,metric="ACC"),2) #90.88
# Lets tabulate the results
Accuracy_results <- bind_rows(Accuracy_results,
                              tibble(Model = "GLM model tuning",
                                     Accuracy = acc_glm_fold,
                                     AUC = auc_glm_fold,
                                     F1_score = f1_glm_fold))
Accuracy_results %>% knitr::kable()
# It slightly increases the accuracy of the model by tuning.
# There is no big difference between AUCs
# Very slight improvement in F1 scores.


# 10. Multilayer Perceptron model
H=holdout(bank_data$y,ratio=2/3)
M15 =fit(y~. -duration,bank_data[H$tr,],model="mlp")
P15=predict(M15,bank_data[H$ts,])
print("AUC: ")
auc_mlp <- print(mmetric(bank_data$y[H$ts],P15,"AUC")[[1]])
print(mmetric(bank_data$y[H$ts],P15,"CONF"))
print("All metrics: ")
print(mmetric(bank_data$y[H$ts],P15,"ALL"))
print("F1 score: ")
f1_mlp <- print(mmetric(bank_data$y[H$ts],P15,"macroF1"))
acc_mlp <- round(mmetric(bank_data$y[H$ts],P15,metric="ACC"),2) #90.52
# Lets tabulate the results
Accuracy_results <- bind_rows(Accuracy_results,
                              tibble(Model = "MLP model",
                                     Accuracy = acc_mlp,
                                     AUC = auc_mlp,
                                     F1_score = f1_mlp))
Accuracy_results %>% knitr::kable()
# There is no improvement in accuracy compared to GLM and KNN models.
# F1 scores are higher than GLM models
# AUC is lower compared to GLM models


# 11. Mutinom(Logistic Regression) model
H=holdout(bank_data$y,ratio=2/3)
M16 = fit(y ~. -duration,bank_data[H$tr,], model = "multinom")
P16=predict(M16,bank_data[H$ts,])
print("AUC: ")
auc_multi <- print(mmetric(bank_data$y[H$ts],P16,"AUC")[[1]])
print(mmetric(bank_data$y[H$ts],P16,"CONF"))
print("All metrics: ")
print(mmetric(bank_data$y[H$ts],P16,"ALL"))
print("F1 score: ")
f1_multi <- print(mmetric(bank_data$y[H$ts],P16,"macroF1"))
acc_multi<- round(mmetric(bank_data$y[H$ts],P16,metric="ACC"),2) #91.06
# Lets tabulate the results
Accuracy_results <- bind_rows(Accuracy_results,
                              tibble(Model = "Multinomial Logistic Regression model",
                                     Accuracy = acc_multi,
                                     AUC = auc_multi,
                                     F1_score = f1_multi))
Accuracy_results %>% knitr::kable()
# The accuracy of the model in predicting term deposit subscription is around 90%.
# Multinomial Logistic Regression model improved the accuracy quite high compared to GLM and SVM models.
# AUC of ROC is higher(around 0.79) than mlp and random forest models.
# F1_score is around 64%.

# 12. Bagging model
H=holdout(bank_data$y,ratio=2/3)
M17 = fit(y ~. -duration,bank_data[H$tr,], model = "bagging")
P17=predict(M17,bank_data[H$ts,])
print("AUC: ")
auc_bag <- print(mmetric(bank_data$y[H$ts],P17,"AUC")[[1]])
print(mmetric(bank_data$y[H$ts],P17,"CONF"))
print("All metrics: ")
print(mmetric(bank_data$y[H$ts],P17,"ALL"))
print("F1 score: ")
f1_bag <- print(mmetric(bank_data$y[H$ts],P17,"macroF1"))
acc_bag <- round(mmetric(bank_data$y[H$ts],P17,metric="ACC"),2) 
# Lets tabulate the results
Accuracy_results <- bind_rows(Accuracy_results,
                              tibble(Model = "Bagging model",
                                     Accuracy = acc_bag,
                                     AUC = auc_bag,
                                     F1_score = f1_bag))
Accuracy_results %>% knitr::kable()
# The overall area under the curve is around 65.
# Bagging model's AUC of ROC is the least value in this project. 
# F1-score is also lower compared to mlp and multinom models.
# There is no significant difference in accuracy compared to previous models.

# 13. Boosting model
H=holdout(bank_data$y,ratio=2/3)
M18 = fit(y ~. -duration,bank_data[H$tr,], model = "boosting")
P18=predict(M18,bank_data[H$ts,])
print("AUC: ")
auc_boost <- print(mmetric(bank_data$y[H$ts],P18,"AUC")[[1]])
print(mmetric(bank_data$y[H$ts],P18,"CONF"))
print("All metrics: ")
print(mmetric(bank_data$y[H$ts],P18,"ALL"))
print("F1 score: ")
f1_boost <- print(mmetric(bank_data$y[H$ts],P18,"macroF1"))
acc_boost <- round(mmetric(bank_data$y[H$ts],P18,metric="ACC"),2) # 90.94
# Lets tabulate the results
Accuracy_results <- bind_rows(Accuracy_results,
                              tibble(Model = "Boosting model",
                                     Accuracy = acc_boost,
                                     AUC = auc_boost,
                                     F1_score = f1_boost))
Accuracy_results %>% knitr::kable()
# One can see boosting model is not better than bagging model although every step
# of boosting tries to improve the accuracy by identifying the errors. 
# Both AUC of ROC and F1_scores are lower than multinom and mlp models.


# 14. Decision tree model
M19 = fit(y ~. -duration,bank_train_data, model="rpart")
plot(M19@object,uniform=TRUE,branch=0,compress=TRUE) 
text(M19@object,xpd=TRUE,fancy=TRUE,fwidth=0.2,fheight=0.2)
P19=predict(M19,bank_validation_data)
print("AUC: ")
auc_dec <- print(mmetric(bank_validation_data$y,P19,"AUC"))
print(mmetric(bank_validation_data$y,P19,"CONF"))
print("All metrics: ")
print(mmetric(bank_validation_data$y,P19,"ALL"))
print("F1 score: ")
f1_dec <- print(mmetric(bank_validation_data$y,P19,"macroF1"))
acc_dec <- round(mmetric(bank_validation_data$y,P19,metric="ACC"),2) 
mgraph(bank_validation_data$y,P19,graph="ROC",TC=1, main = "No ROC",
       baseline=TRUE,Grid=10,leg = "No")
mgraph(bank_validation_data$y,P19,graph="ROC",TC=2,main = "Yes ROC",
       baseline=TRUE,Grid=10, leg = "Yes")
# Lets tabulate the results
Accuracy_results <- bind_rows(Accuracy_results,
                              tibble(Model = "Decision Tree",
                                     Accuracy = acc_dec,
                                     AUC = auc_dec,
                                     F1_score = f1_dec))
Accuracy_results %>% knitr::kable()
# The accuracy of the decision tree model is around 90%.
# Decision tree model is slightly better than Boosting and GLM model.
# but its AUC of ROC and F1 scores are lower compared to boosting model.


# To check if cross validation of decision tree improves accuracy of the model in
# predicting term deposit subscription, lets implement it.
Mc=crossvaldata(y~. -duration,bank_data,fit,predict,ngroup=10,seed=12345,model="rpart",
                task="prob", control = rpart::rpart.control(cp=0.05))
print("cross validation object:")
print(Mc)
print(mmetric(bank_data$y,Mc$cv.fit,metric="CONF"))
print("AUC: ")
auc_rpart <- print(mmetric(bank_data$y,Mc$cv.fit,"AUC"))
print("F1 score: ")
f1_rpart <- print(mmetric(bank_data$y,Mc$cv.fit,"macroF1"))
print("Accuracy")
acc_rpart_cross <- print(mmetric(bank_data$y,Mc$cv.fit,metric="ACC")) # 10-fold
# Lets tabulate the results
Accuracy_results <- bind_rows(Accuracy_results,
                              tibble(Model = "Decision Tree by cross validation",
                                     Accuracy = acc_rpart_cross,
                                     AUC = auc_rpart,
                                     F1_score = f1_rpart))
Accuracy_results %>% knitr::kable()
# The main objective is to avoid overtraining and overfitting.
# This model doesnot increase accuracy much
# Both AUC and F1 scores are lower than standard decision tree because it minimizes the model error.
# Though standard accuracy produces good accuracy compared to this.

# By summing up the prediction of term deposit subscription by different models, it is concluded 
# that cTree model by internal validation gives best accuracy, F1 score and AUC of the ROC.
# Hence, ctree is used as a final model against validation set to predict term deposit subscription for clients.
# Here, we use the entire *bank_data* dataset for training against *validation*(testing) set.

# Final model against validation by cTree(internal validation)
mint_final=c("kfold",10,123) # internal validation method
s1=list(search=mparheuristic("ctree",n=8,lower=0.1,upper=0.99),method=mint_final,metric = "AUC")
M_heu_final=fit(y~. -duration,bank_data,model="ctree",search=s1,fdebug=TRUE)
print("Heurestic parameter object:")
print(M_heu_final@mpar)
P_heu_final=predict(M_heu_final,validation)
print(mmetric(validation$y,P_heu_final,"CONF"))
print("AUC of ROC:")
final_auc_ctreeval <- print(mmetric(validation$y,P_heu_final,"AUC")[[1]]) 
print("All metrics:")
print(mmetric(validation$y,P_heu_final,"ALL"))
print("F1 score: ")
final_f1_ctreeval <- print(mmetric(validation$y,P_heu_final,"macroF1"))
final_accuracy <- round(mmetric(validation$y,P_heu_final,metric="ACC"),2) 
# Lets tabulate the results
Accuracy_results <- bind_rows(Accuracy_results,
                              tibble(Model = "Final model by cTree(internal validation)",
                                     Accuracy = final_accuracy,
                                     AUC = final_auc_ctreeval,
                                     F1_score = final_f1_ctreeval))
Accuracy_results %>% knitr::kable()
# The final model's accuracy is 91.38%. 
# AUC of ROC and F1 scores are 0.9409127 and 77%


# Visualisation of final model by ROC 
mgraph(validation$y,P_heu_final,graph="ROC",TC=2,main = "Yes ROC",
       baseline=TRUE,Grid=10, leg = "Yes")
mgraph(validation$y,P_heu_final,graph="ROC",TC=1, main = "No ROC",
       baseline=TRUE,Grid=10,leg = "No")


# Hence, ctree by internal validation model is chosen as the best and final model 
# because of its increased accuracy, overall area under the curve and F1 scores compared to other models. 
# Thus, final model uses entire bank_data set against validation set(final holdout test). 
# The accuracy produced by the final ctree  model is 91.38%.
# Hence, it is proven that this model predicts term deposit subscription of clients better than any other models.

